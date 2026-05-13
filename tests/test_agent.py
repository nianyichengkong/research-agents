"""Agent graph tests — behavior: ReAct loop reasons, acts, and stops correctly."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestAgentGraph:
    """create_agent_graph() returns a compiled graph that runs the ReAct loop."""

    def test_finishes_when_llm_returns_text_without_tool_calls(self):
        """Graph ends immediately when LLM gives a direct text answer."""
        from src.agent import create_agent_graph
        from src.config import Config

        config = Config(
            zhipu_api_key="fake",
            tavily_api_key="fake",
            max_iterations=10,
            max_searches=6,
            timeout_seconds=300,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="答案是42。")

        graph = create_agent_graph(mock_llm, tools=[], config=config)
        result = graph.invoke({"messages": [HumanMessage(content="测试")], "search_count": 0, "iteration": 0, "topic": "测试"})

        assert len(result["messages"]) == 2
        assert result["messages"][-1].content == "答案是42。"

    def test_calls_tool_then_finishes(self):
        """Graph calls tool when LLM requests it, then finishes when LLM returns text."""
        from src.agent import create_agent_graph
        from src.config import Config
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        config = Config(zhipu_api_key="fake", tavily_api_key="fake")

        mock_llm = MagicMock()
        # First call: LLM wants to call tool
        mock_llm.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}],
            ),
            # Second call: LLM gives final answer
            AIMessage(content="1加2等于3。"),
        ]

        graph = create_agent_graph(mock_llm, tools=[add], config=config)
        result = graph.invoke({"messages": [HumanMessage(content="1+2=?")], "search_count": 0, "iteration": 0, "topic": "测试"})

        assert mock_llm.invoke.call_count == 2
        final = result["messages"][-1]
        assert "3" in final.content

    def test_stops_at_max_iterations(self):
        """Graph forces stop when max iterations reached."""
        from src.agent import create_agent_graph
        from src.config import Config
        from langchain_core.tools import tool

        @tool
        def echo(text: str) -> str:
            """Echo text."""
            return text

        config = Config(zhipu_api_key="fake", tavily_api_key="fake", max_iterations=3)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "loop"}, "id": "call_1"}]),
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "loop"}, "id": "call_2"}]),
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "loop"}, "id": "call_3"}]),
        ]

        graph = create_agent_graph(mock_llm, tools=[echo], config=config)
        result = graph.invoke({"messages": [HumanMessage(content="loop test")], "search_count": 0, "iteration": 0, "topic": "测试"})

        assert mock_llm.invoke.call_count == config.max_iterations
