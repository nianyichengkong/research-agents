"""Error handling tests — behavior: agent stops on critical failures."""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool


class TestToolExecutionError:
    """Agent stops when tool execution raises an exception."""

    def test_stops_on_tool_error(self):
        from src.agent import create_agent_graph
        from src.config import Config

        @tool
        def broken_tool(query: str) -> str:
            """A tool that always fails."""
            raise RuntimeError("API connection failed")

        config = Config(zhipu_api_key="fake", tavily_api_key="fake")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "broken_tool", "args": {"query": "test"}, "id": "c1"}]),
        ]

        graph = create_agent_graph(mock_llm, tools=[broken_tool], config=config)

        with pytest.raises(RuntimeError, match="API connection failed"):
            graph.invoke({"messages": [HumanMessage(content="test")], "search_count": 0, "iteration": 0, "topic": "test"})


class TestLLMError:
    """Agent stops when LLM call raises an exception."""

    def test_stops_on_llm_error(self):
        from src.agent import create_agent_graph
        from src.config import Config

        config = Config(zhipu_api_key="fake", tavily_api_key="fake")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("LLM API unreachable")

        graph = create_agent_graph(mock_llm, tools=[], config=config)

        with pytest.raises(ConnectionError, match="LLM API unreachable"):
            graph.invoke({"messages": [HumanMessage(content="test")], "search_count": 0, "iteration": 0, "topic": "test"})


class TestMissingTool:
    """Agent reports error when LLM calls a non-existent tool."""

    def test_reports_missing_tool(self):
        from src.agent import create_agent_graph
        from src.config import Config

        config = Config(zhipu_api_key="fake", tavily_api_key="fake")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "c1"}]),
            AIMessage(content="工具调用失败：工具不存在"),
        ]

        graph = create_agent_graph(mock_llm, tools=[], config=config)
        result = graph.invoke({"messages": [HumanMessage(content="test")], "search_count": 0, "iteration": 0, "topic": "test"})

        # Tool node returns error message, not crash
        messages = result["messages"]
        tool_msgs = [m for m in messages if hasattr(m, "tool_call_id") and m.tool_call_id]
        assert len(tool_msgs) == 1
        assert "not found" in tool_msgs[0].content
