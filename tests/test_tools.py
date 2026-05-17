"""Tools module tests — behavior: search tools are created and flow through the ReAct loop."""

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage


class TestCreateSearchTools:
    """create_search_tools() returns usable tool instances."""

    @patch("langchain_tavily.TavilySearch")
    def test_returns_two_tools(self, mock_tavily_class):
        mock_tavily_class.return_value = MagicMock(name="tavily_search")
        from src.tools import create_search_tools
        tools = create_search_tools("fake-key")
        assert len(tools) == 2

    @patch("langchain_tavily.TavilySearch")
    def test_search_tool_has_correct_name(self, mock_tavily_class):
        mock_instance = MagicMock()
        mock_instance.name = "tavily_search"
        mock_tavily_class.return_value = mock_instance
        from src.tools import create_search_tools
        tools = create_search_tools("fake-key")
        assert tools[0].name == "tavily_search"

    @patch("langchain_tavily.TavilySearch")
    def test_extract_tool_has_correct_name(self, mock_tavily_class):
        mock_tavily_class.return_value = MagicMock(name="tavily_search")
        from src.tools import create_search_tools
        tools = create_search_tools("fake-key")
        assert tools[1].name == "web_extract"


class TestSearchInGraph:
    """Search results flow through the ReAct loop correctly."""

    def test_graph_calls_search_tool_and_returns_synthesis(self):
        from src.agent import create_agent_graph
        from src.config import Config

        config = Config(zhipu_api_key="fake", tavily_api_key="fake")

        mock_search = MagicMock()
        mock_search.name = "tavily_search"
        mock_search.invoke.return_value = "中国新能源汽车市场：2024年市场规模达到1000万辆"

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "tavily_search", "args": {"query": "2024中国新能源汽车市场"}, "id": "call_1"}]),
            AIMessage(content="根据调研，2024年中国新能源汽车市场规模达到1000万辆。"),
        ]

        graph = create_agent_graph(mock_llm, tools=[mock_search], config=config)
        result = graph.invoke({"messages": [HumanMessage(content="调研新能源汽车市场")], "search_count": 0, "iteration": 0, "topic": "新能源汽车"})

        assert mock_llm.invoke.call_count == 2
        assert "1000万" in result["messages"][-1].content
