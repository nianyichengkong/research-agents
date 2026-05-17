"""Memory module tests — behavior: SQLite checkpoint persists state across invocations."""

import os
import tempfile
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool


@tool
def dummy_tool(text: str) -> str:
    """Echo text."""
    return text


class TestCheckpointPersistence:
    """Checkpoints save and restore conversation state."""

    def test_saves_and_restores_state(self):
        from src.memory import create_checkpointer
        from src.agent import create_agent_graph
        from src.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = Config(zhipu_api_key="fake", tavily_api_key="fake")

            with create_checkpointer(db_path) as checkpointer:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = AIMessage(content="第一次回答")

                graph = create_agent_graph(mock_llm, tools=[dummy_tool], config=config, checkpointer=checkpointer)

                thread_config = {"configurable": {"thread_id": "thread-1"}}
                graph.invoke(
                    {"messages": [HumanMessage(content="你好")], "search_count": 0, "iteration": 0, "topic": "测试"},
                    config=thread_config,
                )

                # Second invoke on same thread — LLM sees previous history
                mock_llm.invoke.return_value = AIMessage(content="第二次回答")
                graph.invoke(
                    {"messages": [HumanMessage(content="追问")], "search_count": 0, "iteration": 0, "topic": "测试"},
                    config=thread_config,
                )

                # Last LLM call should include messages from first round
                last_call_args = mock_llm.invoke.call_args[0][0]
                history_contents = [m.content for m in last_call_args]
                assert "你好" in history_contents
                assert "第一次回答" in history_contents


class TestThreadIsolation:
    """Different thread_ids have independent conversations."""

    def test_threads_are_isolated(self):
        from src.memory import create_checkpointer
        from src.agent import create_agent_graph
        from src.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = Config(zhipu_api_key="fake", tavily_api_key="fake")

            with create_checkpointer(db_path) as checkpointer:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = AIMessage(content="回答")

                graph = create_agent_graph(mock_llm, tools=[dummy_tool], config=config, checkpointer=checkpointer)

                # Thread 1
                graph.invoke(
                    {"messages": [HumanMessage(content="线程1的消息")], "search_count": 0, "iteration": 0, "topic": "t1"},
                    config={"configurable": {"thread_id": "thread-1"}},
                )

                # Thread 2 — should not see thread 1's messages
                mock_llm.invoke.return_value = AIMessage(content="回答2")
                graph.invoke(
                    {"messages": [HumanMessage(content="线程2的消息")], "search_count": 0, "iteration": 0, "topic": "t2"},
                    config={"configurable": {"thread_id": "thread-2"}},
                )

                # Thread 2's LLM call only sees its own messages
                last_call_args = mock_llm.invoke.call_args[0][0]
                history = [m.content for m in last_call_args]
                assert "线程1的消息" not in history
                assert "线程2的消息" in history
