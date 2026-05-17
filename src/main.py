"""CLI entry point for the market research ReAct agent."""

import sys
import os
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.config import Config
from src.llm import create_llm
from src.agent import create_agent_graph
from src.tools import create_search_tools
from src.output import save_report


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Supports +, -, *, /."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


class ResearchTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ResearchTimeout("调研超时，已强制停止")


def _print_stream_event(event: dict):
    """Print a single stream event with Thought/Action/Observation formatting."""
    for node_name, state_update in event.items():
        if "messages" not in state_update:
            continue
        for msg in state_update["messages"]:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                        print(f"  [Action] {tc['name']}({args_str})")
                elif msg.content:
                    print(f"  [Thought] {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            elif isinstance(msg, ToolMessage):
                content = msg.content
                print(f"  [Observation] {content[:150]}{'...' if len(content) > 150 else ''}")


def run_research(graph, topic: str, config: Config) -> dict:
    """Run research with timeout protection and streaming output."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(config.timeout_seconds)
    try:
        all_messages = [HumanMessage(content=topic)]
        for event in graph.stream(
            {"messages": [HumanMessage(content=topic)], "search_count": 0, "iteration": 0, "topic": topic},
            stream_mode="updates",
        ):
            for node_name, state_update in event.items():
                if "messages" in state_update:
                    for msg in state_update["messages"]:
                        all_messages.append(msg)
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    args_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                                    print(f"  [Action] {tc['name']}({args_str})")
                            elif msg.content:
                                print(f"  [Thought] {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
                        elif isinstance(msg, ToolMessage):
                            content = msg.content
                            print(f"  [Observation] {content[:150]}{'...' if len(content) > 150 else ''}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return {"messages": all_messages}


def main():
    config = Config.load()
    search_tools = create_search_tools(config.tavily_api_key)
    all_tools = search_tools + [calculator]

    llm = create_llm(
        api_key=config.zhipu_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4",
        tools=all_tools,
    )

    graph = create_agent_graph(llm, tools=all_tools, config=config)

    print("=" * 50)
    print("  市场调研助手 (ReAct Agent)")
    print("  输入调研主题开始，输入 exit 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        try:
            result = run_research(graph, user_input, config)
        except ResearchTimeout as e:
            print(f"\n[错误] {e}")
            continue
        except Exception as e:
            print(f"\n[错误] {type(e).__name__}: {e}")
            continue

        if result:
            messages = result.get("messages", [])
            final_message = messages[-1] if messages else None
            if final_message and isinstance(final_message, AIMessage):
                print(f"\n{final_message.content}")
                path = save_report(final_message.content, topic=user_input)
                print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main()
