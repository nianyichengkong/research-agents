"""CLI entry point for the market research ReAct agent."""

import sys
import os
import signal
import uuid

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
from src.memory import create_checkpointer


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


def run_research(graph, topic: str, config: Config, thread_config: dict) -> dict:
    """Run research with timeout protection and streaming output."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(config.timeout_seconds)
    try:
        all_messages = [HumanMessage(content=topic)]
        for event in graph.stream(
            {"messages": [HumanMessage(content=topic)], "search_count": 0, "iteration": 0, "topic": topic},
            config=thread_config,
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

    db_path = os.environ.get("DB_PATH", "research_agent.db")
    with create_checkpointer(db_path) as checkpointer:
        graph = create_agent_graph(llm, tools=all_tools, config=config, checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": thread_id}}

        print("=" * 50)
        print("  市场调研助手 (ReAct Agent)")
        print(f"  模型: glm-4 | 最大轮次: {config.max_iterations} | 搜索上限: {config.max_searches}")
        print(f"  当前会话: {thread_id[:8]}...")
        print("-" * 50)
        print("  输入调研主题开始调研")
        print("  help  - 查看帮助")
        print("  new   - 新建会话")
        print("  exit  - 退出程序")
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
            if user_input.lower() == "new":
                thread_id = str(uuid.uuid4())
                thread_config = {"configurable": {"thread_id": thread_id}}
                print(f"\n已创建新会话: {thread_id[:8]}...")
                continue
            if user_input.lower() == "help":
                print("\n  使用说明:")
                print("  - 直接输入调研主题（如：中国新能源汽车市场）开始调研")
                print("  - 调研完成后可以继续追问（如：补充比亚迪的竞品分析）")
                print("  - new  开始新会话（清除上下文）")
                print("  - exit 退出程序")
                print(f"  - 报告自动保存到 output/ 目录")
                continue

            try:
                result = run_research(graph, user_input, config, thread_config)
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
