"""CLI entry point for the market research ReAct agent."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.config import Config
from src.llm import create_llm
from src.agent import create_agent_graph
from src.tools import create_search_tools


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Supports +, -, *, /."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


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

        result = graph.invoke({
            "messages": [HumanMessage(content=user_input)],
            "search_count": 0,
            "iteration": 0,
            "topic": user_input,
        })

        final_message = result["messages"][-1]
        print(f"\n{final_message.content}")


if __name__ == "__main__":
    main()
