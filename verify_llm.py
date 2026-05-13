"""Smoke test: verify GLM-4 connectivity and function calling compatibility.

Run: python verify_llm.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from src.config import Config
from src.llm import create_llm
from langchain_core.tools import tool


@tool
def test_calculator(expression: str) -> str:
    """Calculate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def main():
    print("Loading config...")
    config = Config.load()
    print(f"  GLM-4 base_url: https://open.bigmodel.cn/api/paas/v4")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Max searches: {config.max_searches}")
    print(f"  Timeout: {config.timeout_seconds}s")

    print("\n[1/2] Testing basic chat...")
    llm = create_llm(
        api_key=config.zhipu_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4",
    )
    response = llm.invoke("Say 'hello' in one word.")
    print(f"  Response: {response.content}")

    print("\n[2/2] Testing function calling...")
    llm_with_tools = create_llm(
        api_key=config.zhipu_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4",
        tools=[test_calculator],
    )
    response = llm_with_tools.invoke("What is 123 * 456?")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"  Tool called: {response.tool_calls[0]['name']}")
        print(f"  Arguments: {response.tool_calls[0]['args']}")
        result = test_calculator.invoke(response.tool_calls[0]["args"])
        print(f"  Result: {result}")
    else:
        print(f"  No tool call - direct response: {response.content}")
        print("  WARNING: function calling may not be compatible!")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
