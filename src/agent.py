from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.config import Config


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    search_count: int
    iteration: int
    topic: str


SYSTEM_PROMPT = """你是一个市场调研分析师助手。你的任务是：
1. 围绕用户给定的主题，搜索相关信息
2. 综合分析搜索结果
3. 生成简洁的市场调研简报

在回答时请使用中文，信息要准确、简洁。"""


def create_agent_graph(llm, tools: list, config: Config):
    tool_map = {t.name: t for t in tools}

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = llm.invoke(messages)
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

    def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_messages = []
        for tc in last_message.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
            else:
                tool_messages.append(
                    ToolMessage(content=f"Error: tool '{tc['name']}' not found", tool_call_id=tc["id"])
                )
        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]

        if state.get("iteration", 0) >= config.max_iterations:
            return END

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")

    return graph.compile()


from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
