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


SYSTEM_PROMPT = """你是一位专业的市场调研分析师。你的任务是围绕用户给定的主题，快速生成一份简洁的市场调研简报。

工作流程：
1. 分析用户给定的调研主题，提取关键搜索维度（市场规模、主要玩家、竞争格局、趋势等）
2. 使用 tavily_search 工具搜索相关信息，每次搜索聚焦一个维度
3. 如果搜索结果中有需要深入了解的页面，使用 web_extract 工具提取详细内容
4. 综合所有搜索结果，撰写一份结构清晰的简报

简报格式要求：
- 使用 Markdown 格式
- 包含标题、关键发现、市场概况、主要参与者、趋势与展望
- 数据和信息标注来源
- 语言简洁，重点突出，控制在 500-800 字

注意事项：
- 优先搜索中文信息源
- 对搜索结果中的数据进行交叉验证
- 如果某个维度信息不足，明确标注"该维度信息有限"
- 使用中文撰写报告"""


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
