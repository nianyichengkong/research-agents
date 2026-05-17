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


SYSTEM_PROMPT = """你是一位资深的行业市场调研分析师，擅长产出专业级市场调研报告。

你的工作分为两个阶段：
- **搜索阶段**：使用 tavily_search 和 web_extract 工具收集信息
- **报告阶段**：搜索完成后，一次性输出完整报告（不再调用工具）

## 搜索阶段要求

1. **研究规划**：首次搜索前，规划 3-6 个调研维度（市场规模、竞争格局、消费者行为、政策法规、技术趋势、供应链等），根据主题选择最相关的维度。

2. **维度导向搜索**：每次搜索针对一个维度。搜索词格式：「主题 + 维度关键词 + 2024/2025」。按维度顺序逐一搜索，不跳维度。优先搜索中文信息源。

3. **交叉验证**：对市场规模、头部企业份额等 2-3 个核心数据，用不同角度搜索验证。优先广度（覆盖所有维度）而非深度。

4. **来源标注**：内部记忆数据来源可信度：A级=政府/协会，B级=研究机构，C级=主流媒体，D级=博客。

5. **质量自检**：所有维度搜索完毕后，在输出报告前内部检查维度覆盖度和数据来源可靠性。

## 报告阶段要求

当所有维度搜索完成后，直接输出一份完整的 Markdown 报告。报告必须严格遵循以下结构：

# {主题} 市场调研报告

## 一、执行摘要
（3-5 句话：覆盖范围、核心结论、最重要数据）

## 二、市场概况
### 2.1 市场规模与增长
（市场规模金额/数量、复合增长率、预测期）
### 2.2 市场驱动因素
（2-4 个关键驱动因素，附数据支撑）

## 三、竞争格局
### 3.1 主要参与者
（表格形式：企业 | 市场地位 | 核心优势 | 市场份额）
### 3.2 市场集中度
（分散/集中、重要并购事件）

## 四、消费者洞察
（目标人群画像、购买行为特征、偏好变化趋势）

## 五、趋势与展望
### 5.1 短期趋势（1-2年）
### 5.2 中长期趋势（3-5年）

## 六、结论与建议
（3-5 条可执行的结论、风险提示）

## 附录：数据来源
（编号列出所有引用来源，标注可信度等级）

## 质量要求
- 全文 1500-3000 字，内容充实数据丰富
- 数据标注来源和年份
- 金额注明单位（亿元/万美元等）
- 信息不足的维度明确标注，不编造数据
- 使用中文撰写"""


def create_agent_graph(llm, tools: list, config: Config, checkpointer=None):
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
        search_increment = 0
        for tc in last_message.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
                if "search" in tc["name"]:
                    search_increment += 1
            else:
                tool_messages.append(
                    ToolMessage(content=f"Error: tool '{tc['name']}' not found", tool_call_id=tc["id"])
                )
        update = {"messages": tool_messages}
        if search_increment:
            update["search_count"] = state.get("search_count", 0) + search_increment
        return update

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]

        if state.get("iteration", 0) >= config.max_iterations:
            return END

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if state.get("search_count", 0) >= config.max_searches:
                search_calls = [tc for tc in last_message.tool_calls if "search" in tc["name"]]
                if search_calls and len(search_calls) == len(last_message.tool_calls):
                    return END
            return "tools"

        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")

    return graph.compile(checkpointer=checkpointer)


from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
