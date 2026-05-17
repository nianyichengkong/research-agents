"""Gradio Web UI for the market research ReAct agent."""

import sys
import os
import uuid
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio
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


def init_agent():
    """Initialize agent components."""
    config = Config.load()
    search_tools = create_search_tools(config.tavily_api_key)
    all_tools = search_tools + [calculator]

    llm = create_llm(
        api_key=config.zhipu_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4",
        tools=all_tools,
    )

    db_path = os.environ.get("DB_PATH", "/tmp/research_agent.db")
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = create_agent_graph(llm, tools=all_tools, config=config, checkpointer=checkpointer)
    return graph, config


graph, config = init_agent()


def respond(message, history, thread_id):
    """Handle user message and stream agent response."""
    if not thread_id:
        thread_id = str(uuid.uuid4())

    thread_config = {"configurable": {"thread_id": thread_id}}
    progress_steps = []
    report_content = ""

    try:
        for event in graph.stream(
            {"messages": [HumanMessage(content=message)], "search_count": 0, "iteration": 0, "topic": message},
            config=thread_config,
            stream_mode="updates",
        ):
            for node_name, state_update in event.items():
                if "messages" in state_update:
                    for msg in state_update["messages"]:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    progress_steps.append(f"🔍 搜索: {tc['args'].get('query', tc['args'])}")
                            elif msg.content and not msg.tool_calls:
                                report_content = msg.content
                        elif isinstance(msg, ToolMessage):
                            snippet = str(msg.content)[:100]
                            progress_steps.append(f"📋 获取到信息...")

            if progress_steps:
                status = "\n".join(f"  {s}" for s in progress_steps[-6:])
                yield f"**调研进行中...**\n\n{status}\n\n---\n⏳ 请稍候，正在分析..."

    except Exception as e:
        yield f"❌ 调研出错: {type(e).__name__}: {e}"
        return

    if report_content:
        try:
            path = save_report(report_content, topic=message)
            yield report_content
        except Exception:
            yield report_content
    else:
        yield "⚠️ 未能生成报告，请重试。"


with gradio.Blocks(title="市场调研助手") as demo:
    gradio.Markdown(
        "# 🔍 市场调研助手\n"
        "输入一个调研主题，AI 自动搜索、分析并生成专业级市场调研报告。\n"
        "**模型**: GLM-4 | **搜索**: Tavily | **框架**: LangGraph ReAct"
    )

    thread_state = gradio.State("")

    chatbot = gradio.Chatbot(
        height=500,
    )

    with gradio.Row():
        msg_input = gradio.Textbox(
            placeholder="输入调研主题，如：中国新能源汽车市场",
            show_label=False,
            scale=4,
        )
        submit_btn = gradio.Button("开始调研", variant="primary", scale=1)

    gradio.Markdown(
        "**提示**: 调研约需 30-60 秒。报告完成后可直接复制 Markdown 内容。"
    )

    def handle_submit(message, history, thread_id):
        history = history or []
        history.append({"role": "user", "content": message})
        yield history, "", thread_id

        for response in respond(message, history, thread_id):
            if history and history[-1]["role"] == "assistant":
                history[-1] = {"role": "assistant", "content": response}
            else:
                history.append({"role": "assistant", "content": response})
            yield history, "", thread_id

    msg_input.submit(handle_submit, [msg_input, chatbot, thread_state], [chatbot, msg_input, thread_state])
    submit_btn.click(handle_submit, [msg_input, chatbot, thread_state], [chatbot, msg_input, thread_state])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
