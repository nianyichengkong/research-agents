# PRD: 基于 LangGraph ReAct 架构的市场调研智能体助手

## Problem Statement

用户需要一个能够全自动完成市场调研并生成简报的工具。当前，市场调研需要人工搜索、筛选、整理信息并撰写报告，耗时且低效。用户希望只需输入一个调研主题，智能体就能自动搜索相关信息、综合分析，并输出一份 Markdown 格式的市场调研简报。此外，用户希望能够基于已生成的报告进行多轮追问和补充。

## Solution

构建一个基于 LangGraph 的纯 ReAct 架构智能体助手。智能体接收调研主题后，通过 ReAct 循环（Reasoning → Acting → Observing）自主规划搜索策略、调用 Tavily 搜索工具获取信息、综合分析后生成 Markdown 简报。支持 SQLite 持久化的短期记忆，实现多轮对话追问能力。通过 LangSmith 实现全程可观测。

## User Stories

1. As a 用户, I want to 在命令行输入一个市场调研主题, so that 智能体自动开始调研流程
2. As a 用户, I want to 智能体自动搜索相关信息, so that 我不需要手动查找资料
3. As a 用户, I want to 智能体综合多源信息生成简报, so that 我能快速了解市场概况
4. As a 用户, I want to 报告以 Markdown 文件保存, so that 我可以方便地查看和编辑
5. As a 用户, I want to 报告文件名包含时间戳, so that 我能区分不同时间的调研结果
6. As a 用户, I want to 在报告生成后继续追问, so that 我可以对特定方面深入了解更多
7. As a 用户, I want to 智能体记住之前调研的上下文, so that 追问时能基于已有信息回答
8. As a 用户, I want to 对话历史持久化到 SQLite, so that 重启后仍能继续之前的对话
9. As a 用户, I want to 智能体在搜索失败或 LLM 异常时明确告知失败原因, so that 我知道哪里出了问题
10. As a 用户, I want to 智能体有最大推理轮次限制, so that 不会陷入死循环浪费资源
11. As a 用户, I want to 智能体有搜索次数上限, so that 不会过度消耗搜索 API 额度
12. As a 用户, I want to 智能体有整体超时保护, so that 一次调研不会无限运行
13. As a 用户, I want to 通过 LangSmith 观测智能体的完整思考过程, so that 我能理解和优化智能体行为
14. As a 用户, I want to 智能体在终端实时展示思考/行动/观察过程, so that 我能看到调研进展
15. As a 用户, I want to 在命令行退出对话, so that 我能优雅地结束会话
16. As a 用户, I want to 通过 .env 文件配置 API Key 和参数, so that 敏感信息不硬编码在代码中
17. As a 用户, I want to 使用智谱 GLM-4 作为 LLM, so that 我可以利用已有的 API Key
18. As a 用户, I want to 搜索工具返回清洗过的内容, so that 智能体能高效处理搜索结果
19. As a 用户, I want to 智能体严格模式下遇到关键错误立即停止, so that 不会生成质量不可靠的报告
20. As a 用户, I want to 报告输出到专用的 output 目录, so that 文件管理清晰有序

## Implementation Decisions

### 架构：纯 ReAct 循环

采用 LangGraph StateGraph 构建 ReAct 循环。状态图中包含以下节点：
- **Agent Node**: LLM 推理节点，接收当前状态（主题 + 历史消息），决定下一步行动（调用工具或生成报告）
- **Tool Node**: 工具执行节点，执行 LLM 选定的工具调用并返回结果
- **条件边**: 判断 LLM 输出是否包含工具调用。有工具调用 → 进入 Tool Node；无工具调用 → 结束循环

状态（State）结构：
- `messages`: 消息历史列表（HumanMessage / AIMessage / ToolMessage）
- `search_count`: 已搜索次数计数器
- `topic`: 当前调研主题

### 模块划分

**1. 配置模块 (Config)**
- 集中管理所有配置项：LLM 参数、Tavily API Key、智谱 API Key、安全边界参数
- 从 `.env` 文件加载敏感配置
- 安全边界常量：MAX_ITERATIONS=10, MAX_SEARCHES=6, TIMEOUT_SECONDS=300

**2. LLM 接口模块 (LLM)**
- 使用 langchain_openai.ChatOpenAI 封装智谱 GLM-4
- base_url 指向智谱 OpenAI 兼容接口
- 通过 `.bind_tools()` 绑定工具定义，启用 function calling
- 配置 model name, temperature, max_tokens 等参数

**3. 工具模块 (Tools)**
- **Tavily 搜索工具**: 使用 langchain_community.tools.tavily_search.TavilySearchResults，返回清洗过的搜索结果摘要
- **网页内容提取工具**: 使用 Tavily 的 extract 功能获取网页正文内容，用于深入阅读关键页面
- **计算工具**: 基础数学计算，用于处理市场规模、增长率等数值
- 所有工具通过 @tool 装饰器或 langchain 内置工具注册，统一 bind 到 LLM

**4. ReAct Agent 图模块 (Agent)**
- 使用 LangGraph StateGraph 构建图
- 定义 State TypedDict 包含 messages, search_count, topic
- Agent Node: 调用 LLM，返回 AIMessage（含工具调用或最终文本）
- Tool Node: 路由并执行工具调用，返回 ToolMessage
- 条件边 `should_continue`: 检查 AIMessage 是否含 tool_calls 且未达上限
- 全局超时通过 Python signal 或 threading.Timer 实现

**5. 记忆模块 (Memory)**
- 使用 langgraph.checkpoint.sqlite.SqliteSaver 实现 SQLite 持久化
- 每个 conversation 维护一个 thread_id
- checkpoint 自动保存每轮状态，支持恢复中断的对话
- 支持 list_history 查看历史会话

**6. 错误处理模块 (ErrorHandler)**
- 严格模式：关键步骤失败立即终止并报告错误
- 搜索失败：抛出明确异常，agent 捕获后停止
- LLM 失败：捕获 API 错误（限流、格式错误），停止并提示
- 工具调用格式错误：捕获解析异常，停止并提示
- 所有错误信息通过 LangSmith 记录

**7. 观测模块 (Observability)**
- LangSmith 集成：通过环境变量 LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT 启用
- 控制台实时输出：通过自定义回调或 LangGraph stream 模式，打印每步的 Thought / Action / Observation
- 以不同颜色或前缀区分思考、行动、观察三个阶段

**8. 报告输出模块 (Output)**
- 调研完成后将 LLM 最终输出写入 Markdown 文件
- 文件保存到 output/ 目录，文件名格式：{主题}_{YYYYMMDD_HHMMSS}.md
- 文件名中的主题做安全处理（去除特殊字符，截断长度）

**9. CLI 入口模块 (Main)**
- 命令行交互循环
- 启动时加载配置，初始化 LLM、工具、图、记忆
- 接收用户输入主题，启动 ReAct 调研
- 调研完成后输出文件路径
- 支持追问：用户可在同一会话中继续输入，agent 基于上下文响应
- 支持 exit / quit 命令退出

### LLM 接口适配

智谱 GLM-4 通过 OpenAI 兼容接口调用：
- base_url: https://open.bigmodel.cn/api/paas/v4
- model: glm-4
- 需确认 glm-4 的 function calling 格式与 OpenAI tools 格式完全兼容

### 工具调用流程

1. LLM 返回 AIMessage 包含 tool_calls（function name + arguments）
2. Tool Node 解析 tool_calls，路由到对应工具执行
3. 工具返回结果封装为 ToolMessage
4. ToolMessage 追加到 messages，进入下一轮 Agent Node
5. 搜索工具执行时递增 search_count，达到上限后不再允许搜索调用

## Testing Decisions

### 测试原则
- 只测试模块的外部行为，不测试内部实现细节
- 对 LLM 调用使用 mock，避免消耗真实 API 额度
- 对 Tavily 调用使用 mock，使用预定义的搜索结果作为 fixture

### 需要测试的模块

**ReAct Agent 图模块**
- 测试条件边逻辑：有 tool_calls 时继续，无 tool_calls 时结束
- 测试搜索次数达到上限时强制结束搜索
- 测试最大推理轮次限制生效
- 测试超时保护触发

**工具模块**
- 测试 Tavily 搜索工具的输入输出格式
- 测试计算工具的正确性
- 测试工具调用参数解析

**错误处理模块**
- 测试搜索失败时正确抛出异常
- 测试 LLM 返回格式错误时的处理
- 测试工具调用不存在工具时的处理

**报告输出模块**
- 测试文件名生成格式正确
- 测试 output 目录不存在时自动创建
- 测试特殊字符主题的安全处理

**记忆模块**
- 测试 checkpoint 保存和恢复
- 测试多轮对话的 thread_id 隔离

## Out of Scope

- Web UI（Streamlit / Gradio 界面）
- 多智能体协作（Supervisor / Multi-Agent 模式）
- Plan-and-Execute 架构
- 固定报告模板
- 输出 PDF / Word 格式
- 导出到 Notion / 飞书等平台
- 本地文档分析（PDF / Word 输入）
- 固定数据源 API 接入（统计局、艾瑞等）
- 用户认证和权限管理
- 生产级部署（Docker / API 服务）
- 中文以外的多语言支持

## Further Notes

- 智谱 GLM-4 的 function calling 能力需在开发初期验证，若兼容性不佳可降级为提示词模拟工具调用
- Tavily 免费额度为 1000 次/月，高频使用需考虑升级计划
- 简报型报告无固定模板，LLM 自行组织结构，质量依赖模型能力，可能需要后续调整 system prompt 优化
- 严格错误模式下，调研可能因网络波动频繁中断，后续可考虑增加宽松模式选项
