<div align="center">

# 🔍 Research Agent

### 基于 LangGraph ReAct 架构的 AI 市场调研分析师

**输入一个主题，30 秒出一份专业级市场调研报告**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![GLM-4](https://img.shields.io/badge/LLM-GLM--4-4A90D9?style=flat-square)](https://open.bigmodel.cn/)
[![Tavily](https://img.shields.io/badge/Search-Tavily-FF6B6B?style=flat-square)](https://tavily.com/)
[![Tests](https://img.shields.io/badge/Tests-24%20passed-brightgreen?style=flat-square)](./tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)

</div>

---

## ✨ 它能做什么

给它一个调研主题，它会 **自主搜索 → 交叉验证 → 综合分析 → 生成报告**，全程无需人工干预。

```
> 中国储能市场

  [Action] tavily_search(中国储能市场规模 增长率 2024 2025)
  [Observation] 2024年市场规模达2233亿美元，CAGR 25.4%...
  [Action] tavily_search(中国储能竞争格局 头部企业 市场份额)
  [Observation] 宁德时代25-30%，比亚迪10-15%...
  [Action] tavily_search(中国储能技术趋势 锂电池 液流电池)
  ...搜索 6 个维度，交叉验证关键数据...

  📄 报告已保存: output/中国储能市场_20260518_014734.md
```

**输出效果：** 一份 1500-3000 字的 Markdown 报告，包含执行摘要、市场概况、竞争格局表格、消费者洞察、趋势展望和数据来源附录。

---

## 🏗️ 架构设计

```
                        ┌─────────────────────────────────────────┐
                        │           用户输入调研主题               │
                        └────────────────┬────────────────────────┘
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │         Agent Node (GLM-4)              │
                        │                                         │
                        │  · 研究规划：拆解 3-6 个调研维度         │
                        │  · 决策：调用工具 or 生成报告             │
                        └────────────────┬────────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │   有 tool_calls？     │
                              └──────────┬──────────┘
                           Yes │                    │ No
                    ┌──────────▼───┐         ┌────▼─────────────────┐
                    │  Tool Node   │         │  输出结构化报告        │
                    │              │         │  保存为 Markdown 文件  │
                    │ · tavily_search│        └──────────────────────┘
                    │ · web_extract  │
                    │ · calculator   │
                    └──────┬────────┘
                           │
                           ▼
                    回到 Agent Node
                    （最多 15 轮 / 8 次搜索）
```

**核心设计理念：**

| 设计决策 | 选择 | 理由 |
|---------|------|------|
| 架构模式 | 纯 ReAct 循环 | LLM 自主决策搜索策略，无需预定义流程 |
| 搜索工具 | Tavily API | 返回清洗过的内容，专为 AI Agent 设计 |
| 记忆持久化 | SQLite Checkpoint | 重启不丢对话，支持多轮追问 |
| 观测性 | LangSmith + 控制台流 | 全链路追踪 + 实时进度展示 |
| 错误处理 | 严格模式 | 失败即停，保证报告质量 |

---

## 🔥 专业级报告的秘密

不是简单的"搜索 + 摘要"。Agent 内置了六阶段专业调研流程：

```
  阶段 1 ──► 研究规划    拆解 3-6 个调研维度（市场规模、竞争、政策...）
  阶段 2 ──► 维度搜索    每次搜索精确瞄准一个维度，不盲目搜索
  阶段 3 ──► 交叉验证    核心数据从多个角度验证，差异大时标注范围
  阶段 4 ──► 来源标注    A级(政府) B级(研究机构) C级(媒体) D级(博客)
  阶段 5 ──► 质量自检    检查维度覆盖度、数据来源、信息空缺
  阶段 6 ──► 结构化输出  严格遵循六段式报告模板
```

**报告结构：**

```
# {主题} 市场调研报告
├── 一、执行摘要           ← 3-5 句话核心结论
├── 二、市场概况
│   ├── 2.1 市场规模与增长  ← 金额、CAGR、预测
│   └── 2.2 市场驱动因素
├── 三、竞争格局
│   ├── 3.1 主要参与者      ← 企业对比表格
│   └── 3.2 市场集中度
├── 四、消费者洞察         ← 人群画像、行为特征
├── 五、趋势与展望
│   ├── 5.1 短期趋势（1-2年）
│   └── 5.2 中长期趋势（3-5年）
├── 六、结论与建议         ← 可执行建议 + 风险提示
└── 附录：数据来源         ← 编号来源 + 可信度等级
```

---

## 📦 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/nianyichengkong/research-agents.git
cd research-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 填入你的 Key：

```env
ZHIPU_API_KEY=your_zhipu_api_key        # 智谱 GLM-4 → https://open.bigmodel.cn/
TAVILY_API_KEY=your_tavily_api_key       # Tavily 搜索 → https://tavily.com/
LANGCHAIN_API_KEY=your_langsmith_key     # LangSmith 观测（可选）
```

### 3. 运行

```bash
python src/main.py
```

```
==================================================
  市场调研助手 (ReAct Agent)
  模型: glm-4 | 最大轮次: 15 | 搜索上限: 8
  当前会话: 51da6cdb...
--------------------------------------------------
  输入调研主题开始调研
  help  - 查看帮助
  new   - 新建会话
  exit  - 退出程序
==================================================

> 中国新能源汽车市场
```

报告自动保存到 `output/` 目录。

---

## 📸 实际输出示例

以下均为 Agent 自主搜索生成的真实报告，未做任何人工修改：

| 调研主题 | 报告大小 | 搜索次数 |
|---------|---------|---------|
| 中国储能市场 | 9.6 KB | 6 次 |
| 中国宠物经济 | 2.7 KB | 4 次 |
| 中国奶茶市场 | 3.1 KB | 3 次 |

> **储能报告亮点：** 覆盖了市场规模（2233亿美元/CAGR 25.4%）、8 家企业竞争表格（宁德时代 25-30%）、消费者画像（电源侧/电网侧/负荷侧）、短期/长期趋势分析、风险提示和 3 类投资建议。数据来源标注了 9 个引用，包含 A/B/C 三个可信度等级。

---

## 🛡️ 安全机制

```
┌─────────────────────────────────────────────┐
│              安全边界保护                      │
│                                              │
│  最大推理轮次：15 轮      防止死循环           │
│  最大搜索次数：8 次        控制 API 消耗       │
│  整体超时：600 秒          防止无限运行         │
│                                              │
│  严格错误模式：                              │
│  · 搜索 API 失败 → 立即停止，报告错误         │
│  · LLM API 失败  → 立即停止，报告错误         │
│  · 工具调用异常  → 立即停止，报告错误         │
│  · 不会生成质量不可靠的报告                    │
└─────────────────────────────────────────────┘
```

---

## 🧪 测试

```bash
pytest tests/ -v
```

```
24 tests passed in 0.42s

tests/test_agent.py         4 passed  ← ReAct 循环逻辑
tests/test_config.py         4 passed  ← 配置加载
tests/test_error_handler.py  3 passed  ← 错误处理
tests/test_llm.py            2 passed  ← LLM 接口
tests/test_memory.py         2 passed  ← SQLite 记忆
tests/test_output.py         5 passed  ← 报告输出
tests/test_tools.py          4 passed  ← 搜索工具
```

---

## 📂 项目结构

```
research-agents/
├── src/
│   ├── agent.py       # ReAct 循环 + 系统提示词（核心）
│   ├── config.py      # 配置管理（.env + 安全边界）
│   ├── llm.py         # LLM 接口（GLM-4 OpenAI 兼容）
│   ├── tools.py       # 搜索工具（Tavily + 网页提取）
│   ├── output.py      # 报告保存（Markdown 文件）
│   ├── memory.py      # SQLite 记忆持久化
│   └── main.py        # CLI 入口
├── tests/             # 24 个单元测试
├── output/            # 生成的报告（自动创建）
├── PRD.md             # 产品需求文档
├── requirements.txt
└── .env.example
```

---

## 🔮 Roadmap

- [ ] 支持更多 LLM（DeepSeek、Qwen、Claude）
- [ ] 固定报告模板（可选行业模板）
- [ ] Web UI（Streamlit/Gradio）
- [ ] 导出 PDF / Word 格式
- [ ] 多智能体协作（Planner + Researcher + Writer）
- [ ] 数据可视化（自动生成图表）

---

## 📄 License

MIT License

---

<div align="center">

**如果这个项目对你有帮助，给个 ⭐ Star 吧！**

</div>
