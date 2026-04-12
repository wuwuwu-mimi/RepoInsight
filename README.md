
# RepoInsight Agent

**一个开源的 GitHub 项目智能分析 Agent**：输入任意 GitHub 仓库 URL，即可自动拉取、结构化分析、生成专业报告，并将分析结果持久化存储到向量数据库中，形成你的「个人开源项目知识库」。支持跨多个仓库的语义搜索和问答。

**一句话亮点**：不是一次性分析工具，而是帮开发者构建持久、可检索的开源项目「第二大脑」。


## ✨ 核心功能

- **一键分析**：输入 GitHub URL → 自动克隆（浅克隆）、解析结构、提取元数据、分析代码
- **多维度报告**：项目概述、技术栈、架构图、优势/劣势、潜在风险、贡献建议等（Markdown + JSON）
- **持久化存储**：分析报告 + 代码片段自动嵌入，向量数据库，支持长期记忆
- **语义搜索**：跨所有已分析仓库提问，例如「哪个项目用了 GraphRAG？」或「对比这几个 RAG 项目的技术选型」
- **导出**：Markdown、PDF、JSON 报告
- **本地优先**：支持 Ollama / DeepSeek / Qwen 等本地模型，完全离线运行

## 🚀 为什么要做这个项目（Motivation）

开发者每天刷 GitHub，但分析完 10 个 repo 后就全忘了。现有工具大多只做一次性分析或文档生成，缺少**持久化 + 跨项目 RAG** 能力。

本项目填补这个空白：让分析结果变成可积累、可检索的知识资产。


### 核心框架
- **Agent 编排**：LangGraph（推荐，状态机强大，支持多 Agent 协作、人机交互）
- **LLM**：Ollama（本地）+ Qwen2.5 / DeepSeek-R1 / Grok（可选云端）
- **向量数据库**：Chroma（MVP，最简单）→ Qdrant / LanceDB（生产）
- **嵌入模型**：bge-m3（多语言强）或 sentence-transformers / text-embedding-3-large

### 数据获取与解析
- GitHub API + PyGitHub（元数据、stars、issues）
- gitpython（浅克隆，避免大仓库炸内存）
- tree-sitter + py-tree-sitter（AST 解析，生成依赖图、代码结构）
- 支持语言：Python、JavaScript/TypeScript、Java、Go 等（可扩展）

### 报告生成与处理
- Pydantic（结构化输出）
- Markdown + WeasyPrint / ReportLab（PDF 导出）
- LangChain


### 观测与评估
- Langfuse / LangSmith（tracing）
- RAGAS（评估检索质量）
- 自定义指标（报告完整性、事实准确性）

### 其他
- Docker 支持（一键部署）
- Poetry / uv（依赖管理）

## 📁 项目架构

**核心 Workflow（LangGraph）**：
1. 输入 URL → Metadata Agent
2. 浅克隆 + Structure Agent（tree + 文件过滤，跳过 node_modules、.git 等）
3. Code Analysis Agent（AST + 关键文件总结）
4. Insight Agent（LLM 生成报告）
5. Report Agent（格式化 + 入库）
6. Human-in-the-loop（可选：用户确认/修正）

## 使用示例
```bash
# 分析一个项目
[x]repoinsight analyze https://github.com/langchain-ai/langgraph --model qwen2.5

# 搜索跨项目知识
[]repoinsight search "哪个项目在用 ColPali 做多模态 RAG"

# 导出 PDF
[]repoinsight export --report-id xxx --format pdf
```
