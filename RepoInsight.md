**RepoInsight Agent 项目需求文档（PRD）**
codex resume 019d7639-e8a4-7e21-bd1a-9dae4b716588  
**版本**：v1.0（2026 年 4 月）  
**作者**：wuwuwuwuw（基于与 Grok 的讨论）  
**状态**：草稿 → 可直接用于开发启动  
**项目目标**：构建一个开源的、Python 驱动的 GitHub 仓库智能分析 Agent，帮助开发者快速理解任意开源项目，并将分析结果持久化形成个人/团队的「开源项目知识库」。

### 1. 项目背景与动机

开发者每天浏览大量 GitHub 仓库，用于学习、选型、复用或贡献。但传统方式存在痛点：
- 手动阅读 README、代码结构耗时长
- 分析完多个项目后，知识碎片化，无法高效跨仓库检索
- 现有工具多为一次性分析或文档生成，缺少**持久化 + 语义搜索**能力

**RepoInsight Agent** 的核心价值：  
输入一个 GitHub URL → 自动多维度分析 → 生成结构化报告 → 将结果嵌入向量数据库 → 支持后续跨项目语义查询（如“对比我分析过的 RAG 项目中，哪个用了 GraphRAG？”）。

这不是又一个“单次代码分析工具”，而是**带长期记忆的开源项目第二大脑**。

### 2. 目标用户

- **主要用户**：开源爱好者、独立开发者、AI 工程师、技术调研者（Python 熟练用户优先）
- **次要用户**：团队技术 Lead、新人 onboarding、研究人员
- **使用场景**：
  - 快速评估一个新库是否值得引入
  - 积累个人技术知识库
  - 对比多个相似项目的技术选型与优劣势
  - 生成贡献建议或 PR 思路

### 3. 核心功能需求（MVP + 后续）

#### 3.1 MVP 功能（Phase 1，目标 1-2 周完成）
- **输入**：GitHub 仓库 URL（支持 public repo）
- **自动分析**：
  - 仓库元数据（stars、forks、license、最后更新、描述、语言分布）
  - README 总结 + 关键部分提取
  - 项目目录结构（树状展示，过滤无关文件如 node_modules、__pycache__ 等）
  - 关键文件内容摘要（requirements.txt / pyproject.toml、main 文件、核心模块等）
- **输出**：
  - 结构化 JSON 报告
  - 美观的 Markdown 报告（包含技术栈推断、架构概述、潜在优势/风险）
  - PDF 导出（可选）
- **持久化**：
  - 将报告、元数据、关键代码片段嵌入向量数据库
  - 支持本地 Chroma 存储
- **检索与问答**：
  - 简单语义搜索：跨已分析仓库进行自然语言查询
- **交互方式**：
  - CLI 主入口（`repoinsight analyze <URL>`）
  - Streamlit / Gradio Web Demo（聊天 + 报告查看 + 搜索）

#### 3.2 增强功能（Phase 2）
- 多 Agent 协作（LangGraph 状态机）：
  - Metadata Agent：读取 GitHub API + README
  - Structure Agent：目录树 + 文件过滤
  - Code Analysis Agent：tree-sitter AST 解析（支持 Python、TS/JS 等）
  - Insight Agent：生成优缺点、风险、贡献建议
  - Report Agent：格式化输出 + 入库
- Human-in-the-loop：用户可审查/修正部分分析结果
- 智能采样：大仓库自动限制分析深度（只分析核心目录）

#### 3.3 进阶功能（Phase 3，后续迭代）
- GraphRAG 支持（实体关系图，代码依赖分析）
- 趋势分析（stars 历史、issue 活跃度）
- 多仓库对比查询
- 集成 GitHub Action（自动分析依赖仓库）
- 支持更多托管平台（GitLab、Hugging Face 等）
- 团队模式（向量库隔离 + 共享）
- 可视化：项目知识图谱、架构图自动生成

### 4. 非功能需求

- **性能**：
  - 小中型仓库（< 500 文件）分析时间 < 3 分钟
  - 大仓库支持浅克隆（--depth 1）+ 智能过滤，避免 OOM
- **隐私与安全**：
  - 本地优先（Ollama + 本地向量库）
  - 不上传代码到第三方（除非用户选择云 LLM）
  - Git token 可选用于 private repo（但 MVP 只支持 public）
- **兼容性**：
  - Python 3.11+
  - 支持主流本地 LLM（Qwen2.5、DeepSeek、Grok 等）
  - 支持多嵌入模型（bge-m3 优先）
- **可扩展性**：
  - 模块化设计，便于添加新 Agent 或语言解析器
- **开源友好**：
  - Apache 2.0 License
  - 完整文档、示例、benchmark
  - Docker 一键部署

### 5. 技术栈推荐

- **Agent 编排**：LangGraph（核心，状态机 + 多 Agent 协作）
- **LLM**：Ollama 本地 + Qwen2.5 / DeepSeek-R1（MVP），可选 Grok / Claude
- **向量数据库**：Chroma（MVP）→ Qdrant / LanceDB
- **嵌入模型**：bge-m3（推荐，多语言强）或 sentence-transformers
- **代码解析**：tree-sitter + py-tree-sitter（AST）、gitpython（克隆）、PyGitHub（API）
- **报告生成**：Pydantic（结构化）、Markdown、WeasyPrint / ReportLab（PDF）
- **UI**：Streamlit / Gradio（快速）→ FastAPI + 前端（进阶）
- **观测**：Langfuse（tracing）
- **评估**：RAGAS + 自定义指标（报告完整性、事实准确性）
- **依赖管理**：Poetry 或 uv
- **部署**：Docker + docker-compose（包含 Ollama + Chroma）

### 6. 项目架构概要

- `agents/`：各角色 Agent（metadata、structure、code_analysis、insight、report）
- `graph/`：LangGraph workflow 定义（状态、节点、边）
- `storage/`：向量数据库 CRUD + 嵌入逻辑
- `parsers/`：tree-sitter、文件过滤器
- `retrieval/`：RAG 查询模块
- `cli/` 与 `web/`：交互入口
- `data/`：本地向量库目录（.gitignore）

核心流程：URL → 浅克隆/API → 多 Agent 流水线 → 结构化报告 → 嵌入存储 → 可检索知识库。

### 7. 竞品分析与差异化

- **OpenBMB/RepoAgent**：主要生成文档，支持本地路径，缺少 URL 一键 + 向量持久化（stars ~937，最后大更新 2024 年底）
- **Code-Analyser (LangGraph)**：支持 URL + 对话 QA，但无完整报告导出与持久化知识库
- **PR-Agent / Tabby 等**：专注代码审查或补全，非仓库整体智能分析
- **通用 RAG 方案**：多为单 repo demo，缺少系统化多 Agent + 跨仓库搜索

**差异化点**：URL 一键 + 结构化多维度报告 + 向量数据库持久化 + 跨项目语义 RAG，这是当前开源中较为空白的组合。

### 8. 成功指标（OKR）

- **MVP 交付**：CLI + Web Demo 可跑通 5 个知名 repo（LangChain、LlamaIndex 等），生成高质量报告
- **开源指标**：首月 200+ stars，README 包含清晰 quick start + 示例报告
- **质量指标**：报告事实准确率 > 85%（人工抽样），检索 Recall@5 > 80%
- **用户反馈**：Issues/PR 活跃，社区有贡献（新语言支持、UI 改进等）

### 9. 风险与缓解

- **大仓库问题**：采用浅克隆 + 文件大小/类型过滤 + 可配置采样
- **LLM 幻觉**：多 Agent 验证 + Human-in-the-loop + RAGAS 评估
- **Token 成本**：本地模型优先，增加缓存机制
- **GitHub API 限流**：使用缓存 + 异步

### 10. 开发路线图

- **Phase 1 (MVP, 1-2 周)**：CLI 分析 + 简单报告 + Chroma 入库 + 基本搜索
- **Phase 2 (2-4 周)**：LangGraph 多 Agent + tree-sitter + Streamlit UI + PDF 导出
- **Phase 3**：GraphRAG、对比查询、GitHub Action 集成、benchmark 完善
- **持续**：文档优化、社区推广（Reddit、X、r/LocalLLaMA 等）

### 11. 附录

- **示例报告结构**（Markdown）：
  - 项目概览
  - 技术栈
  - 架构分析
  - 优势与潜在风险
  - 贡献建议
  - 关键代码片段

- **快速启动命令示例**：
  ```bash
  repoinsight analyze https://github.com/langchain-ai/langgraph --model qwen2.5
  repoinsight search "使用了 GraphRAG 的项目"
  ```

这个需求文档已经足够详细，可以直接作为开发蓝图使用。你可以把它保存为 `docs/PRD.md` 或 `REQUIREMENTS.md`，并在 README 中链接。

**下一步建议**：
- 如果你想立即开始 coding，我可以给你 **LangGraph workflow 的代码 skeleton**
- 或者 **pyproject.toml + 依赖列表**
- 或者 **第一个 Metadata Agent 的完整代码**
- 或者帮你细化某个 Phase 的任务列表

告诉我你现在最想推进哪部分，我们继续往下走！这个项目做出来，绝对有潜力在开源社区获得关注。🚀