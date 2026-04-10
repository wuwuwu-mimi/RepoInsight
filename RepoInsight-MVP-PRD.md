# RepoInsight Agent MVP 需求文档（精简可落地版）

**版本**：v1.1（2026 年 4 月）  
**作者**：wuwuwuwuw + Codex  
**状态**：建议作为首版开发蓝图  
**对应原稿**：`RepoInsight.md`

## 1. 文档目标

这份文档用于将原始 PRD 收敛为一个 **2 周内更有机会真正做完、做稳、做出可验证价值** 的 MVP 版本。

核心原则只有三条：

- 先验证“单仓库分析 + 可复用沉淀”是否成立
- 先做稳定串行流程，不提前引入多 Agent 复杂度
- 先保证结果可验证，再追求炫酷能力

---

## 2. MVP 一句话定义

用户输入一个公开 GitHub 仓库 URL，系统自动完成基础分析，生成结构化 JSON 和可读 Markdown 报告，并把分析结果写入本地知识库，支持后续基于自然语言的简单搜索与检索。

---

## 3. MVP 目标与不做什么

### 3.1 MVP 目标

MVP 只验证以下 4 件事：

1. 能否低成本分析一个公开仓库
2. 能否生成一份对开发者真正有用的报告
3. 能否将结果沉淀到本地知识库
4. 能否基于已分析仓库做基础检索

### 3.2 MVP 明确不做

以下内容 **不进入首版交付范围**：

- 多 Agent 协作
- GraphRAG
- PDF 导出
- Web UI（Streamlit / Gradio）
- GitHub Action 集成
- private repo 支持
- 多平台支持（GitLab / Hugging Face）
- 自动架构图生成
- 完整 AST 级深度代码理解

这些能力可以进入后续版本，但不应该阻塞首版落地。

---

## 4. 目标用户

### 4.1 主要用户

- 独立开发者
- AI / Python 工程师
- 需要做开源项目调研的人

### 4.2 典型场景

- 快速判断一个仓库是否值得深入看
- 给未来的自己沉淀项目分析笔记
- 在多个已分析仓库中搜索特定技术点

---

## 5. 用户故事

### 用户故事 1：快速分析

作为开发者，我希望执行一条命令就能得到一个仓库的概览、技术栈、目录结构和关键模块摘要，这样我不用手动通读整个项目。

### 用户故事 2：长期沉淀

作为开发者，我希望分析结果能被保存，而不是一次性输出后就丢失，这样我后续还能复用这些结论。

### 用户故事 3：跨仓库检索

作为开发者，我希望对已经分析过的仓库进行自然语言搜索，例如“有哪些项目用了向量数据库”。

---

## 6. 核心能力范围

## 6.1 输入

- GitHub public repo URL
- 可选参数：
  - `--branch`
  - `--model`
  - `--max-files`
  - `--output-dir`

### 6.2 分析内容

MVP 仅覆盖以下内容：

- 仓库基础元数据
  - repo 名称
  - owner
  - stars / forks
  - license
  - 默认分支
  - 最后更新时间
  - GitHub 描述
- README 分析
  - 项目用途总结
  - 安装 / 使用方式提取
  - 关键特性提取
- 目录结构分析
  - 树状结构展示
  - 过滤无关目录与二进制文件
- 关键文件摘要
  - `README*`
  - `requirements.txt`
  - `pyproject.toml`
  - `package.json`
  - 入口文件和高信号核心模块
- 技术栈推断
  - 语言
  - 核心依赖
  - 运行方式
  - 推测架构类型

### 6.3 输出内容

- 一份结构化 JSON 报告
- 一份 Markdown 报告
- 一份本地知识库记录

---

## 7. 首版必须具备的 CLI 命令

### 7.1 分析命令

```bash
repoinsight analyze https://github.com/langchain-ai/langgraph
```

功能：

- 克隆或读取仓库
- 生成分析结果
- 输出报告文件
- 写入本地知识库

### 7.2 搜索命令

```bash
repoinsight search "哪些项目使用了 graph database"
```

功能：

- 在本地已分析仓库中检索相关结果
- 返回匹配仓库 + 命中原因 + 证据片段

### 7.3 查看命令

```bash
repoinsight show langgraph
```

功能：

- 查看某个已分析仓库的摘要结果

---

## 8. 报告结构设计

Markdown 报告建议统一为以下结构：

1. 项目概览
2. 仓库元数据
3. 项目用途总结
4. 技术栈推断
5. 目录结构概览
6. 核心文件摘要
7. 架构判断
8. 优势与风险
9. 适用场景
10. 关键证据片段

说明：

- 每个结论尽量附带证据来源
- 对“推断类”结论要明确标注是 inference，而不是事实

---

## 9. 数据模型设计

为避免后续检索质量差，MVP 不只保存文本向量，还需要保存结构化字段。

建议至少定义以下 schema：

```json
{
  "repo_id": "owner/name",
  "url": "https://github.com/owner/name",
  "name": "name",
  "owner": "owner",
  "description": "repo description",
  "stars": 0,
  "forks": 0,
  "license": "Apache-2.0",
  "last_updated": "2026-04-10T00:00:00Z",
  "languages": ["Python"],
  "topics": ["rag", "agent"],
  "tech_stack": ["langgraph", "chromadb", "fastapi"],
  "entrypoints": ["main.py", "app.py"],
  "key_dependencies": ["langchain", "chromadb"],
  "architecture_tags": ["rag", "cli", "agent-workflow"],
  "summary": "short summary",
  "strengths": ["..."],
  "risks": ["..."],
  "evidence_spans": [
    {
      "source": "README.md",
      "quote": "..."
    }
  ]
}
```

### 设计原则

- 向量库负责“语义召回”
- 结构化字段负责“过滤与精确命中”
- 二者结合，才能支持像“哪些项目用了 GraphRAG”这种查询

---

## 10. 知识库存储策略

### 10.1 MVP 方案

- 元数据：本地 JSON / SQLite
- 向量存储：Chroma
- 原始报告：落盘到 `reports/`

### 10.2 检索策略

搜索时采用两阶段：

1. 先基于结构化字段做筛选或粗召回
2. 再基于向量检索找相关证据片段

输出时给出：

- 命中的仓库
- 命中理由
- 相关片段
- 相似度或相关性说明

---

## 11. 技术方案（MVP 版）

### 11.1 推荐栈

- 语言：Python 3.11+
- CLI：Typer
- 数据模型：Pydantic
- Git 操作：GitPython 或直接调用 git
- GitHub API：PyGitHub
- 向量库：Chroma
- 嵌入模型：bge-m3 或兼容 sentence-transformers 的模型
- LLM 接入：
  - 默认本地 Ollama
  - 可选云模型，但默认关闭
- 模板渲染：Jinja2 或直接拼接 Markdown

### 11.2 暂不引入

- LangGraph
- tree-sitter
- Langfuse
- RAGAS

这些工具都很好，但不是 MVP 的阻塞依赖。

---

## 12. 系统流程

```text
输入 URL
  -> 获取 GitHub 元数据
  -> 浅克隆仓库
  -> 过滤文件
  -> 读取 README 和关键文件
  -> 生成结构化中间结果
  -> 调用 LLM 生成摘要/判断
  -> 输出 JSON + Markdown
  -> 写入本地知识库
```

这个流程建议先用 **串行 pipeline** 实现，不引入复杂工作流编排。

---

## 13. 文件过滤规则

MVP 必须优先解决“大仓库读不动”的问题。

### 13.1 默认忽略目录

- `.git`
- `node_modules`
- `dist`
- `build`
- `.next`
- `coverage`
- `__pycache__`
- `venv`
- `.venv`

### 13.2 默认忽略文件类型

- 图片
- 视频
- 压缩包
- 二进制文件
- 超大日志文件
- lock 文件（可选保留摘要）

### 13.3 限制策略

- 单文件大小上限
- 最大文件数限制
- 核心目录优先
- 根目录配置文件优先

---

## 14. 非功能需求

### 14.1 性能

- 普通中小仓库分析时间目标：1 到 3 分钟
- 默认使用浅克隆：`--depth 1`
- 支持大仓库采样分析

### 14.2 隐私

- 默认本地模型
- 默认不上传源码到云模型
- 若用户启用云模型，需明确提示

### 14.3 稳定性

- 任一文件读取失败不应导致整个分析崩溃
- GitHub API 失败时应支持降级逻辑

### 14.4 可维护性

- 代码按模块拆分：ingest、analyze、report、storage、search

---

## 15. 成功标准

MVP 的成功标准不看“功能多不多”，而看“是否稳定可用”。

### 15.1 功能验收

至少能稳定分析 5 个公开仓库，并生成可读报告：

- `langchain-ai/langgraph`
- `run-llama/llama_index`
- `microsoft/autogen`
- `Significant-Gravitas/AutoGPT`
- 任意一个中小型 Python 工具仓库

### 15.2 结果验收

每个报告至少回答清楚以下问题：

1. 这个仓库是干什么的
2. 它主要用了什么技术
3. 入口或核心模块在哪里
4. 它的大致架构是什么
5. 它的潜在优点和风险是什么

### 15.3 搜索验收

给定 10 个固定问题，搜索结果至少能返回合理候选和证据片段。

示例问题：

- 哪些项目使用了 LangGraph
- 哪些项目是 RAG 相关
- 哪些项目包含 Web UI
- 哪些项目重点依赖 Chroma

---

## 16. 风险与缓解

### 风险 1：LLM 结论不稳定

缓解：

- 报告区分“事实”与“推断”
- 所有关键结论尽量附证据
- 降低自由生成比例，多用模板化输出

### 风险 2：检索效果差

缓解：

- 不只存文本 embedding
- 强制保存结构化标签和证据片段
- 搜索采用“字段过滤 + 向量召回”混合模式

### 风险 3：大仓库分析过慢

缓解：

- 浅克隆
- 文件过滤
- 限制文件数和文件大小
- 只优先分析高信号文件

### 风险 4：项目首版过于复杂

缓解：

- 不做多 Agent
- 不做 UI
- 不做 PDF
- 不做 GraphRAG

---

## 17. 开发拆分建议

### Phase A：最小分析链路

- URL 解析
- GitHub 元数据获取
- 仓库浅克隆
- 文件过滤
- README 与关键文件抽取

### Phase B：报告生成

- 结构化 schema
- JSON 输出
- Markdown 输出

### Phase C：知识库存储

- 本地 metadata 存储
- Chroma 向量入库

### Phase D：检索

- `search` 命令
- 返回仓库 + 证据片段

### Phase E：打磨

- 错误处理
- 示例仓库验证
- README 与 quick start

---

## 18. 目录结构建议

```text
repoinsight/
  cli/
  ingest/
  analyze/
  report/
  storage/
  search/
  models/
  utils/
reports/
data/
tests/
```

模块职责：

- `ingest/`：GitHub API、克隆、文件扫描
- `analyze/`：摘要、技术栈推断、结构化中间结果
- `report/`：JSON / Markdown 输出
- `storage/`：metadata + vector store
- `search/`：检索与结果组装
- `models/`：Pydantic schema

---

## 19. 后续版本路线

当 MVP 被验证后，再考虑以下升级：

### v1.2

- Streamlit Web Demo
- 更好的搜索体验
- 分析历史列表

### v1.3

- LangGraph 工作流
- 多 Agent 分工
- tree-sitter 增强解析

### v2.0

- GraphRAG
- 多仓库对比问答
- 团队知识库
- GitHub Action 自动化分析

---

## 20. 最终结论

RepoInsight 的方向是成立的，但首版必须坚持一个原则：

> 先把“分析一个仓库并沉淀为可检索知识”这条主链路做稳，再扩展多 Agent、GraphRAG 和 UI。

如果首版能稳定做到：

- 一条命令分析一个 repo
- 生成有用报告
- 保存到本地知识库
- 支持基础搜索

那么这个项目就已经具备继续扩展的价值。
