# RepoInsight

一个面向 GitHub 公共仓库的本地分析与问答工具。

输入仓库 URL，RepoInsight 会自动拉取仓库元数据、克隆代码、扫描关键文件、生成结构化分析报告，并把分析结果写入本地知识库和向量索引，后续可以继续 `search` / `answer`。

- 分析链路可用
- 本地 RAG 可用
- 单仓库问答可用
- 分析侧 / 问答侧多 Agent 编排可用
- LangGraph 编排可切换使用

注意：当前最成熟的分析深度主要集中在 Python、JavaScript / TypeScript。

## 当前能力

- 仓库分析
  - GitHub 公共仓库元数据获取
  - README 获取
  - 本地克隆与缓存
  - 文件扫描、关键文件识别、目录树预览
  - 项目画像、技术栈、项目类型、优势 / 风险分析
  - Markdown / JSON / LLM 上下文 / PDF 报告输出
- RAG 与知识库存储
  - 本地知识文档落盘
  - Chroma 向量索引
  - 本地轻量检索回退
  - 向量索引健康检查、重建、孤儿清理
- 问答
  - `search`：跨已分析仓库搜索
  - `answer`：针对单仓库问答
  - 支持抽取式回答与 LLM 增强回答
  - 支持代码实现类问题的代码级证据追踪
- 多 Agent 编排
  - `answer` 侧已有完整多 Agent 编排
  - `analyze` 侧已有 `planner_agent`、动态任务裁剪、任务卡片、并行波次雏形
  - 支持 `local` / `langgraph` 两种编排器
- 本地模型支持
  - Embedding：`service` / `ollama` / `sentence-transformers`
  - LLM：兼容服务商 API，也支持本地 Ollama

## 当前状态

截至 2026-04-17，本地最近一次全量测试结果：

- `python -B -m pytest -q`
- `122 passed, 1 warning`

这说明当前 CLI 主链路已经稳定，可继续正常使用和迭代。

## 还没完成的部分

- Web UI
- GraphRAG / 图数据库
- 多仓库对比查询
- GitHub Action 集成
- 更多托管平台支持
- 更深的多语言专项分析

所以，当前更适合把 RepoInsight 理解为：

- 一个可正常使用的 CLI 分析 / 检索 / 问答工具
- 而不是一个已经全部完成的最终产品

## 环境要求

- Python `>= 3.14`

项目当前 `pyproject.toml` 中声明的是 Python 3.14 及以上，请尽量按这个版本运行。

## 安装

推荐先创建虚拟环境，再安装依赖：

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .
```

如果你使用 `uv`，也可以按自己的习惯安装。

## 配置

复制环境变量模板：

```bash
copy .env.example .env
```

你至少应该修改这两类配置：

- Embedding 配置
  - `REPOINSIGHT_EMBEDDING_PROVIDER`
  - `REPOINSIGHT_EMBEDDING_MODEL`
  - `REPOINSIGHT_EMBEDDING_BASE_URL`
  - `REPOINSIGHT_EMBEDDING_API_KEY`
- LLM 配置
  - `REPOINSIGHT_LLM_PROVIDER`
  - `REPOINSIGHT_LLM_MODEL`
  - `REPOINSIGHT_LLM_BASE_URL`
  - `REPOINSIGHT_LLM_API_KEY`

本地 Ollama 也已支持，例如：

```env
REPOINSIGHT_LLM_PROVIDER=ollama
REPOINSIGHT_LLM_MODEL=qwen2.5:7b
REPOINSIGHT_LLM_BASE_URL=http://127.0.0.1:11434
REPOINSIGHT_LLM_API_KEY=ollama
```

## 快速开始

### 1. 分析一个仓库

```bash
python main.py analyze https://github.com/langchain-ai/langgraph
```

可选：

```bash
python main.py analyze https://github.com/langchain-ai/langgraph --orchestrator local
python main.py analyze https://github.com/langchain-ai/langgraph --orchestrator langgraph
python main.py analyze https://github.com/langchain-ai/langgraph --embedding-mode ollama
python main.py analyze https://github.com/langchain-ai/langgraph --no-save-report
```

### 2. 搜索已分析知识

```bash
python main.py search "哪些项目用了 FastAPI"
```

### 3. 针对单仓库提问

```bash
python main.py answer langchain-ai/langgraph "这个项目是做什么的？"
```

实现类问题示例：

```bash
python main.py answer langchain-ai/langgraph "路由注册是怎么实现的？" --no-llm
```

### 4. 管理本地知识库与向量索引

```bash
python main.py list
python main.py remove langchain-ai/langgraph
python main.py remove-vector langchain-ai/langgraph
python main.py rebuild-vector
python main.py vector-health
python main.py embedding-health
python main.py cleanup-orphans
```

### 5. 导出报告

```bash
python main.py export langchain-ai/langgraph --format pdf
```

## 主要命令

- `analyze`：分析仓库并生成报告
- `search`：跨知识库搜索
- `answer`：针对单仓库问答
- `list`：列出已缓存仓库
- `remove`：删除本地仓库缓存，可选删除关联报告
- `remove-vector`：仅删除向量索引中的仓库数据
- `rebuild-vector`：从本地知识文档重建向量库
- `vector-health`：检查向量库状态
- `embedding-health`：检查 embedding 服务状态
- `cleanup-orphans`：清理孤儿报告 / 知识文档 / 向量索引
- `export`：导出报告
- `version`：查看版本

## 多 Agent 说明

### analyze 侧

当前分析链路已经支持这些角色：

- `planner_agent`
- `repo_agent`
- `readme_agent`
- `structure_agent`
- `codebase_agent`
- `profile_agent`
- `insight_agent`
- `verifier_agent`
- `memory_agent`

并且已经具备：

- 动态任务裁剪
- 任务卡片
- 角色依赖
- 并行波次元数据
- `local` / `langgraph` 双编排器

### answer 侧

当前问答链路已支持：

- `router_agent`
- `retrieval_agent`
- `code_agent`
- `synthesis_agent`
- `verifier_agent`
- `recovery_agent`
- `revision_agent`

## 当前推荐使用方式

如果你是第一次使用，建议走这条路径：

1. `python main.py analyze <GitHub URL>`
2. `python main.py answer <owner/repo> "这个项目是做什么的？" --no-llm`
3. `python main.py search "你的问题"`
4. 再按需切换：
   - `--embedding-mode ollama`
   - `--orchestrator langgraph`
   - LLM 配置