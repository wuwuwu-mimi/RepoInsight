from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    """表示一条可用于检索的知识文档。"""

    # 文档唯一标识，通常由 repo_id、文档类型和路径拼接而成。
    doc_id: str = Field(..., description='文档唯一标识')

    # 所属仓库标识，例如 owner/repo。
    repo_id: str = Field(..., description='所属仓库标识')

    # 文档类型，例如 repo_summary、repo_fact、readme_summary、key_file_summary。
    doc_type: str = Field(..., description='文档类型')

    # 文档标题，便于搜索结果展示。
    title: str = Field(..., description='文档标题')

    # 文档主体文本，会作为后续检索和问答的主要上下文。
    content: str = Field(..., description='文档内容')

    # 关键来源路径；若是仓库级摘要则可为空。
    source_path: str | None = Field(default=None, description='来源路径')

    # 便于后续做结构化过滤的元数据。
    metadata: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict,
        description='文档元数据',
    )


class SearchHit(BaseModel):
    """表示一次检索命中的结果。"""

    # 命中的知识文档。
    document: KnowledgeDocument = Field(..., description='命中文档')

    # 当前检索打分，值越高说明越相关。
    score: float = Field(..., description='相关性得分')

    # 面向终端展示的简短摘要片段。
    snippet: str = Field(..., description='命中摘要片段')


class SearchResult(BaseModel):
    """表示一次查询返回的检索结果。"""

    # 用户原始查询文本。
    query: str = Field(..., description='查询文本')

    # 命中结果列表。
    hits: list[SearchHit] = Field(default_factory=list, description='命中结果')

    # 当前知识库中的仓库总数。
    repo_count: int = Field(default=0, description='仓库总数')

    # 当前知识库中的文档总数。
    document_count: int = Field(default=0, description='文档总数')
