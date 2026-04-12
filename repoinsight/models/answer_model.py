from pydantic import BaseModel, Field


class AnswerEvidence(BaseModel):
    """表示一条回答证据。"""

    # 证据所属仓库。
    repo_id: str = Field(..., description='仓库标识')

    # 证据对应的文档类型。
    doc_type: str = Field(..., description='文档类型')

    # 证据来源文件路径；仓库级文档时可为空。
    source_path: str | None = Field(default=None, description='来源路径')

    # 用于终端展示的证据摘要。
    snippet: str = Field(..., description='证据摘要')


class RepoAnswerResult(BaseModel):
    """表示一次仓库问答结果。"""

    # 目标仓库标识。
    repo_id: str = Field(..., description='仓库标识')

    # 用户原始问题。
    question: str = Field(..., description='用户问题')

    # 回答主体。
    answer: str = Field(..., description='回答内容')

    # 当前使用的回答模式，例如 extractive。
    answer_mode: str = Field(default='extractive', description='回答模式')

    # 当前使用的检索后端。
    backend: str = Field(default='local', description='检索后端')

    # 当前是否走了降级路径。
    fallback_used: bool = Field(default=False, description='是否走降级')

    # 当前是否允许尝试调用 LLM。
    llm_enabled: bool = Field(default=True, description='是否启用 LLM')

    # 本次是否实际尝试调用过 LLM。
    llm_attempted: bool = Field(default=False, description='是否尝试调用 LLM')

    # 本次 LLM 调用失败原因；未失败时为空。
    llm_error: str | None = Field(default=None, description='LLM 错误信息')

    # 本次回答引用的证据列表。
    evidence: list[AnswerEvidence] = Field(default_factory=list, description='回答证据')
