from pydantic import BaseModel, Field


class VectorStoreHealthResult(BaseModel):
    """表示当前向量库的健康检查结果。"""

    # 向量后端名称，当前固定为 chroma。
    backend: str = Field(default='chroma', description='向量后端名称')

    # 向量运行时是否可用，例如 chromadb 是否已安装。
    runtime_available: bool = Field(default=False, description='向量运行时是否可用')

    # 向量库目录是否存在。
    store_exists: bool = Field(default=False, description='向量库目录是否存在')

    # 向量库根目录路径。
    store_path: str = Field(..., description='向量库根目录路径')

    # 当前本地知识库中的仓库数。
    knowledge_repo_count: int = Field(default=0, description='知识库仓库数')

    # 当前本地知识库中的文档数。
    knowledge_document_count: int = Field(default=0, description='知识库文档数')

    # 向量库中已索引的仓库数；探测失败时可能为 0。
    indexed_repo_count: int = Field(default=0, description='向量库中的仓库数')

    # 向量库 collection 中的文档数；探测失败时可能为 0。
    indexed_document_count: int = Field(default=0, description='向量库中的文档数')

    # 当前健康检查是否通过。
    healthy: bool = Field(default=False, description='是否健康')

    # 额外状态说明，适合在 CLI 中直接展示。
    message: str = Field(default='', description='状态说明')

    # 健康检查失败时的错误信息。
    error: str | None = Field(default=None, description='错误信息')


class VectorStoreRebuildResult(BaseModel):
    """表示一次向量库重建操作的结果。"""

    # 向量后端名称，当前固定为 chroma。
    backend: str = Field(default='chroma', description='向量后端名称')

    # 重建前健康检查是否通过。
    healthy_before: bool = Field(default=False, description='重建前是否健康')

    # 是否成功删除了旧向量库目录。
    removed_existing_store: bool = Field(default=False, description='是否删除旧向量库')

    # 是否成功完成重建。
    success: bool = Field(default=False, description='是否重建成功')

    # 本次写入向量库的仓库数。
    indexed_repo_count: int = Field(default=0, description='写入的仓库数')

    # 本次写入向量库的文档数。
    indexed_document_count: int = Field(default=0, description='写入的文档数')

    # 向量库根目录路径。
    store_path: str = Field(..., description='向量库根目录路径')

    # 额外状态说明，适合在 CLI 中直接展示。
    message: str = Field(default='', description='状态说明')

    # 重建失败时的错误信息。
    error: str | None = Field(default=None, description='错误信息')
