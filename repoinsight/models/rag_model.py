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


class ConfigSummary(BaseModel):
    """表示一个面向多语言仓库的配置摘要。"""

    # 关联仓库标识。
    repo_id: str = Field(..., description='仓库标识')

    # 来源配置文件路径。
    source_path: str = Field(..., description='来源配置文件路径')

    # 配置类型，例如 package_manager、build、deploy、ci。
    config_kind: str = Field(..., description='配置类型')

    # 作用语言或生态范围，例如 python、nodejs、java、go、rust、generic。
    language_scope: str = Field(..., description='语言范围')

    # 从该配置中识别出的框架。
    frameworks: list[str] = Field(default_factory=list, description='识别出的框架')

    # 从该配置中识别出的包管理器。
    package_managers: list[str] = Field(default_factory=list, description='识别出的包管理器')

    # 从该配置中识别出的构建工具。
    build_tools: list[str] = Field(default_factory=list, description='识别出的构建工具')

    # 从该配置中识别出的测试工具。
    test_tools: list[str] = Field(default_factory=list, description='识别出的测试工具')

    # 从该配置中识别出的部署工具。
    deploy_tools: list[str] = Field(default_factory=list, description='识别出的部署工具')

    # 配置中直接提炼出的关键结论，适合后续检索与问答复用。
    key_points: list[str] = Field(default_factory=list, description='配置关键结论')

    # 配置里暴露出的脚本、命令或运行提示。
    scripts_or_commands: list[str] = Field(default_factory=list, description='脚本或命令')

    # 从配置中识别出的数据库、缓存、消息队列等外部服务依赖。
    service_dependencies: list[str] = Field(default_factory=list, description='外部服务依赖')

    # 配置中出现的环境变量名。
    env_vars: list[str] = Field(default_factory=list, description='环境变量')

    # 与当前配置直接相关的其他路径，例如扩展配置、工作目录或挂载路径。
    related_paths: list[str] = Field(default_factory=list, description='相关路径')

    # 当前配置所属的子项目根目录；单仓库时通常为 .
    subproject_root: str | None = Field(default=None, description='所属子项目根目录')

    # 子项目标签，例如 monorepo_workspace、service、package。
    subproject_markers: list[str] = Field(default_factory=list, description='子项目标记')

    # 从当前配置文件中抽取到的关键符号，尽量带上类型和行号。
    code_symbols: list[str] = Field(default_factory=list, description='关键符号')

    # 从当前配置文件中抽取到的模块依赖，尽量带上关系类型和行号。
    module_relations: list[str] = Field(default_factory=list, description='模块依赖关系')

    # 归纳后的配置摘要文本。
    summary: str = Field(..., description='配置摘要')

    # 支撑该摘要的证据列表。
    evidence: list[str] = Field(default_factory=list, description='配置证据')


class EntrypointSummary(BaseModel):
    """表示一个面向多语言仓库的入口摘要。"""

    # 关联仓库标识。
    repo_id: str = Field(..., description='仓库标识')

    # 来源入口文件路径。
    source_path: str = Field(..., description='来源入口文件路径')

    # 入口类型，例如 cli、web_api、frontend、worker、library、workflow。
    entrypoint_kind: str = Field(..., description='入口类型')

    # 作用语言或生态范围，例如 python、nodejs、java、go、rust、generic。
    language_scope: str = Field(..., description='语言范围')

    # 对入口职责的简要概括。
    responsibility: str = Field(..., description='入口职责')

    # 启动提示，例如命令、脚本或运行线索。
    startup_hints: list[str] = Field(default_factory=list, description='启动提示')

    # 可直接尝试的启动命令或调用方式。
    startup_commands: list[str] = Field(default_factory=list, description='启动命令')

    # 入口关联到的核心组件名称或路径。
    related_components: list[str] = Field(default_factory=list, description='关联组件')

    # 与入口协同工作的配置文件。
    dependent_configs: list[str] = Field(default_factory=list, description='依赖配置')

    # 从入口文件中推断出的对外暴露接口，例如 app 对象、CLI 命令组。
    exposed_interfaces: list[str] = Field(default_factory=list, description='暴露接口')

    # 入口依赖的外部服务线索，例如 Redis、PostgreSQL。
    service_dependencies: list[str] = Field(default_factory=list, description='外部服务依赖')

    # 当前入口所属的子项目根目录；单仓库时通常为 .
    subproject_root: str | None = Field(default=None, description='所属子项目根目录')

    # 子项目标签，例如 monorepo_workspace、service、package。
    subproject_markers: list[str] = Field(default_factory=list, description='子项目标记')

    # 从当前入口文件中抽取到的关键符号，尽量带上类型和行号。
    code_symbols: list[str] = Field(default_factory=list, description='关键符号')

    # 从当前入口文件中抽取到的模块依赖，尽量带上关系类型和行号。
    module_relations: list[str] = Field(default_factory=list, description='模块依赖关系')

    # 归纳后的入口摘要文本。
    summary: str = Field(..., description='入口摘要')

    # 支撑该摘要的证据列表。
    evidence: list[str] = Field(default_factory=list, description='入口证据')


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

    # 当前使用的检索后端，例如 chroma 或 local。
    backend: str = Field(default='local', description='检索后端')

    # 当前知识库中的仓库总数。
    repo_count: int = Field(default=0, description='仓库总数')

    # 当前知识库中的文档总数。
    document_count: int = Field(default=0, description='文档总数')


class IndexResult(BaseModel):
    """表示一次知识索引写入结果。"""

    # 本地 JSON 知识文档保存路径。
    local_path: str = Field(..., description='本地知识文档路径')

    # 是否成功写入向量索引。
    vector_indexed: bool = Field(default=False, description='是否完成向量索引')

    # 实际使用的向量后端名称，例如 chroma；未启用时为 none。
    vector_backend: str = Field(default='none', description='向量后端')

    # 附加状态说明，便于 CLI 提示用户。
    message: str | None = Field(default=None, description='状态说明')
