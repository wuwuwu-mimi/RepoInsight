from pydantic import BaseModel, Field

from repoinsight.models.file_model import ScanResult
from repoinsight.models.repo_model import RepoInfo


class KeyFileContent(BaseModel):
    """表示一个已读取的关键文件内容。"""

    # 关键文件相对路径。
    path: str = Field(..., description='关键文件相对路径')

    # 文件原始大小，单位为字节。
    size_bytes: int = Field(..., description='文件大小（字节）')

    # 实际读取出来的文本内容。
    content: str = Field(..., description='文件文本内容')

    # 是否因为过大而被截断。
    truncated: bool = Field(default=False, description='是否被截断')


class TechStackItem(BaseModel):
    """表示一条推断出的技术栈信息。"""

    # 技术名称，例如 Python、FastAPI、LangGraph。
    name: str = Field(..., description='技术名称')

    # 技术分类，例如 language、framework、tool。
    category: str = Field(..., description='技术分类')

    # 用于说明推断来源的证据文本。
    evidence: str = Field(..., description='推断证据')

    # 证据强度，例如 strong、medium、weak。
    evidence_level: str = Field(default='medium', description='证据强度')

    # 证据来源类型，例如 dependency、config、import、runtime_call、metadata。
    evidence_source: str = Field(default='unknown', description='证据来源类型')

    # 证据对应的来源文件；若来自 GitHub 元数据则可为空。
    source_path: str | None = Field(default=None, description='证据来源文件')


class CodeSymbol(BaseModel):
    """表示从关键文件中抽取出的符号定义。"""

    # 符号名称，例如 app、main、build_cli。
    name: str = Field(..., description='符号名称')

    # 符号类型，例如 function、class、variable、route。
    symbol_type: str = Field(..., description='符号类型')

    # 符号所在文件路径。
    source_path: str = Field(..., description='来源文件路径')

    # 符号所在的大致行号。
    line_number: int | None = Field(default=None, description='大致行号')


class ModuleRelation(BaseModel):
    """表示关键文件之间或对外模块的引用关系。"""

    # 当前关系的源文件。
    source_path: str = Field(..., description='源文件路径')

    # 关系目标，例如导入的模块或包名。
    target: str = Field(..., description='关系目标')

    # 关系类型，例如 import、require、use、route。
    relation_type: str = Field(..., description='关系类型')

    # 关系出现的大致行号。
    line_number: int | None = Field(default=None, description='大致行号')


class CodeEntity(BaseModel):
    """统一描述代码分析阶段抽取出的实体。"""

    entity_kind: str = Field(..., description='实体类型')
    name: str = Field(..., description='实体名称')
    qualified_name: str | None = Field(default=None, description='实体限定名')
    source_path: str | None = Field(default=None, description='来源文件路径')
    language_scope: str | None = Field(default=None, description='语言范围')
    location: str | None = Field(default=None, description='源码位置')
    tags: list[str] = Field(default_factory=list, description='实体标签')


class CodeRelationEdge(BaseModel):
    """统一描述代码实体之间的关系边。"""

    source_ref: str = Field(..., description='关系源')
    target_ref: str = Field(..., description='关系目标')
    relation_type: str = Field(..., description='关系类型')
    source_path: str | None = Field(default=None, description='来源文件路径')
    line_number: int | None = Field(default=None, description='大致行号')


class FunctionSummary(BaseModel):
    """表示仓库中一个可检索的函数或方法摘要。"""

    # 函数或方法名，例如 create_app、bootstrap、handle_login。
    name: str = Field(..., description='函数或方法名')

    # 带上下文的限定名，例如 create_app、App.bootstrap。
    qualified_name: str = Field(..., description='函数限定名')

    # 来源文件的相对路径。
    source_path: str = Field(..., description='来源文件路径')

    # 语言范围，例如 python、nodejs。
    language_scope: str = Field(..., description='语言范围')

    # 起始行号。
    line_start: int | None = Field(default=None, description='起始行号')

    # 结束行号。
    line_end: int | None = Field(default=None, description='结束行号')

    # 函数签名，尽量保留原始声明形式。
    signature: str = Field(..., description='函数签名')

    # 若为类方法，则记录所属类名；普通函数为 None。
    owner_class: str | None = Field(default=None, description='所属类名')

    # 是否为异步函数。
    is_async: bool = Field(default=False, description='是否为异步函数')

    # 装饰器列表，例如 app.get、classmethod。
    decorators: list[str] = Field(default_factory=list, description='装饰器列表')

    # 形参名列表。
    parameters: list[str] = Field(default_factory=list, description='参数列表')

    # 函数体内识别到的调用目标，例如 FastAPI、include_router、redis.get。
    called_symbols: list[str] = Field(default_factory=list, description='调用目标列表')

    # 返回值线索，例如 app、settings、dict。
    return_signals: list[str] = Field(default_factory=list, description='返回值线索')

    # 供报告和 RAG 直接复用的函数摘要。
    summary: str = Field(..., description='函数摘要')


class ClassSummary(BaseModel):
    """表示仓库中一个可检索的类摘要。"""

    # 类名，例如 Settings、RepoAnalyzer。
    name: str = Field(..., description='类名')

    # 带上下文的限定名；当前 MVP 与 name 一致，预留给后续嵌套类场景。
    qualified_name: str = Field(..., description='类限定名')

    # 来源文件的相对路径。
    source_path: str = Field(..., description='来源文件路径')

    # 语言范围，例如 python、nodejs。
    language_scope: str = Field(..., description='语言范围')

    # 起始行号。
    line_start: int | None = Field(default=None, description='起始行号')

    # 结束行号。
    line_end: int | None = Field(default=None, description='结束行号')

    # 继承或扩展的基类列表。
    bases: list[str] = Field(default_factory=list, description='基类列表')

    # 装饰器列表。
    decorators: list[str] = Field(default_factory=list, description='装饰器列表')

    # 类中识别到的方法名列表。
    methods: list[str] = Field(default_factory=list, description='方法列表')

    # 供报告和 RAG 直接复用的类摘要。
    summary: str = Field(..., description='类摘要')


class ApiRouteSummary(BaseModel):
    """表示仓库中一个可检索的接口或路由摘要。"""

    # 路由路径，例如 /login、/api/users/{id}。
    route_path: str = Field(..., description='路由路径')

    # HTTP 方法列表，例如 GET、POST；WebSocket 可记录为 WEBSOCKET。
    http_methods: list[str] = Field(default_factory=list, description='HTTP 方法列表')

    # 来源文件的相对路径。
    source_path: str = Field(..., description='来源文件路径')

    # 语言范围，例如 python、nodejs。
    language_scope: str = Field(..., description='语言范围')

    # 识别出的框架线索，例如 fastapi、flask、express。
    framework: str | None = Field(default=None, description='框架线索')

    # 处理该路由的函数或方法名。
    handler_name: str = Field(..., description='处理函数名')

    # 带上下文的处理函数限定名，例如 AuthView.login。
    handler_qualified_name: str = Field(..., description='处理函数限定名')

    # 若为类方法，则记录所属类名；普通函数为 None。
    owner_class: str | None = Field(default=None, description='所属类名')

    # 路由定义所在的大致行号。
    line_number: int | None = Field(default=None, description='定义行号')

    # 与该路由直接相关的装饰器或注册调用。
    decorators: list[str] = Field(default_factory=list, description='路由装饰器列表')

    # 路由处理逻辑中识别到的调用目标。
    called_symbols: list[str] = Field(default_factory=list, description='调用目标列表')

    # 供报告和 RAG 直接复用的接口摘要。
    summary: str = Field(..., description='接口摘要')


class SubprojectSummary(BaseModel):
    """表示复杂仓库中的一个子项目或工作区。"""

    # 子项目根目录，相对于仓库根目录。
    root_path: str = Field(..., description='子项目根目录')

    # 子项目语言范围，例如 python、nodejs、go。
    language_scope: str = Field(..., description='语言范围')

    # 子项目识别类型，例如 package、service、workspace。
    project_kind: str = Field(..., description='子项目类型')

    # 与子项目相关的配置文件。
    config_paths: list[str] = Field(default_factory=list, description='配置文件列表')

    # 与子项目相关的入口文件。
    entrypoint_paths: list[str] = Field(default_factory=list, description='入口文件列表')

    # 子项目标签，例如 monorepo_workspace、service。
    markers: list[str] = Field(default_factory=list, description='子项目标签')


class ProjectProfile(BaseModel):
    """表示面向多语言仓库的结构化项目画像。"""

    # 当前识别出的主语言，会优先结合 GitHub 元数据与关键文件推断。
    primary_language: str | None = Field(default=None, description='主语言')

    # 仓库中识别出的语言列表，例如 Python、TypeScript、Go。
    languages: list[str] = Field(default_factory=list, description='语言列表')

    # 仓库中识别出的运行时，例如 Node.js。
    runtimes: list[str] = Field(default_factory=list, description='运行时列表')

    # 仓库中识别出的框架，例如 FastAPI、React、Next.js。
    frameworks: list[str] = Field(default_factory=list, description='框架列表')

    # 仓库中识别出的构建工具，例如 Vite、Cargo、Gradle。
    build_tools: list[str] = Field(default_factory=list, description='构建工具列表')

    # 仓库中识别出的包管理器，例如 npm、pnpm、pip、Poetry。
    package_managers: list[str] = Field(default_factory=list, description='包管理器列表')

    # 仓库中识别出的测试工具，例如 Pytest、Jest、Vitest。
    test_tools: list[str] = Field(default_factory=list, description='测试工具列表')

    # 仓库中识别出的 CI/CD 工具，例如 GitHub Actions。
    ci_cd_tools: list[str] = Field(default_factory=list, description='CI/CD 工具列表')

    # 仓库中识别出的部署或基础设施工具，例如 Docker、Docker Compose。
    deploy_tools: list[str] = Field(default_factory=list, description='部署工具列表')

    # 识别到的入口文件或关键入口位置。
    entrypoints: list[str] = Field(default_factory=list, description='入口文件列表')

    # 用于标记项目特征的额外标签，例如 monorepo、library、cli。
    project_markers: list[str] = Field(default_factory=list, description='项目特征标签')

    # 从关键文件中抽取出的子项目列表，便于理解 monorepo 或多服务结构。
    subprojects: list[SubprojectSummary] = Field(default_factory=list, description='子项目列表')

    # 从关键文件中抽取出的符号定义，便于做更细粒度证据展示。
    code_symbols: list[CodeSymbol] = Field(default_factory=list, description='代码符号列表')

    # 从关键文件中抽取出的模块关系，便于理解入口和依赖链。
    module_relations: list[ModuleRelation] = Field(default_factory=list, description='模块关系列表')

    # 统一抽象后的代码实体，便于后续多语言检索与图关系建模。
    code_entities: list[CodeEntity] = Field(default_factory=list, description='统一代码实体列表')

    # 统一抽象后的代码关系边，便于后续关系分析与 GraphRAG 扩展。
    code_relation_edges: list[CodeRelationEdge] = Field(default_factory=list, description='统一代码关系边列表')

    # 函数级实现摘要，适合回答“某个功能怎么实现”的问题。
    function_summaries: list[FunctionSummary] = Field(default_factory=list, description='函数级摘要列表')

    # 类级实现摘要，适合回答模块职责和对象协作类问题。
    class_summaries: list[ClassSummary] = Field(default_factory=list, description='类级摘要列表')

    # 接口级摘要，适合回答某个 API 或路由是如何实现的。
    api_route_summaries: list[ApiRouteSummary] = Field(default_factory=list, description='接口级摘要列表')

    # 仅包含强/中证据的确认技术栈，适合作为默认展示与问答依据。
    confirmed_signals: list[TechStackItem] = Field(default_factory=list, description='确认技术栈信号')

    # 仅包含弱证据的候选技术栈，便于后续人工复核。
    weak_signals: list[TechStackItem] = Field(default_factory=list, description='弱证据候选信号')

    # 所有结构化识别结果的证据集合，可继续复用于报告和规则判断。
    signals: list[TechStackItem] = Field(default_factory=list, description='项目画像识别信号')


class AnalysisRunResult(BaseModel):
    """表示一次 analyze 主流程的聚合结果。"""

    # GitHub 元数据与 README 结果。
    repo_info: RepoInfo = Field(..., description='仓库元数据与 README 信息')

    # 本地克隆路径。
    clone_path: str = Field(..., description='本地克隆路径')

    # 仓库扫描结果。
    scan_result: ScanResult = Field(..., description='仓库扫描结果')

    # 已读取的关键文件内容。
    key_file_contents: list[KeyFileContent] = Field(default_factory=list, description='关键文件内容列表')

    # 规则推断出的技术栈结果。
    tech_stack: list[TechStackItem] = Field(default_factory=list, description='技术栈推断结果')

    # 面向多语言仓库的结构化项目画像。
    project_profile: ProjectProfile = Field(default_factory=ProjectProfile, description='项目画像')

    # 推断出的项目类型，例如 CLI 工具、Web 服务或 AI / RAG 项目。
    project_type: str | None = Field(default=None, description='项目类型')

    # 用于说明项目类型判断依据的文本。
    project_type_evidence: str | None = Field(default=None, description='项目类型推断依据')

    # 规则推断出的项目优势。
    strengths: list[str] = Field(default_factory=list, description='项目优势')

    # 规则推断出的潜在风险。
    risks: list[str] = Field(default_factory=list, description='潜在风险')

    # 规则推断出的初步观察结论。
    observations: list[str] = Field(default_factory=list, description='初步观察')

    # 各 stage 的执行记录，便于调试和后续编排接入。
    stage_trace: list['StageTraceEntry'] = Field(default_factory=list, description='阶段执行记录')


class StageTraceEntry(BaseModel):
    """表示单个分析 stage 的执行记录。"""

    # 当前阶段名称。
    stage_name: str = Field(..., description='阶段名称')

    # 当前阶段状态，例如 success 或 failed。
    status: str = Field(..., description='阶段状态')

    # 阶段完成后记录的简短说明。
    detail: str | None = Field(default=None, description='阶段说明')

    # 阶段失败时的错误信息。
    error_message: str | None = Field(default=None, description='失败错误信息')


class AnalysisState(BaseModel):
    """表示分析流程在各个 stage 之间传递的状态。"""

    # 当前要分析的 GitHub 仓库 URL。
    url: str = Field(..., description='待分析的仓库 URL')

    # GitHub 元数据与 README 结果。
    repo_info: RepoInfo | None = Field(default=None, description='仓库元数据与 README 信息')

    # 本地克隆路径。
    clone_path: str | None = Field(default=None, description='本地克隆路径')

    # 仓库扫描结果。
    scan_result: ScanResult | None = Field(default=None, description='仓库扫描结果')

    # 已读取的关键文件内容。
    key_file_contents: list[KeyFileContent] = Field(default_factory=list, description='关键文件内容列表')

    # 面向多语言仓库的结构化项目画像。
    project_profile: ProjectProfile = Field(default_factory=ProjectProfile, description='项目画像')

    # 规则推断出的技术栈结果。
    tech_stack: list[TechStackItem] = Field(default_factory=list, description='技术栈推断结果')

    # 推断出的项目类型，例如 CLI 工具、Web 服务、AI / RAG 项目。
    project_type: str | None = Field(default=None, description='项目类型')

    # 用于说明项目类型判断依据的文本。
    project_type_evidence: str | None = Field(default=None, description='项目类型推断依据')

    # 规则推断出的项目优势。
    strengths: list[str] = Field(default_factory=list, description='项目优势')

    # 规则推断出的潜在风险。
    risks: list[str] = Field(default_factory=list, description='潜在风险')

    # 规则推断出的初步观察结论。
    observations: list[str] = Field(default_factory=list, description='初步观察')

    # 当前正在执行的 stage 名称。
    current_stage: str | None = Field(default=None, description='当前阶段名称')

    # 已成功完成的 stage 名称列表。
    completed_stages: list[str] = Field(default_factory=list, description='已完成阶段列表')

    # 各 stage 的执行记录。
    stage_trace: list[StageTraceEntry] = Field(default_factory=list, description='阶段执行记录')


AnalysisRunResult.model_rebuild()
AnalysisState.model_rebuild()
