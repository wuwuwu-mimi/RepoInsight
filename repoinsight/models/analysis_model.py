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
