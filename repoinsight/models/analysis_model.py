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
