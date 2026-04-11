from pydantic import BaseModel, Field


class FileEntry(BaseModel):
    """表示扫描后保留的单个文件。"""

    # 仓库内的相对路径，例如 src/main.py。
    path: str = Field(..., description="文件相对路径")

    # 文件名，例如 main.py。
    name: str = Field(..., description="文件名")

    # 文件扩展名，例如 .py；无扩展名时为 None。
    extension: str | None = Field(default=None, description="文件扩展名")

    # 文件大小，单位为字节。
    size_bytes: int = Field(..., description="文件大小（字节）")

    # 父目录相对路径；根目录下文件则为空字符串。
    parent_dir: str = Field(..., description="父目录相对路径")

    # 是否属于关键文件，例如 README.md、pyproject.toml。
    is_key_file: bool = Field(default=False, description="是否为关键文件")


class IgnoredEntry(BaseModel):
    """表示被扫描器忽略的文件或目录。"""

    # 被忽略路径的相对路径。
    path: str = Field(..., description="被忽略的路径")

    # 路径类型，例如 file 或 directory。
    entry_type: str = Field(..., description="路径类型")

    # 被忽略原因，例如 ignored_dir、file_too_large。
    reason: str = Field(..., description="被忽略原因")


class ScanStats(BaseModel):
    """记录本次扫描的统计信息。"""

    # 扫描过程中看到的总文件数。
    total_seen: int = Field(default=0, description="扫描到的总文件数")

    # 保留下来的候选文件数。
    kept_count: int = Field(default=0, description="保留的候选文件数")

    # 被忽略的路径数量，包含文件和目录。
    ignored_count: int = Field(default=0, description="被忽略的路径数量")

    # 被识别为关键文件的数量。
    key_file_count: int = Field(default=0, description="关键文件数量")


class ScanResult(BaseModel):
    """表示仓库文件扫描结果。"""

    # 仓库根目录绝对路径。
    root_path: str = Field(..., description="仓库根目录绝对路径")

    # 保留下来的候选文件列表。
    all_files: list[FileEntry] = Field(default_factory=list, description="候选文件列表")

    # 从候选文件中识别出的关键文件列表。
    key_files: list[FileEntry] = Field(default_factory=list, description="关键文件列表")

    # 用于 CLI 或报告展示的目录树预览。
    tree_preview: list[str] = Field(default_factory=list, description="目录树预览")

    # 被忽略的文件或目录列表。
    ignored_entries: list[IgnoredEntry] = Field(default_factory=list, description="被忽略路径列表")

    # 扫描统计信息。
    stats: ScanStats = Field(default_factory=ScanStats, description="扫描统计信息")
