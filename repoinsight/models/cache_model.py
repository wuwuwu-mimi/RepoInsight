from datetime import datetime

from pydantic import BaseModel, Field


class CachedRepoEntry(BaseModel):
    """表示本地 clone 缓存中的单个仓库。"""

    # 仓库标识，例如 owner/repo。
    repo_id: str = Field(..., description='仓库标识')

    # 仓库所有者，例如 octocat。
    owner: str = Field(..., description='仓库所有者')

    # 仓库名称，例如 Hello-World。
    repo: str = Field(..., description='仓库名称')

    # 本地仓库绝对路径。
    local_path: str = Field(..., description='本地仓库路径')

    # 是否检测到 .git 目录。
    is_git_repo: bool = Field(..., description='是否为有效 Git 仓库')

    # 本地目录最后修改时间。
    last_modified: datetime | None = Field(default=None, description='最后修改时间')

    # 本地目录大小，单位为字节。
    size_bytes: int | None = Field(default=None, description='目录大小（字节）')

    # 是否存在对应的 Markdown 分析报告。
    has_report: bool = Field(default=False, description='是否存在分析报告')

    # 对应的 Markdown 报告路径；不存在时为 None。
    report_path: str | None = Field(default=None, description='分析报告路径')


class CachedRepoListResult(BaseModel):
    """表示本地 clone 仓库列表结果。"""

    # clone 根目录绝对路径。
    clone_root: str = Field(..., description='clone 根目录')

    # 已缓存仓库列表。
    repos: list[CachedRepoEntry] = Field(default_factory=list, description='本地仓库列表')

    # 当前本地缓存的仓库总数。
    total_count: int = Field(default=0, description='仓库总数')
