from datetime import datetime

from pydantic import BaseModel, Field


class RepoModel(BaseModel):
    """分析流程使用的标准化仓库元数据模型。"""

    # 仓库所有者用户名，例如 "langchain-ai"
    owner: str = Field(..., description="仓库所有者用户名")

    # 仓库短名称，例如 "langgraph"
    name: str = Field(..., description="仓库名称")

    # 完整仓库标识，格式为 "所有者/仓库名"
    full_name: str = Field(..., description="仓库完整名称")

    # 仓库在 GitHub 上的标准页面地址
    html_url: str = Field(..., description="GitHub 仓库页面地址")

    # GitHub 仓库页面显示的简短描述
    description: str | None = Field(
        default=None,
        description="GitHub 仓库描述",
    )

    # 用于浏览和克隆的默认主分支
    default_branch: str = Field(..., description="仓库默认分支")

    # 仓库是否为 GitHub 私有仓库
    private: bool = Field(default=False, description="是否为私有仓库")

    # 仓库是否是其他仓库的复刻
    is_fork: bool = Field(default=False, description="是否为复刻仓库")

    # 仓库是否已归档，不再积极维护
    archived: bool = Field(default=False, description="是否已归档")

    # GitHub 上配置的项目主页或文档地址
    homepage: str | None = Field(default=None, description="仓库主页地址")

    # GitHub 检测到的许可证名称（类 SPDX 格式）
    license_name: str | None = Field(default=None, description="检测到的许可证名称")

    # 用户为仓库设置的 GitHub 主题标签
    topics: list[str] = Field(
        default_factory=list,
        description="GitHub 仓库主题标签",
    )

    # GitHub 返回的主要开发语言
    primary_language: str | None = Field(
        default=None,
        description="GitHub 报告的主要语言",
    )

    # 各语言代码字节数统计，键为语言名称
    languages: dict[str, int] = Field(
        default_factory=dict,
        description="各语言代码字节分布",
    )

    # GitHub 报告的仓库近似大小，单位：KB
    size_kb: int | None = Field(default=None, description="仓库大小（KB）")

    # 总星标数，可作为项目受欢迎程度的参考
    stargazers_count: int = Field(default=0, description="仓库星标数量")

    # 总复刻数，可作为项目使用和贡献情况的参考
    forks_count: int = Field(default=0, description="仓库复刻数量")

    # GitHub 报告的总关注者/订阅者数量
    watchers_count: int = Field(default=0, description="仓库关注者数量")

    # GitHub 上当前开启的问题总数
    open_issues_count: int = Field(default=0, description="开启的问题数量")

    # 仓库在 GitHub 上的创建时间
    created_at: datetime | None = Field(
        default=None,
        description="仓库创建时间",
    )

    # 仓库元数据最后一次更新时间
    updated_at: datetime | None = Field(
        default=None,
        description="仓库更新时间",
    )

    # 最后一次推送代码的时间，通常比 updated_at 更能体现维护活跃度
    pushed_at: datetime | None = Field(
        default=None,
        description="仓库最后推送时间",
    )


class RepoInfo(BaseModel):
    repo_model: RepoModel
    readme: str | None = Field(default=None, description="README.md")
