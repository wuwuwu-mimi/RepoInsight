import os

from repoinsight.ingest.github_client import GithubClient
from repoinsight.models.repo_model import RepoInfo, RepoModel
from repoinsight.utils.repo_util import to_info


def get_repo_metadata(url: str) -> RepoModel:
    """只获取仓库元数据，不拉取 README。"""
    client = GithubClient(access_key=os.getenv("GITHUB_ACCESS_KEY", None))
    return client.get_repo_info(url=url)


def get_repo_readme(url: str) -> str | None:
    """只获取仓库 README 文本。"""
    client = GithubClient(access_key=os.getenv("GITHUB_ACCESS_KEY", None))
    return client.get_readme(url=url)


def get_repo_info(url: str) -> RepoInfo:
    """兼容旧接口，一次性获取仓库元数据与 README。"""
    repo_info = get_repo_metadata(url)
    readme = get_repo_readme(url)
    return to_info(repo_info, readme)
