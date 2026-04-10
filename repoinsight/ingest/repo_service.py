import os

from repoinsight.ingest.github_client import GithubClient
from repoinsight.models.repo_model import RepoInfo
from repoinsight.utils.repo_util import to_info


def get_repo_info(url: str) -> RepoInfo:
    client = GithubClient(access_key=os.getenv("GITHUB_ACCESS_KEY", None))
    repo_info = client.get_repo_info(url=url)
    readme = client.get_readme(url=url)
    return to_info(repo_info, readme)
