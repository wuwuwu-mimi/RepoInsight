from repoinsight.models.repo_model import RepoModel, RepoInfo


def to_info(repo: RepoModel, readme: str | None) -> RepoInfo:
    return RepoInfo(repo_model=repo, readme=readme)
