import base64
from urllib.parse import urlparse

import httpx

from repoinsight.models.repo_model import RepoModel


class GithubClient:
    def __init__(self, access_key: str | None = None):
        # 访问公开仓库时 token 不是必需的，但带上后可以减少速率限制问题。
        self.token = access_key
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def get_repo_info(self, url: str) -> RepoModel:
        """获取仓库基础信息，并转换为统一的 RepoModel。"""
        api_url = self._to_api_url(url)
        with httpx.Client(trust_env=False, timeout=20.0) as client:
            # 先获取仓库基础元数据。
            repo_resp = client.get(api_url, headers=self.headers)
            repo_resp.raise_for_status()
            repo = repo_resp.json()

            # 再获取语言分布信息，后续做技术栈判断会用到。
            lang_resp = client.get(f"{api_url}/languages", headers=self.headers)
            lang_resp.raise_for_status()
            languages = lang_resp.json()

        return self._to_model(repo, languages)

    def get_readme(self, url: str) -> str | None:
        """获取并解码仓库 README 内容。"""
        api_url = self._to_api_url(url)
        try:
            resp = httpx.get(
                f"{api_url}/readme",
                headers=self.headers,
                trust_env=False,
                timeout=20.0,
            )
            if resp.status_code != 200:
                return None

            readme = resp.json()
            content_b64 = readme.get("content", "")
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return decoded.strip()
        except Exception:
            return None

    def _to_model(self, repo: dict, languages: dict[str, int]) -> RepoModel:
        """把 GitHub API 返回结果映射成内部统一模型。"""
        return RepoModel(
            owner=repo["owner"]["login"],
            name=repo["name"],
            full_name=repo["full_name"],
            html_url=repo["html_url"],
            description=repo.get("description"),
            default_branch=repo["default_branch"],
            private=repo["private"],
            is_fork=repo["fork"],
            archived=repo["archived"],
            homepage=repo.get("homepage"),
            license_name=repo["license"]["name"] if repo.get("license") else None,
            topics=repo.get("topics", []),
            primary_language=repo.get("language"),
            languages=languages,
            size_kb=repo.get("size"),
            stargazers_count=repo["stargazers_count"],
            forks_count=repo["forks_count"],
            watchers_count=repo["watchers_count"],
            open_issues_count=repo["open_issues_count"],
            created_at=repo.get("created_at"),
            updated_at=repo.get("updated_at"),
            pushed_at=repo.get("pushed_at"),
        )

    def _to_api_url(self, url: str) -> str:
        """把 GitHub 网页地址转换成对应的 REST API 地址。"""
        if url.startswith("https://api.github.com/repos/"):
            return url

        parsed = urlparse(url)
        if parsed.netloc != "github.com":
            return url

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(
                "GitHub 仓库地址格式应为 https://github.com/<owner>/<repo>"
            )

        owner, repo = parts[0], parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]

        return f"https://api.github.com/repos/{owner}/{repo}"
