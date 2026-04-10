from typing import Annotated
from rich import print
from rich.panel import Panel
import typer

from repoinsight.ingest.repo_service import get_repo_info
from repoinsight.utils.check_util import check_url

app = typer.Typer()


@app.command(help="get version")
def version() -> None:
    print("RepoInsight -- Version 1.0")


@app.command(name="analyze", help="Pull repo information")
def analyze(
        url: Annotated[str, typer.Argument(help="GitHub repository URL")],
) -> None:
    if check_url(url):
        """get repo information"""
        repo_info = get_repo_info(url)
        repo_data = repo_info.repo_model
        readme_success = bool(repo_info.readme and repo_info.readme.strip())
        output = (
            f"[b cyan]仓库:             [/] {repo_data.full_name}\n"
            f"[b cyan]描述:             [/] {repo_data.description or '无'}\n"
            f"[b cyan]默认分支:         [/] {repo_data.default_branch}\n"
            f"[b cyan]主要语言:         [/] {repo_data.primary_language or '未指定'}\n"
            f"[b cyan]主题标签:         [/] {', '.join(repo_data.topics) if repo_data.topics else '无'}\n"
            f"[b cyan]Stars:            [/] {repo_data.stargazers_count}\n"
            f"[b cyan]README:           [/] {'已获取' if readme_success else '未获取'}\n"
            f"[b cyan]最后更新时间:     [/] {repo_data.updated_at}\n"
            f"[b cyan]仓库地址:         [/] {repo_data.html_url}\n"
            f"[b cyan]许可证:           [/] {repo_data.license_name or "无"}\n"
        )

        print(Panel(output, title="[bold green]GitHub 仓库信息", border_style="blue"))
    else:
        print(f"{url}[red] is not a github repository[/red]")


if __name__ == "__main__":
    app()
