from datetime import datetime
from typing import Annotated

import typer
from rich import print
from rich.panel import Panel
from rich.table import Table

from repoinsight.analyze.pipeline import run_analysis
from repoinsight.ingest.repo_cache import list_cloned_repos, remove_cloned_repo
from repoinsight.llm.context_builder import remove_llm_context_text, save_llm_context_text
from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.report.json_report import remove_json_report, save_json_report
from repoinsight.report.markdown_report import remove_markdown_report, save_markdown_report
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.index_service import index_analysis_result, remove_indexed_repo

app = typer.Typer()


@app.command(help='get version')
def version() -> None:
    print('RepoInsight -- Version 1.0')


@app.command(name='analyze', help='Pull repo information and scan repository')
def analyze(
    url: Annotated[str, typer.Argument(help='GitHub repository URL')],
    save_report: Annotated[
        bool,
        typer.Option('--save-report/--no-save-report', help='是否保存 Markdown 报告'),
    ] = True,
    output_dir: Annotated[
        str,
        typer.Option('--output-dir', help='报告输出目录'),
    ] = 'reports',
) -> None:
    """执行 analyze 主流程，并把关键结果输出到终端。"""
    try:
        result = run_analysis(url)
    except Exception as exc:
        print(f'[red]分析失败：{exc}[/red]')
        return

    _render_repo_summary(result)
    _render_scan_summary(result)
    _render_insight_summary(result)
    _render_project_profile_summary(result)
    _render_tech_stack_summary(result)
    _render_key_file_summary(result)

    if save_report:
        try:
            markdown_path = save_markdown_report(result, output_dir=output_dir)
            json_path = save_json_report(result, output_dir=output_dir)
            llm_context_path = save_llm_context_text(result, output_dir=output_dir)
            knowledge_path = index_analysis_result(result)
            print(f'[green]Markdown 报告已保存[/green]：{markdown_path}')
            print(f'[green]JSON 报告已保存[/green]：{json_path}')
            print(f'[green]LLM 上下文已保存[/green]：{llm_context_path}')
            print(f'[green]知识索引已保存[/green]：{knowledge_path}')
        except Exception as exc:
            print(f'[red]报告保存失败：{exc}[/red]')


@app.command(help='remove a repo')
def remove(
    repo_name: str,
    remove_report: Annotated[
        bool,
        typer.Option('--remove-report/--keep-report', help='是否同时删除本地分析报告'),
    ] = False,
    output_dir: Annotated[
        str,
        typer.Option('--output-dir', help='报告输出目录'),
    ] = 'reports',
) -> None:
    try:
        removed = remove_cloned_repo(repo_name)
    except Exception as exc:
        print(f'[red]删除失败：{exc}[/red]')
        return

    if removed:
        print(f'[green]{repo_name}[/green] 已删除')
    else:
        print(f'[yellow]{repo_name}[/yellow] 本地不存在')

    if remove_report:
        try:
            markdown_removed = remove_markdown_report(repo_name, output_dir=output_dir)
            json_removed = remove_json_report(repo_name, output_dir=output_dir)
            llm_context_removed = remove_llm_context_text(repo_name, output_dir=output_dir)
            knowledge_removed = remove_indexed_repo(repo_name)
        except Exception as exc:
            print(f'[red]报告删除失败：{exc}[/red]')
            return

        if markdown_removed or json_removed or llm_context_removed or knowledge_removed:
            print(f'[green]{repo_name}[/green] 的分析产物已删除')
        else:
            print(f'[yellow]{repo_name}[/yellow] 没有可删除的分析产物')


@app.command(name='list', help='list all cloned repos')
def list_repos() -> None:
    """列出本地 clone 缓存中的所有仓库。"""
    result = list_cloned_repos()
    if result.total_count == 0:
        print('[yellow]当前没有已缓存的仓库[/yellow]')
        return

    table = Table(title='本地已缓存仓库', show_lines=False)
    table.add_column('仓库', style='cyan', no_wrap=True)
    table.add_column('报告', style='yellow', no_wrap=True)
    table.add_column('最后修改时间', style='green')
    table.add_column('大小', justify='right', style='magenta')
    table.add_column('本地路径', style='white')

    for repo in result.repos:
        table.add_row(
            repo.repo_id,
            '有' if repo.has_report else '无',
            _format_datetime(repo.last_modified),
            _format_size(repo.size_bytes),
            repo.local_path,
        )

    print(table)
    print(f'[bold]总数[/bold]：{result.total_count}')
    print(f'[bold]clone 根目录[/bold]：{result.clone_root}')


@app.command(name='search', help='search analyzed repositories')
def search(
    query: Annotated[str, typer.Argument(help='检索问题或关键词')],
    top_k: Annotated[int, typer.Option('--top-k', min=1, help='返回结果数量')] = 5,
) -> None:
    """在本地知识库中检索已分析仓库。"""
    result = search_knowledge_base(query=query, top_k=top_k)
    if not result.hits:
        print('[yellow]当前没有命中结果，请先执行 analyze 建立知识索引[/yellow]')
        print(f'[bold]知识库仓库数[/bold]：{result.repo_count}')
        print(f'[bold]知识库文档数[/bold]：{result.document_count}')
        return

    table = Table(title=f'检索结果：{query}', show_lines=False)
    table.add_column('仓库', style='cyan', no_wrap=True)
    table.add_column('类型', style='green', no_wrap=True)
    table.add_column('来源', style='yellow')
    table.add_column('得分', justify='right', style='magenta')
    table.add_column('摘要片段', style='white')

    for hit in result.hits:
        source = hit.document.source_path or '仓库级文档'
        table.add_row(
            hit.document.repo_id,
            hit.document.doc_type,
            source,
            f'{hit.score:.2f}',
            hit.snippet,
        )

    print(table)
    print(f'[bold]知识库仓库数[/bold]：{result.repo_count}')
    print(f'[bold]知识库文档数[/bold]：{result.document_count}')



def _render_repo_summary(result: AnalysisRunResult) -> None:
    """输出仓库基础信息面板。"""
    repo = result.repo_info.repo_model
    readme_success = bool(result.repo_info.readme and result.repo_info.readme.strip())
    output = (
        f"[b cyan]仓库:[/] {repo.full_name}\n"
        f"[b cyan]描述:[/] {repo.description or '无'}\n"
        f"[b cyan]默认分支:[/] {repo.default_branch}\n"
        f"[b cyan]主要语言:[/] {repo.primary_language or '未指定'}\n"
        f"[b cyan]主题标签:[/] {', '.join(repo.topics) if repo.topics else '无'}\n"
        f"[b cyan]Stars:[/] {repo.stargazers_count}\n"
        f"[b cyan]README:[/] {'已获取' if readme_success else '未获取'}\n"
        f"[b cyan]最后更新时间:[/] {repo.updated_at}\n"
        f"[b cyan]仓库地址:[/] {repo.html_url}\n"
        f"[b cyan]许可证:[/] {repo.license_name or '无'}\n"
        f"[b cyan]本地路径:[/] {result.clone_path}"
    )
    print(Panel(output, title='[bold green]GitHub 仓库信息', border_style='blue'))



def _render_scan_summary(result: AnalysisRunResult) -> None:
    """输出扫描统计信息。"""
    stats = result.scan_result.stats
    output = (
        f"[b cyan]扫描文件总数:[/] {stats.total_seen}\n"
        f"[b cyan]候选文件数:[/] {stats.kept_count}\n"
        f"[b cyan]忽略路径数:[/] {stats.ignored_count}\n"
        f"[b cyan]关键文件数:[/] {stats.key_file_count}"
    )
    print(Panel(output, title='[bold green]扫描统计', border_style='green'))



def _render_insight_summary(result: AnalysisRunResult) -> None:
    """输出项目类型、优势、风险和初步观察。"""
    lines: list[str] = []

    if result.project_type:
        lines.append(f"[b cyan]项目类型:[/] {result.project_type}")
        lines.append(f"[b cyan]判断依据:[/] {result.project_type_evidence or '无'}")
    else:
        lines.append('[b cyan]项目类型:[/] 暂未明确识别')

    if result.observations:
        lines.append('')
        lines.append('[b cyan]初步观察:[/]')
        for item in result.observations:
            lines.append(f'- {item}')
    else:
        lines.append('')
        lines.append('[b cyan]初步观察:[/]')
        lines.append('- 暂无')

    if result.strengths:
        lines.append('')
        lines.append('[b cyan]优势:[/]')
        for item in result.strengths:
            lines.append(f'- {item}')
    else:
        lines.append('')
        lines.append('[b cyan]优势:[/]')
        lines.append('- 暂无')

    if result.risks:
        lines.append('')
        lines.append('[b cyan]风险:[/]')
        for item in result.risks:
            lines.append(f'- {item}')
    else:
        lines.append('')
        lines.append('[b cyan]风险:[/]')
        lines.append('- 暂无')

    print(Panel('\n'.join(lines), title='[bold green]项目分析结论', border_style='magenta'))



def _render_project_profile_summary(result: AnalysisRunResult) -> None:
    """输出结构化项目画像，便于后续扩展到多语言仓库。"""
    profile = result.project_profile
    output = (
        f"[b cyan]主语言:[/] {profile.primary_language or '未识别'}\n"
        f"[b cyan]语言:[/] {_join_or_none(profile.languages)}\n"
        f"[b cyan]运行时:[/] {_join_or_none(profile.runtimes)}\n"
        f"[b cyan]框架:[/] {_join_or_none(profile.frameworks)}\n"
        f"[b cyan]构建工具:[/] {_join_or_none(profile.build_tools)}\n"
        f"[b cyan]包管理器:[/] {_join_or_none(profile.package_managers)}\n"
        f"[b cyan]测试工具:[/] {_join_or_none(profile.test_tools)}\n"
        f"[b cyan]CI/CD:[/] {_join_or_none(profile.ci_cd_tools)}\n"
        f"[b cyan]部署工具:[/] {_join_or_none(profile.deploy_tools)}\n"
        f"[b cyan]入口文件:[/] {_join_or_none(profile.entrypoints)}\n"
        f"[b cyan]项目标记:[/] {_join_or_none(profile.project_markers)}"
    )
    print(Panel(output, title='[bold green]项目画像', border_style='cyan'))


def _render_tech_stack_summary(result: AnalysisRunResult) -> None:
    """输出技术栈推断结果。"""
    if not result.tech_stack:
        print('[yellow]未推断出明确的技术栈[/yellow]')
        return

    table = Table(title='技术栈推断', show_lines=False)
    table.add_column('技术', style='cyan', no_wrap=True)
    table.add_column('分类', style='green', no_wrap=True)
    table.add_column('证据', style='white')

    for item in result.tech_stack:
        table.add_row(item.name, item.category, item.evidence)

    print(table)



def _render_key_file_summary(result: AnalysisRunResult) -> None:
    """输出关键文件读取结果。"""
    if not result.key_file_contents:
        print('[yellow]未读取到关键文件内容[/yellow]')
        return

    table = Table(title='关键文件内容预览', show_lines=False)
    table.add_column('路径', style='cyan')
    table.add_column('大小', justify='right', style='magenta')
    table.add_column('是否截断', justify='center', style='yellow')
    table.add_column('内容预览', style='white')

    for item in result.key_file_contents:
        preview = item.content[:120].replace('\n', ' ')
        if len(item.content) > 120:
            preview += '...'
        table.add_row(
            item.path,
            _format_size(item.size_bytes),
            '是' if item.truncated else '否',
            preview or '空内容',
        )

    print(table)



def _format_datetime(value: datetime | None) -> str:
    """把时间格式化为更适合终端展示的字符串。"""
    if value is None:
        return '未知'
    return value.strftime('%Y-%m-%d %H:%M:%S')



def _format_size(size_bytes: int | None) -> str:
    """把字节数转换为便于阅读的大小文本。"""
    if size_bytes is None:
        return '未知'

    value = float(size_bytes)
    units = ['B', 'KB', 'MB', 'GB']
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == 'B':
                return f'{int(value)} {unit}'
            return f'{value:.1f} {unit}'
        value /= 1024

    return f'{size_bytes} B'


def _join_or_none(items: list[str]) -> str:
    """把列表拼接成更适合终端展示的文本。"""
    return ', '.join(items) if items else '无'


if __name__ == '__main__':
    app()
