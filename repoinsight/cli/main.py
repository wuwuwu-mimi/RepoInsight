import json
from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich import print
from rich.panel import Panel
from rich.table import Table

from repoinsight.agents import (
    LangGraphUnavailableError,
    run_langgraph_analysis,
    run_langgraph_answer,
    run_multi_agent_analysis,
    run_multi_agent_answer,
)
from repoinsight.ingest.repo_cache import list_cloned_repos, remove_cloned_repo
from repoinsight.llm.config import get_llm_config_help_text, get_llm_settings
from repoinsight.llm.context_builder import remove_llm_context_text, save_llm_context_text
from repoinsight.agents.models import (
    AnswerVerificationResult,
    CodeInvestigationResult,
    CoordinatedAnalysisResult,
    CoordinatedAnswerResult,
)
from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.report.json_report import remove_json_report, save_json_report
from repoinsight.report.markdown_report import remove_markdown_report, save_markdown_report
from repoinsight.report.pdf_report import export_repo_report_to_pdf, remove_pdf_report
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.embedding_service import (
    check_embedding_health,
    EmbeddingProviderSwitchError,
    get_embedding_settings,
    set_embedding_provider_override,
)
from repoinsight.storage.index_service import (
    check_vector_store_health,
    get_vector_store_rebuild_overview,
    index_analysis_result,
    rebuild_vector_store,
    remove_indexed_repo,
    remove_vector_indexed_repo,
)

app = typer.Typer()
console = Console()


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
    embedding_mode: Annotated[
        str,
        typer.Option('--embedding-mode', help='embedding 模式，可选 service、ollama 或 sentence-transformers'),
    ] = 'service',
    output_dir: Annotated[
        str,
        typer.Option('--output-dir', help='报告输出目录'),
    ] = 'reports',
    orchestrator: Annotated[
        str,
        typer.Option('--orchestrator', help='分析编排器，可选 local 或 langgraph'),
    ] = 'local',
) -> None:
    """执行 analyze 主流程，并把关键结果输出到终端。"""
    if not _apply_embedding_mode(embedding_mode):
        return
    normalized_orchestrator = orchestrator.strip().lower()
    if normalized_orchestrator not in {'local', 'langgraph'}:
        print(f'[red]不支持的编排器：{orchestrator}[/red]')
        print('[yellow]当前可选：local, langgraph[/yellow]')
        return
    try:
        if normalized_orchestrator == 'langgraph':
            coordinated = run_langgraph_analysis(url, persist_knowledge=False)
        else:
            coordinated = run_multi_agent_analysis(url, persist_knowledge=False)
    except LangGraphUnavailableError as exc:
        print(f'[red]LangGraph 编排不可用：{exc}[/red]')
        return
    except Exception as exc:
        print(f'[red]分析失败：{exc}[/red]')
        return

    result = coordinated.analysis_result
    _render_repo_summary(result)
    _render_scan_summary(result)
    _render_insight_summary(result)
    _render_project_profile_summary(result)
    _render_tech_stack_summary(result)
    _render_key_file_summary(result)
    print(f'[bold]编排器[/bold]：{normalized_orchestrator}')
    _render_analysis_agent_trace(coordinated)

    if save_report:
        try:
            markdown_path = save_markdown_report(result, output_dir=output_dir)
            json_path = save_json_report(result, output_dir=output_dir)
            llm_context_path = save_llm_context_text(result, output_dir=output_dir)
            index_result = index_analysis_result(result)
            print(f'[green]Markdown 报告已保存[/green]：{markdown_path}')
            print(f'[green]JSON 报告已保存[/green]：{json_path}')
            print(f'[green]LLM 上下文已保存[/green]：{llm_context_path}')
            print(f'[green]知识文档已保存[/green]：{index_result.local_path}')
            if index_result.vector_indexed:
                print(f'[green]向量索引已写入[/green]：{index_result.vector_backend}')
            elif index_result.message:
                print(f'[yellow]{index_result.message}[/yellow]')
        except Exception as exc:
            print(f'[red]报告保存失败：{exc}[/red]')


@app.command(help='remove a repo')
def remove(
    repo_name: Annotated[str, typer.Argument(help='目标仓库标识，格式为 owner/repo')],
    remove_report: Annotated[
        bool,
        typer.Option('--remove-report/--keep-report', help='是否同时删除分析产物与知识索引，默认删除'),
    ] = True,
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
            pdf_removed = remove_pdf_report(repo_name, output_dir=output_dir)
            llm_context_removed = remove_llm_context_text(repo_name, output_dir=output_dir)
            knowledge_removed = remove_indexed_repo(repo_name)
        except Exception as exc:
            print(f'[red]报告删除失败：{exc}[/red]')
            return

        if markdown_removed or json_removed or pdf_removed or llm_context_removed or knowledge_removed:
            print(f'[green]{repo_name}[/green] 的分析产物已删除')
        else:
            print(f'[yellow]{repo_name}[/yellow] 没有可删除的分析产物')
    else:
        print('[yellow]提示：已保留报告、知识库和向量索引；如需仅清理向量库，可执行 remove-vector。[/yellow]')


@app.command(name='remove-vector', help='remove one repo from vector database only')
def remove_vector(
    repo_name: Annotated[str, typer.Argument(help='目标仓库标识，格式为 owner/repo')],
) -> None:
    """仅删除指定仓库在向量数据库中的索引。"""
    try:
        removed = remove_vector_indexed_repo(repo_name)
    except Exception as exc:
        print(f'[red]向量索引删除失败：{exc}[/red]')
        return

    if removed:
        print(f'[green]{repo_name}[/green] 的向量索引已删除')
    else:
        print(f'[yellow]{repo_name}[/yellow] 在向量数据库中不存在，或当前未启用向量库')


@app.command(name='vector-health', help='check current vector database health')
def vector_health() -> None:
    """检查当前 Chroma 向量库的健康状态。"""
    _render_vector_health(check_vector_store_health())


@app.command(name='embedding-health', help='check current embedding service health')
def embedding_health(
    embedding_mode: Annotated[
        str,
        typer.Option('--embedding-mode', help='embedding 模式，可选 service、ollama 或 sentence-transformers'),
    ] = 'service',
) -> None:
    """检查当前 embedding 服务是否可用。"""
    if not _apply_embedding_mode(embedding_mode):
        return
    _render_embedding_health(check_embedding_health())


@app.command(name='rebuild-vector', help='rebuild vector database from local knowledge docs')
def rebuild_vector(
    embedding_mode: Annotated[
        str,
        typer.Option('--embedding-mode', help='embedding 模式，可选 service、ollama 或 sentence-transformers'),
    ] = 'service',
) -> None:
    """删除旧向量库并根据本地知识文档重建。"""
    if not _apply_embedding_mode(embedding_mode):
        return
    print('[bold]重建前状态[/bold]')
    before = get_vector_store_rebuild_overview()
    _render_vector_health(before)

    result = rebuild_vector_store(health_snapshot=before)
    if result.success:
        print(f'[green]{result.message}[/green]')
    else:
        print(f'[red]{result.message}[/red]')
        if result.error:
            print(f'[yellow]错误详情[/yellow]：{result.error}')
        return

    print(f'[bold]向量库目录[/bold]：{result.store_path}')
    print(f'[bold]删除旧库[/bold]：{"是" if result.removed_existing_store else "否"}')
    print(f'[bold]写入仓库数[/bold]：{result.indexed_repo_count}')
    print(f'[bold]写入文档数[/bold]：{result.indexed_document_count}')

    print('[bold]重建后状态[/bold]')
    _render_vector_health(check_vector_store_health())



@app.command(name='export', help='export one saved report to another format')
def export_report(
    repo_name: Annotated[str, typer.Argument(help='目标仓库标识，格式为 owner/repo')],
    format: Annotated[
        str,
        typer.Option('--format', help='导出格式，当前仅支持 pdf'),
    ] = 'pdf',
    output_dir: Annotated[
        str,
        typer.Option('--output-dir', help='报告输出目录'),
    ] = 'reports',
) -> None:
    """把已保存的 Markdown 报告导出为其他格式。"""
    normalized_format = format.strip().lower()
    if normalized_format != 'pdf':
        print(f'[red]暂不支持导出格式：{format}[/red]')
        print('[yellow]当前仅支持：pdf[/yellow]')
        return

    try:
        pdf_path = export_repo_report_to_pdf(repo_id=repo_name, output_dir=output_dir)
    except FileNotFoundError as exc:
        print(f'[red]导出失败：{exc}[/red]')
        print('[yellow]请先执行 analyze 生成 Markdown 报告[/yellow]')
        return
    except Exception as exc:
        print(f'[red]导出失败：{exc}[/red]')
        return

    print(f'[green]PDF 报告已导出[/green]：{pdf_path}')


@app.command(name='list', help='list all cloned repos')
def list_repos() -> None:
    """列出本地 clone 缓存中的所有仓库。"""
    result = list_cloned_repos()
    if result.total_count == 0:
        print('[yellow]当前没有已缓存的仓库[/yellow]')
        return

    table = Table(title='仓库资产总览', show_lines=False)
    table.add_column('仓库', style='cyan', no_wrap=True)
    table.add_column('状态', style='bright_white', no_wrap=True)
    table.add_column('Clone', style='green', no_wrap=True)
    table.add_column('报告', style='yellow', no_wrap=True)
    table.add_column('知识库', style='blue', no_wrap=True)
    table.add_column('向量库', style='magenta', no_wrap=True)
    table.add_column('最后修改时间', style='green')
    table.add_column('大小', justify='right', style='magenta')
    table.add_column('本地路径', style='white')

    for repo in result.repos:
        table.add_row(
            repo.repo_id,
            repo.asset_status,
            '有' if repo.has_clone else '无',
            _render_report_status(repo),
            '有' if repo.has_knowledge else '无',
            '有' if repo.has_vector_index else '无',
            _format_datetime(repo.last_modified),
            _format_size(repo.size_bytes),
            repo.local_path or '无',
        )

    print(table)
    print(f'[bold]总数[/bold]：{result.total_count}')
    print(f'[bold]clone 根目录[/bold]：{result.clone_root}')


@app.command(name='cleanup-orphans', help='cleanup orphaned reports, knowledge docs and vector indexes')
def cleanup_orphans(
    dry_run: Annotated[
        bool,
        typer.Option('--dry-run/--execute', help='是否只预览孤儿资产，不实际删除'),
    ] = False,
    output_dir: Annotated[
        str,
        typer.Option('--output-dir', help='报告输出目录'),
    ] = 'reports',
) -> None:
    """清理没有本地 clone、但仍残留报告/知识库/向量索引的孤儿资产。"""
    result = list_cloned_repos()
    orphan_repos = [
        repo
        for repo in result.repos
        if not repo.has_clone
        and (
            repo.has_markdown_report
            or repo.has_json_report
            or repo.has_pdf_report
            or repo.has_llm_context
            or repo.has_knowledge
            or repo.has_vector_index
        )
    ]

    if not orphan_repos:
        print('[green]当前没有孤儿资产需要清理[/green]')
        return

    if dry_run:
        print('[yellow]以下仓库存在孤儿资产（仅预览，未删除）[/yellow]')
        for repo in orphan_repos:
            print(
                f'- {repo.repo_id} | 状态：{repo.asset_status} | '
                f'报告：{_render_report_status(repo)} | 知识库：{"有" if repo.has_knowledge else "无"} | '
                f'向量库：{"有" if repo.has_vector_index else "无"}'
            )
        print(f'[bold]孤儿仓库数[/bold]：{len(orphan_repos)}')
        return

    removed_count = 0
    for repo in orphan_repos:
        markdown_removed = remove_markdown_report(repo.repo_id, output_dir=output_dir)
        json_removed = remove_json_report(repo.repo_id, output_dir=output_dir)
        pdf_removed = remove_pdf_report(repo.repo_id, output_dir=output_dir)
        llm_context_removed = remove_llm_context_text(repo.repo_id, output_dir=output_dir)
        knowledge_removed = remove_indexed_repo(repo.repo_id)
        if markdown_removed or json_removed or pdf_removed or llm_context_removed or knowledge_removed:
            removed_count += 1
            print(f'[green]已清理[/green]：{repo.repo_id}（{repo.asset_status}）')
        else:
            print(f'[yellow]未清理到可删除内容[/yellow]：{repo.repo_id}')

    print(f'[bold]已处理孤儿仓库数[/bold]：{removed_count}/{len(orphan_repos)}')


@app.command(name='search', help='search analyzed repositories')
def search(
    query: Annotated[str, typer.Argument(help='检索问题或关键词')],
    top_k: Annotated[int, typer.Option('--top-k', min=1, help='返回结果数量')] = 5,
    embedding_mode: Annotated[
        str,
        typer.Option('--embedding-mode', help='embedding 模式，可选 service、ollama 或 sentence-transformers'),
    ] = 'service',
) -> None:
    """在本地知识库中检索已分析仓库。"""
    if not _apply_embedding_mode(embedding_mode):
        return
    result = search_knowledge_base(query=query, top_k=top_k)
    if not result.hits:
        print('[yellow]当前没有命中结果，请先执行 analyze 建立知识索引[/yellow]')
        print(f'[bold]检索后端[/bold]：{result.backend}')
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
    print(f'[bold]检索后端[/bold]：{result.backend}')
    print(f'[bold]知识库仓库数[/bold]：{result.repo_count}')
    print(f'[bold]知识库文档数[/bold]：{result.document_count}')


@app.command(name='answer', help='answer a question for one analyzed repository')
def answer(
    repo_name: Annotated[str, typer.Argument(help='目标仓库标识，格式为 owner/repo')],
    question: Annotated[str, typer.Argument(help='希望询问的问题')],
    top_k: Annotated[int, typer.Option('--top-k', min=1, help='检索证据数量')] = 5,
    embedding_mode: Annotated[
        str,
        typer.Option('--embedding-mode', help='embedding 模式，可选 service、ollama 或 sentence-transformers'),
    ] = 'service',
    use_llm: Annotated[
        bool,
        typer.Option('--llm/--no-llm', help='是否启用 LLM 生成最终回答'),
    ] = True,
    as_json: Annotated[
        bool,
        typer.Option('--json', help='是否输出 JSON 结果'),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option('--stream/--no-stream', help='是否在终端流式输出 LLM 回答'),
    ] = True,
    orchestrator: Annotated[
        str,
        typer.Option('--orchestrator', help='问答编排器，可选 local 或 langgraph'),
    ] = 'local',
) -> None:
    """针对单个已分析仓库做基于检索的 MVP 回答。"""
    if not _apply_embedding_mode(embedding_mode):
        return
    stream_callback = None
    should_stream = bool(stream and use_llm and not as_json)
    if should_stream:
        print('[bold green]LLM 正在流式生成回答...[/bold green]')
        stream_callback = _build_stream_callback()

    normalized_orchestrator = orchestrator.strip().lower()
    if normalized_orchestrator not in {'local', 'langgraph'}:
        print(f'[red]不支持的编排器：{orchestrator}[/red]')
        print('[yellow]当前可选：local, langgraph[/yellow]')
        return
    try:
        if normalized_orchestrator == 'langgraph':
            coordinated = run_langgraph_answer(
                repo_id=repo_name,
                question=question,
                top_k=top_k,
                use_llm=use_llm,
                llm_stream=should_stream,
                on_llm_chunk=stream_callback,
            )
        else:
            coordinated = run_multi_agent_answer(
                repo_id=repo_name,
                question=question,
                top_k=top_k,
                use_llm=use_llm,
                llm_stream=should_stream,
                on_llm_chunk=stream_callback,
            )
    except LangGraphUnavailableError as exc:
        print(f'[red]LangGraph 编排不可用：{exc}[/red]')
        return
    result = coordinated.answer_result

    if as_json:
        _print_json(coordinated.model_dump(exclude_none=True))
        return

    if should_stream and result.answer_mode == 'llm':
        console.print()
    else:
        print(
            Panel(
                result.answer,
                title=f'[bold green]仓库问答：{repo_name}',
                border_style='cyan',
            )
        )

    print(f'[bold]回答模式[/bold]：{result.answer_mode}')
    print(f'[bold]编排器[/bold]：{normalized_orchestrator}')
    print(f'[bold]检索后端[/bold]：{result.backend}')
    print(f'[bold]是否降级[/bold]：{"是" if result.fallback_used else "否"}')
    print(f'[bold]问题焦点[/bold]：{coordinated.route_decision.focus}')
    print(f'[bold]检索命中数[/bold]：{coordinated.retrieval_hit_count}')
    print(f'[bold]LLM 开关[/bold]：{"开启" if result.llm_enabled else "关闭"}')
    print(f'[bold]LLM 配置[/bold]：{"已配置" if get_llm_settings() else "未配置"}')
    print(f'[bold]LLM 尝试[/bold]：{"是" if result.llm_attempted else "否"}')
    if result.llm_error:
        print(f'[yellow]LLM 回退原因[/yellow]：{result.llm_error}')

    _render_answer_agent_trace(coordinated)
    _render_code_investigation(coordinated.code_investigation)
    _render_answer_verification(coordinated.verification_result)

    if not result.evidence:
        print('[yellow]当前没有可展示的证据，请先执行 analyze 建立知识索引[/yellow]')
        if result.llm_enabled and get_llm_settings() is None:
            print(f'[yellow]{get_llm_config_help_text()}[/yellow]')
        return

    table = Table(title='回答证据', show_lines=False)
    table.add_column('类型', style='green', no_wrap=True)
    table.add_column('来源', style='yellow')
    table.add_column('摘要片段', style='white')

    for evidence in result.evidence:
        table.add_row(
            evidence.doc_type,
            evidence.source_path or '仓库级文档',
            evidence.snippet,
        )

    print(table)
    if result.llm_enabled and get_llm_settings() is None:
        print(f'[yellow]{get_llm_config_help_text()}[/yellow]')



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
        f"[b cyan]项目标记:[/] {_join_or_none(profile.project_markers)}\n"
        f"[b cyan]子项目:[/] {_join_or_none([item.root_path for item in profile.subprojects])}\n"
        f"[b cyan]关键符号数:[/] {len(profile.code_symbols)}\n"
        f"[b cyan]模块依赖数:[/] {len(profile.module_relations)}\n"
        f"[b cyan]函数级摘要数:[/] {len(profile.function_summaries)}\n"
        f"[b cyan]类级摘要数:[/] {len(profile.class_summaries)}"
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
    table.add_column('证据强度', style='yellow', no_wrap=True)
    table.add_column('证据来源', style='magenta', no_wrap=True)
    table.add_column('证据', style='white')

    for item in result.tech_stack:
        table.add_row(item.name, item.category, item.evidence_level, item.evidence_source, item.evidence)

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


def _render_analysis_agent_trace(result: CoordinatedAnalysisResult) -> None:
    """输出 analyze 阶段的 Agent 执行轨迹。"""
    if not result.agent_trace:
        return

    table = Table(title='分析 Agent 轨迹', show_lines=False)
    table.add_column('Agent', style='cyan', no_wrap=True)
    table.add_column('状态', style='green', no_wrap=True)
    table.add_column('尝试', style='magenta', no_wrap=True)
    table.add_column('耗时', style='blue', no_wrap=True)
    table.add_column('阶段', style='yellow')
    table.add_column('说明', style='white')

    for record in result.agent_trace:
        stage_text = ', '.join(record.completed_stage_names or record.stage_names) or '无'
        detail_text = record.detail or record.error_message or '无'
        duration_text = f'{record.duration_ms} ms' if record.duration_ms is not None else '未知'
        attempt_text = str(record.attempt_count)
        if record.used_retry:
            attempt_text += ' (重试)'
        table.add_row(record.display_name, record.status, attempt_text, duration_text, stage_text, detail_text)

    print(table)
    if result.shared_context:
        context_lines = [
            f'[b cyan]{key}:[/] {value}'
            for key, value in result.shared_context.items()
        ]
        print(Panel('\n'.join(context_lines), title='[bold green]分析共享上下文', border_style='yellow'))


def _render_answer_agent_trace(result: CoordinatedAnswerResult) -> None:
    """输出问答阶段的 Agent 执行轨迹。"""
    if not result.agent_trace:
        return

    table = Table(title='问答 Agent 轨迹', show_lines=False)
    table.add_column('Agent', style='cyan', no_wrap=True)
    table.add_column('状态', style='green', no_wrap=True)
    table.add_column('尝试', style='magenta', no_wrap=True)
    table.add_column('耗时', style='blue', no_wrap=True)
    table.add_column('阶段', style='yellow')
    table.add_column('说明', style='white')

    for record in result.agent_trace:
        stage_text = ', '.join(record.completed_stage_names or record.stage_names) or '无'
        detail_text = record.detail or record.error_message or '无'
        duration_text = f'{record.duration_ms} ms' if record.duration_ms is not None else '未知'
        attempt_text = str(record.attempt_count)
        if record.used_retry:
            attempt_text += ' (重试)'
        table.add_row(record.display_name, record.status, attempt_text, duration_text, stage_text, detail_text)

    print(table)
    if result.shared_context:
        context_lines = [
            f'[b cyan]{key}:[/] {value}'
            for key, value in result.shared_context.items()
        ]
        print(Panel('\n'.join(context_lines), title='[bold green]共享上下文', border_style='yellow'))


def _render_code_investigation(result: CodeInvestigationResult | None) -> None:
    """输出 code_agent 的代码调查结果。"""
    if result is None:
        return

    output = (
        f"[b cyan]调查摘要:[/] {result.summary}\n"
        f"[b cyan]置信度:[/] {result.confidence_level}\n"
        f"[b cyan]相关性评分:[/] {result.relevance_score:.2f}\n"
        f"[b cyan]缓存命中:[/] {'是' if result.cache_hit else '否'}\n"
        f"[b cyan]恢复扩检:[/] {'已提升' if result.recovery_improved else ('已尝试' if result.recovery_attempted else '未触发')}\n"
        f"[b cyan]命中符号:[/] {_join_or_none(result.matched_symbols)}\n"
        f"[b cyan]命中路由:[/] {_join_or_none(result.matched_routes)}\n"
        f"[b cyan]源码路径:[/] {_join_or_none(result.source_paths)}\n"
        f"[b cyan]关键位置:[/] {_join_or_none(result.evidence_locations)}\n"
        f"[b cyan]下游调用:[/] {_join_or_none(result.called_symbols)}\n"
        f"[b cyan]关系链:[/] {_join_or_none(result.relation_chains)}"
    )
    print(Panel(output, title='[bold green]代码调查', border_style='blue'))
    if result.quality_notes:
        print(
            Panel(
                '\n'.join(f'- {item}' for item in result.quality_notes),
                title='[bold green]质量评估',
                border_style='yellow',
            )
        )

    if result.trace_steps:
        table = Table(title='源码追踪步骤', show_lines=False)
        table.add_column('深度', style='magenta', no_wrap=True)
        table.add_column('类型', style='cyan', no_wrap=True)
        table.add_column('标签', style='green')
        table.add_column('上游', style='blue')
        table.add_column('位置', style='yellow')
        table.add_column('摘要', style='white')
        for step in result.trace_steps:
            table.add_row(
                str(step.depth),
                step.step_kind,
                step.label,
                step.parent_label or '入口',
                step.location or step.source_path or '未知',
                step.summary,
            )
        print(table)

    snippets = [step for step in result.trace_steps if step.snippet][:2]
    for step in snippets:
        print(
            Panel(
                step.snippet or '',
                title=f'[bold green]源码片段：{step.label}',
                border_style='magenta',
            )
        )


def _render_answer_verification(result: AnswerVerificationResult | None) -> None:
    """输出 verifier_agent 的回答一致性检查结果。"""
    if result is None:
        return

    output = (
        f"[b cyan]验证结论:[/] {result.verdict}\n"
        f"[b cyan]支撑评分:[/] {result.support_score:.2f}\n"
        f"[b cyan]已支撑结论:[/] {result.supported_claim_count}/{result.checked_claim_count}\n"
        f"[b cyan]原因标签:[/] {_join_or_none(result.issue_tags)}"
    )
    print(Panel(output, title='[bold green]回答验证', border_style='green'))
    if result.issues:
        print(
            Panel(
                '\n'.join(f'- {item}' for item in result.issues),
                title='[bold green]验证风险',
                border_style='yellow',
            )
        )
    if result.notes:
        print(
            Panel(
                '\n'.join(f'- {item}' for item in result.notes),
                title='[bold green]验证说明',
                border_style='blue',
            )
        )



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


def _apply_embedding_mode(embedding_mode: str) -> bool:
    """应用当前命令的 embedding 模式覆盖值，非法时给出明确提示。"""
    try:
        set_embedding_provider_override(embedding_mode)
    except EmbeddingProviderSwitchError as exc:
        print(f'[red]{exc}[/red]')
        print('[yellow]当前可选：service、ollama、sentence-transformers[/yellow]')
        return False
    return True


def _render_vector_health(result) -> None:
    """渲染向量库健康状态，便于在终端中快速判断是否需要重建。"""
    status = '[green]健康[/green]' if result.healthy else '[red]异常[/red]'
    runtime = '可用' if result.runtime_available else '不可用'
    exists = '存在' if result.store_exists else '不存在'
    embedding_settings = get_embedding_settings()

    print(Panel(
        (
            f'[b cyan]状态:[/] {status}\n'
            f'[b cyan]运行时:[/] {runtime}\n'
            f'[b cyan]目录:[/] {exists}\n'
            f'[b cyan]Embedding 模式:[/] {embedding_settings.provider}\n'
            f'[b cyan]Embedding 模型:[/] {embedding_settings.model}\n'
            f'[b cyan]向量库路径:[/] {result.store_path}\n'
            f'[b cyan]知识库仓库数:[/] {result.knowledge_repo_count}\n'
            f'[b cyan]知识库文档数:[/] {result.knowledge_document_count}\n'
            f'[b cyan]已索引仓库数:[/] {result.indexed_repo_count}\n'
            f'[b cyan]已索引文档数:[/] {result.indexed_document_count}\n'
            f'[b cyan]说明:[/] {result.message}'
        ),
        title='[bold green]向量库健康检查',
        border_style='magenta',
    ))
    if result.error:
        print(f'[yellow]错误详情[/yellow]：{result.error}')


def _render_embedding_health(result) -> None:
    """渲染 embedding 健康状态，便于快速判断当前 provider 是否可用。"""
    status = '[green]健康[/green]' if result.healthy else '[red]异常[/red]'
    output = (
        f'[b cyan]状态:[/] {status}\n'
        f'[b cyan]Embedding 模式:[/] {result.provider}\n'
        f'[b cyan]Embedding 模型:[/] {result.model}\n'
        f'[b cyan]Base URL:[/] {result.base_url or "无"}\n'
        f'[b cyan]耗时:[/] {f"{result.latency_ms} ms" if result.latency_ms is not None else "未知"}\n'
        f'[b cyan]向量维度:[/] {result.vector_size if result.vector_size is not None else "未知"}\n'
        f'[b cyan]说明:[/] {result.message}'
    )
    print(Panel(output, title='[bold green]Embedding 健康检查', border_style='blue'))
    if result.error:
        print(f'[yellow]错误详情[/yellow]：{result.error}')


def _render_report_status(repo) -> str:
    """把多种报告状态压缩成紧凑展示文本。"""
    parts: list[str] = []
    if repo.has_markdown_report:
        parts.append('MD')
    if repo.has_json_report:
        parts.append('JSON')
    if repo.has_pdf_report:
        parts.append('PDF')
    if repo.has_llm_context:
        parts.append('LLM')
    return '/'.join(parts) if parts else '无'


def _print_json(payload: object) -> None:
    """输出结构化 JSON，便于后续脚本调用。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_stream_callback():
    """构造一个适合 Rich 终端流式输出的回调。"""
    def _callback(chunk: str) -> None:
        console.print(chunk, end='', markup=False, highlight=False)

    return _callback


if __name__ == '__main__':
    app()
