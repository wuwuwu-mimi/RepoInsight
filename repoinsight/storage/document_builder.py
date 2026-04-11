from repoinsight.models.analysis_model import AnalysisRunResult, KeyFileContent
from repoinsight.models.rag_model import KnowledgeDocument


# 为了让 RAG MVP 更稳定，关键文件摘要默认只保留前 2000 个字符。
MAX_KEY_FILE_CONTENT_CHARS = 2000


def build_knowledge_documents(result: AnalysisRunResult) -> list[KnowledgeDocument]:
    """把一次仓库分析结果拆分为可检索的知识文档。"""
    repo = result.repo_info.repo_model
    repo_id = repo.full_name

    documents: list[KnowledgeDocument] = [
        _build_repo_summary_document(result),
        _build_repo_fact_document(result),
    ]

    if result.repo_info.readme and result.repo_info.readme.strip():
        documents.append(_build_readme_summary_document(result))

    for item in result.key_file_contents:
        documents.append(_build_key_file_summary_document(repo_id, item, result))

    return documents


def _build_repo_summary_document(result: AnalysisRunResult) -> KnowledgeDocument:
    """构建更适合自然语言检索的仓库级摘要文档。"""
    repo = result.repo_info.repo_model
    profile = result.project_profile
    lines = [
        f"仓库 {repo.full_name} 的项目概览。",
        f"项目描述：{repo.description or '无'}。",
        f"项目类型：{result.project_type or '暂未明确识别'}。",
        f"判断依据：{result.project_type_evidence or '无'}。",
        f"主语言：{profile.primary_language or '未识别'}。",
        f"技术栈：{_join_or_none([item.name for item in result.tech_stack])}。",
        f"框架：{_join_or_none(profile.frameworks)}。",
        f"运行时：{_join_or_none(profile.runtimes)}。",
        f"构建工具：{_join_or_none(profile.build_tools)}。",
        f"包管理器：{_join_or_none(profile.package_managers)}。",
        f"入口文件：{_join_or_none(profile.entrypoints)}。",
        f"优势：{_join_or_none(result.strengths)}。",
        f"风险：{_join_or_none(result.risks)}。",
        f"初步观察：{_join_or_none(result.observations)}。",
    ]
    return KnowledgeDocument(
        doc_id=f'{repo.full_name}::repo_summary',
        repo_id=repo.full_name,
        doc_type='repo_summary',
        title=f'{repo.full_name} 仓库摘要',
        content='\n'.join(lines),
        metadata=_build_common_metadata(result),
    )


def _build_repo_fact_document(result: AnalysisRunResult) -> KnowledgeDocument:
    """构建更偏事实层的仓库文档，便于精确检索与后续过滤。"""
    repo = result.repo_info.repo_model
    profile = result.project_profile
    stats = result.scan_result.stats
    lines = [
        f"repo_id: {repo.full_name}",
        f"description: {repo.description or '无'}",
        f"default_branch: {repo.default_branch}",
        f"primary_language: {repo.primary_language or '未指定'}",
        f"languages: {_join_or_none(profile.languages)}",
        f"frameworks: {_join_or_none(profile.frameworks)}",
        f"runtimes: {_join_or_none(profile.runtimes)}",
        f"build_tools: {_join_or_none(profile.build_tools)}",
        f"package_managers: {_join_or_none(profile.package_managers)}",
        f"test_tools: {_join_or_none(profile.test_tools)}",
        f"ci_cd_tools: {_join_or_none(profile.ci_cd_tools)}",
        f"deploy_tools: {_join_or_none(profile.deploy_tools)}",
        f"entrypoints: {_join_or_none(profile.entrypoints)}",
        f"project_markers: {_join_or_none(profile.project_markers)}",
        f"topics: {_join_or_none(repo.topics)}",
        f"stars: {repo.stargazers_count}",
        f"license: {repo.license_name or '无'}",
        f"key_file_count: {stats.key_file_count}",
        f"tree_preview: {_join_or_none(result.scan_result.tree_preview[:20])}",
    ]
    return KnowledgeDocument(
        doc_id=f'{repo.full_name}::repo_fact',
        repo_id=repo.full_name,
        doc_type='repo_fact',
        title=f'{repo.full_name} 仓库事实',
        content='\n'.join(lines),
        metadata=_build_common_metadata(result),
    )


def _build_readme_summary_document(result: AnalysisRunResult) -> KnowledgeDocument:
    """构建 README 文档，先保留原始内容截断版，便于后续逐步引入 LLM 摘要。"""
    repo = result.repo_info.repo_model
    readme_text = (result.repo_info.readme or '').strip()
    readme_text = readme_text[:MAX_KEY_FILE_CONTENT_CHARS]
    content = (
        f"仓库 {repo.full_name} 的 README 摘要候选内容。\n"
        f"以下内容来自 README 原文截断，可用于后续检索与 LLM 精炼。\n\n"
        f"{readme_text}"
    )
    return KnowledgeDocument(
        doc_id=f'{repo.full_name}::readme_summary',
        repo_id=repo.full_name,
        doc_type='readme_summary',
        title=f'{repo.full_name} README 摘要',
        content=content,
        source_path='README',
        metadata=_build_common_metadata(result),
    )


def _build_key_file_summary_document(
    repo_id: str,
    item: KeyFileContent,
    result: AnalysisRunResult,
) -> KnowledgeDocument:
    """把关键文件内容转换成适合检索的知识文档。"""
    content = item.content[:MAX_KEY_FILE_CONTENT_CHARS]
    lines = [
        f"仓库 {repo_id} 的关键文件 {item.path} 摘要候选内容。",
        f"文件大小：{item.size_bytes} 字节。",
        f"是否截断：{'是' if item.truncated or len(item.content) > MAX_KEY_FILE_CONTENT_CHARS else '否'}。",
        '以下内容来自关键文件原文截断，可用于后续检索与 LLM 精炼。',
        '',
        content or '<空内容>',
    ]
    metadata = _build_common_metadata(result)
    metadata['source_path'] = item.path
    return KnowledgeDocument(
        doc_id=f'{repo_id}::key_file::{item.path}',
        repo_id=repo_id,
        doc_type='key_file_summary',
        title=f'{repo_id}::{item.path}',
        content='\n'.join(lines),
        source_path=item.path,
        metadata=metadata,
    )


def _build_common_metadata(result: AnalysisRunResult) -> dict[str, str | int | float | bool | list[str]]:
    """构建所有知识文档都会复用的一组基础元数据。"""
    repo = result.repo_info.repo_model
    profile = result.project_profile
    return {
        'repo_id': repo.full_name,
        'project_type': result.project_type or '',
        'primary_language': profile.primary_language or '',
        'languages': profile.languages,
        'frameworks': profile.frameworks,
        'runtimes': profile.runtimes,
        'build_tools': profile.build_tools,
        'package_managers': profile.package_managers,
        'test_tools': profile.test_tools,
        'deploy_tools': profile.deploy_tools,
        'entrypoints': profile.entrypoints,
        'topics': repo.topics,
        'stars': repo.stargazers_count,
    }


def _join_or_none(items: list[str]) -> str:
    """把列表拼接成更适合知识文档展示的文本。"""
    return ', '.join(items) if items else '无'
