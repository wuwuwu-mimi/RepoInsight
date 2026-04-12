from repoinsight.models.analysis_model import AnalysisRunResult, KeyFileContent
from repoinsight.models.rag_model import ConfigSummary, EntrypointSummary, KnowledgeDocument
from repoinsight.storage.summary_builders import build_config_summaries, build_entrypoint_summaries


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

    for summary in build_config_summaries(result):
        documents.append(_build_config_summary_document(summary, result))

    for summary in build_entrypoint_summaries(result):
        documents.append(_build_entrypoint_summary_document(summary, result))

    for subproject in result.project_profile.subprojects:
        documents.append(_build_subproject_summary_document(result, subproject.root_path))

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
        f"子项目：{_join_or_none(_describe_subprojects(result))}。",
        f"关键符号数量：{len(profile.code_symbols)}。",
        f"模块依赖关系数量：{len(profile.module_relations)}。",
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
        f"subprojects: {_join_or_none(_describe_subprojects(result))}",
        f"code_symbol_names: {_join_or_none(_collect_repo_code_symbol_names(result))}",
        f"module_relation_targets: {_join_or_none(_collect_repo_module_targets(result))}",
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
    structure_metadata = _build_file_structure_metadata(result, item.path)
    lines = [
        f"仓库 {repo_id} 的关键文件 {item.path} 摘要候选内容。",
        f"文件大小：{item.size_bytes} 字节。",
        f"是否截断：{'是' if item.truncated or len(item.content) > MAX_KEY_FILE_CONTENT_CHARS else '否'}。",
        f"所属子项目：{structure_metadata.get('subproject_root') or '无'}。",
        f"关键符号：{_join_or_none(structure_metadata.get('code_symbols', []))}。",
        f"模块依赖：{_join_or_none(structure_metadata.get('module_relations', []))}。",
        '以下内容来自关键文件原文截断，可用于后续检索与 LLM 精炼。',
        '',
        content or '<空内容>',
    ]
    metadata = _build_common_metadata(result)
    metadata.update(structure_metadata)
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


def _build_config_summary_document(
    summary: ConfigSummary,
    result: AnalysisRunResult,
) -> KnowledgeDocument:
    """把统一的配置摘要模型转换成知识文档。"""
    metadata = _build_common_metadata(result)
    metadata.update(
        {
            'source_path': summary.source_path,
            'config_kind': summary.config_kind,
            'language_scope': summary.language_scope,
            'summary_kind': 'config',
            'config_key_points': summary.key_points,
            'config_scripts_or_commands': summary.scripts_or_commands,
            'config_service_dependencies': summary.service_dependencies,
            'config_env_vars': summary.env_vars,
            'config_related_paths': summary.related_paths,
            'subproject_root': summary.subproject_root or '',
            'subproject_markers': summary.subproject_markers,
            'code_symbols': summary.code_symbols,
            'code_symbol_names': _extract_names_from_formatted_items(summary.code_symbols),
            'module_relations': summary.module_relations,
            'module_relation_targets': _extract_targets_from_formatted_relations(summary.module_relations),
        }
    )
    lines = [
        f'仓库 {summary.repo_id} 的配置摘要。',
        f'来源文件：{summary.source_path}',
        f'配置类型：{summary.config_kind}',
        f'语言范围：{summary.language_scope}',
        f'所属子项目：{summary.subproject_root or "无"}',
        f'子项目标签：{_join_or_none(summary.subproject_markers)}',
        f'框架：{_join_or_none(summary.frameworks)}',
        f'包管理器：{_join_or_none(summary.package_managers)}',
        f'构建工具：{_join_or_none(summary.build_tools)}',
        f'测试工具：{_join_or_none(summary.test_tools)}',
        f'部署工具：{_join_or_none(summary.deploy_tools)}',
        f'关键结论：{_join_or_none(summary.key_points)}',
        f'脚本或命令：{_join_or_none(summary.scripts_or_commands)}',
        f'外部服务依赖：{_join_or_none(summary.service_dependencies)}',
        f'环境变量：{_join_or_none(summary.env_vars)}',
        f'相关路径：{_join_or_none(summary.related_paths)}',
        f'关键符号：{_join_or_none(summary.code_symbols)}',
        f'模块依赖：{_join_or_none(summary.module_relations)}',
        f'摘要：{summary.summary}',
        f'证据：{_join_or_none(summary.evidence)}',
    ]
    return KnowledgeDocument(
        doc_id=f'{summary.repo_id}::config_summary::{summary.source_path}',
        repo_id=summary.repo_id,
        doc_type='config_summary',
        title=f'{summary.repo_id}::{summary.source_path} 配置摘要',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_entrypoint_summary_document(
    summary: EntrypointSummary,
    result: AnalysisRunResult,
) -> KnowledgeDocument:
    """把统一的入口摘要模型转换成知识文档。"""
    metadata = _build_common_metadata(result)
    metadata.update(
        {
            'source_path': summary.source_path,
            'entrypoint_kind': summary.entrypoint_kind,
            'language_scope': summary.language_scope,
            'summary_kind': 'entrypoint',
            'entrypoint_startup_commands': summary.startup_commands,
            'entrypoint_dependent_configs': summary.dependent_configs,
            'entrypoint_exposed_interfaces': summary.exposed_interfaces,
            'entrypoint_service_dependencies': summary.service_dependencies,
            'subproject_root': summary.subproject_root or '',
            'subproject_markers': summary.subproject_markers,
            'code_symbols': summary.code_symbols,
            'code_symbol_names': _extract_names_from_formatted_items(summary.code_symbols),
            'module_relations': summary.module_relations,
            'module_relation_targets': _extract_targets_from_formatted_relations(summary.module_relations),
        }
    )
    lines = [
        f'仓库 {summary.repo_id} 的入口摘要。',
        f'来源文件：{summary.source_path}',
        f'入口类型：{summary.entrypoint_kind}',
        f'语言范围：{summary.language_scope}',
        f'所属子项目：{summary.subproject_root or "无"}',
        f'子项目标签：{_join_or_none(summary.subproject_markers)}',
        f'职责：{summary.responsibility}',
        f'启动提示：{_join_or_none(summary.startup_hints)}',
        f'启动命令：{_join_or_none(summary.startup_commands)}',
        f'关联组件：{_join_or_none(summary.related_components)}',
        f'依赖配置：{_join_or_none(summary.dependent_configs)}',
        f'暴露接口：{_join_or_none(summary.exposed_interfaces)}',
        f'外部服务依赖：{_join_or_none(summary.service_dependencies)}',
        f'关键符号：{_join_or_none(summary.code_symbols)}',
        f'模块依赖：{_join_or_none(summary.module_relations)}',
        f'摘要：{summary.summary}',
        f'证据：{_join_or_none(summary.evidence)}',
    ]
    return KnowledgeDocument(
        doc_id=f'{summary.repo_id}::entrypoint_summary::{summary.source_path}',
        repo_id=summary.repo_id,
        doc_type='entrypoint_summary',
        title=f'{summary.repo_id}::{summary.source_path} 入口摘要',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_subproject_summary_document(
    result: AnalysisRunResult,
    root_path: str,
) -> KnowledgeDocument:
    """为 monorepo 或多服务仓库生成子项目级知识文档。"""
    repo_id = result.repo_info.repo_model.full_name
    subproject = next(
        (item for item in result.project_profile.subprojects if item.root_path == root_path),
        None,
    )
    if subproject is None:
        raise ValueError(f'未找到子项目：{root_path}')

    metadata = _build_common_metadata(result)
    metadata.update(
        {
            'source_path': subproject.root_path,
            'subproject_root': subproject.root_path,
            'subproject_markers': subproject.markers,
            'subproject_language_scope': subproject.language_scope,
            'subproject_kind': subproject.project_kind,
            'subproject_config_paths': subproject.config_paths,
            'subproject_entrypoint_paths': subproject.entrypoint_paths,
        }
    )
    lines = [
        f'仓库 {repo_id} 的子项目摘要。',
        f'子项目根目录：{subproject.root_path}',
        f'语言范围：{subproject.language_scope}',
        f'子项目类型：{subproject.project_kind}',
        f'子项目标签：{_join_or_none(subproject.markers)}',
        f'配置文件：{_join_or_none(subproject.config_paths)}',
        f'入口文件：{_join_or_none(subproject.entrypoint_paths)}',
    ]
    return KnowledgeDocument(
        doc_id=f'{repo_id}::subproject_summary::{subproject.root_path}',
        repo_id=repo_id,
        doc_type='subproject_summary',
        title=f'{repo_id}::{subproject.root_path} 子项目摘要',
        content='\n'.join(lines),
        source_path=subproject.root_path,
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
        'project_markers': profile.project_markers,
        'subproject_roots': [item.root_path for item in profile.subprojects],
        'subproject_markers': _collect_all_subproject_markers(result),
        'code_symbol_names': _collect_repo_code_symbol_names(result),
        'module_relation_targets': _collect_repo_module_targets(result),
        'topics': repo.topics,
        'stars': repo.stargazers_count,
    }


def _build_file_structure_metadata(
    result: AnalysisRunResult,
    source_path: str,
) -> dict[str, str | int | float | bool | list[str]]:
    """为单个文件补充结构化元数据，便于精细检索和证据展示。"""
    subproject_root, subproject_markers = _resolve_subproject_context(result, source_path)
    code_symbols = [
        _format_symbol_text(item.name, item.symbol_type, item.line_number)
        for item in result.project_profile.code_symbols
        if item.source_path == source_path
    ]
    module_relations = [
        _format_relation_text(item.target, item.relation_type, item.line_number)
        for item in result.project_profile.module_relations
        if item.source_path == source_path
    ]
    evidence_locations = [
        f'{source_path}:L{item.line_number}'
        for item in result.project_profile.code_symbols
        if item.source_path == source_path and item.line_number is not None
    ]
    return {
        'subproject_root': subproject_root or '',
        'subproject_markers': subproject_markers,
        'code_symbols': code_symbols,
        'code_symbol_names': [item.name for item in result.project_profile.code_symbols if item.source_path == source_path],
        'module_relations': module_relations,
        'module_relation_targets': [item.target for item in result.project_profile.module_relations if item.source_path == source_path],
        'evidence_locations': evidence_locations[:12],
    }


def _resolve_subproject_context(result: AnalysisRunResult, source_path: str) -> tuple[str | None, list[str]]:
    """根据文件路径找到其所属的子项目上下文。"""
    matched_root: str | None = None
    matched_markers: list[str] = []
    for item in result.project_profile.subprojects:
        root_path = item.root_path
        if root_path == '.':
            if matched_root is None:
                matched_root = root_path
                matched_markers = item.markers
            continue
        if source_path == root_path or source_path.startswith(f'{root_path}/'):
            if matched_root is None or len(root_path) > len(matched_root):
                matched_root = root_path
                matched_markers = item.markers
    return matched_root, matched_markers


def _describe_subprojects(result: AnalysisRunResult) -> list[str]:
    """把子项目模型转换成更适合展示的简短文本。"""
    descriptions: list[str] = []
    for item in result.project_profile.subprojects:
        descriptions.append(f'{item.root_path}({item.language_scope}/{item.project_kind})')
    return descriptions


def _collect_all_subproject_markers(result: AnalysisRunResult) -> list[str]:
    """汇总所有子项目标签，便于 metadata 加权。"""
    markers: list[str] = []
    for item in result.project_profile.subprojects:
        markers.extend(item.markers)
    return _unique_keep_order(markers)


def _collect_repo_code_symbol_names(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级符号名，便于技术/架构问题做结构化匹配。"""
    return _unique_keep_order([item.name for item in result.project_profile.code_symbols])[:30]


def _collect_repo_module_targets(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级模块依赖目标，便于架构和技术栈检索。"""
    return _unique_keep_order([item.target for item in result.project_profile.module_relations])[:30]


def _extract_names_from_formatted_items(items: list[str]) -> list[str]:
    """从格式化后的符号文本中提取名称，便于 metadata 精确命中。"""
    names: list[str] = []
    for item in items:
        parts = item.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return _unique_keep_order(names)


def _extract_targets_from_formatted_relations(items: list[str]) -> list[str]:
    """从格式化后的依赖关系文本中提取目标模块名。"""
    targets: list[str] = []
    for item in items:
        parts = item.split()
        if len(parts) >= 2:
            targets.append(parts[1])
    return _unique_keep_order(targets)


def _format_symbol_text(name: str, symbol_type: str, line_number: int | None) -> str:
    """把符号信息格式化为简洁的结构化文本。"""
    if line_number is None:
        return f'{symbol_type} {name}'
    return f'{symbol_type} {name} @L{line_number}'


def _format_relation_text(target: str, relation_type: str, line_number: int | None) -> str:
    """把模块依赖关系格式化为简洁的结构化文本。"""
    if line_number is None:
        return f'{relation_type} {target}'
    return f'{relation_type} {target} @L{line_number}'


def _unique_keep_order(items: list[str]) -> list[str]:
    """对列表做稳定去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _join_or_none(items: list[str]) -> str:
    """把列表拼接成更适合知识文档展示的文本。"""
    return ', '.join(items) if items else '无'
