from repoinsight.models.analysis_model import AnalysisRunResult, ApiRouteSummary, ClassSummary, FunctionSummary, KeyFileContent
from repoinsight.models.rag_model import ConfigSummary, EntrypointSummary, KnowledgeDocument
from repoinsight.analyze.code_semantics import infer_call_relation_type
from repoinsight.storage.summary_builders import build_config_summaries, build_entrypoint_summaries


# 为了让 RAG MVP 更稳定，关键文件摘要默认只保留前 2000 个字符。
MAX_KEY_FILE_CONTENT_CHARS = 2000
MAX_CODE_CHUNK_LINES = 24
MAX_CODE_CHUNK_CHARS = 1800
MAX_ROUTE_CHUNK_CONTEXT_BEFORE = 2
MAX_ROUTE_CHUNK_CONTEXT_AFTER = 10
MAX_CONFIG_CHUNK_LINES = 40


def build_knowledge_documents(result: AnalysisRunResult) -> list[KnowledgeDocument]:
    """把一次仓库分析结果拆分为可检索的知识文档。"""
    repo = result.repo_info.repo_model
    repo_id = repo.full_name
    source_lookup = _build_key_file_content_lookup(result)

    documents: list[KnowledgeDocument] = [
        _build_repo_summary_document(result),
        _build_repo_fact_document(result),
    ]

    if result.repo_info.readme and result.repo_info.readme.strip():
        documents.append(_build_readme_summary_document(result))

    for summary in build_config_summaries(result):
        documents.append(_build_config_summary_document(summary, result))
        chunk_document = _build_config_chunk_document(summary, result, source_lookup)
        if chunk_document is not None:
            documents.append(chunk_document)

    for summary in build_entrypoint_summaries(result):
        documents.append(_build_entrypoint_summary_document(summary, result))

    for summary in result.project_profile.api_route_summaries:
        documents.append(_build_api_route_summary_document(summary, result))
        chunk_document = _build_route_handler_chunk_document(summary, result, source_lookup)
        if chunk_document is not None:
            documents.append(chunk_document)

    for summary in result.project_profile.function_summaries:
        documents.append(_build_function_summary_document(summary, result))
        chunk_document = _build_function_body_chunk_document(summary, result, source_lookup)
        if chunk_document is not None:
            documents.append(chunk_document)

    for summary in result.project_profile.class_summaries:
        documents.append(_build_class_summary_document(summary, result))
        chunk_document = _build_class_body_chunk_document(summary, result, source_lookup)
        if chunk_document is not None:
            documents.append(chunk_document)

    for subproject in result.project_profile.subprojects:
        documents.append(_build_subproject_summary_document(result, subproject.root_path))

    for item in result.key_file_contents:
        documents.append(_build_key_file_summary_document(repo_id, item, result))

    return documents


def _build_key_file_content_lookup(result: AnalysisRunResult) -> dict[str, KeyFileContent]:
    """按源码路径建立关键文件内容索引，便于生成代码块级知识文档。"""
    return {
        item.path: item
        for item in result.key_file_contents
    }


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


def _build_api_route_summary_document(
    summary: ApiRouteSummary,
    result: AnalysisRunResult,
) -> KnowledgeDocument:
    """把接口/路由级摘要转换成可检索知识文档。"""
    metadata = _build_common_metadata(result)
    subproject_root, subproject_markers = _resolve_subproject_context(result, summary.source_path)
    methods_text = '/'.join(summary.http_methods) if summary.http_methods else 'HTTP'
    route_ref = f'{methods_text} {summary.route_path}'
    relation_edges = [
        (summary.source_path, route_ref, 'contain_route'),
        (route_ref, summary.handler_qualified_name, 'handle_route'),
        *[
            (summary.handler_qualified_name, called_symbol, infer_call_relation_type(called_symbol))
            for called_symbol in summary.called_symbols
        ],
    ]
    metadata.update(
        {
            'source_path': summary.source_path,
            'summary_kind': 'api_route',
            'route_path': summary.route_path,
            'http_methods': summary.http_methods,
            'framework': summary.framework or '',
            'handler_name': summary.handler_name,
            'handler_qualified_name': summary.handler_qualified_name,
            'owner_class': summary.owner_class or '',
            'language_scope': summary.language_scope,
            'line_number': summary.line_number or 0,
            'decorators': summary.decorators,
            'called_symbols': summary.called_symbols,
            'subproject_root': subproject_root or '',
            'subproject_markers': subproject_markers,
            'code_entity_names': [route_ref, summary.handler_name],
            'code_entity_kinds': ['api_route', 'function'],
            'code_entity_refs': [route_ref, summary.handler_qualified_name],
        }
    )
    metadata.update(_build_relation_metadata(relation_edges))
    lines = [
        f'仓库 {result.repo_info.repo_model.full_name} 中的接口/路由实现摘要。',
        f'来源文件：{summary.source_path}',
        f'路由路径：{summary.route_path}',
        f'HTTP 方法：{methods_text}',
        f'框架线索：{summary.framework or "未知"}',
        f'处理函数：{summary.handler_name}',
        f'处理限定名：{summary.handler_qualified_name}',
        f'所属类：{summary.owner_class or "无"}',
        f'代码位置：{_format_line_range(summary.line_number, summary.line_number)}',
        f'装饰器/注册：{_join_or_none(summary.decorators)}',
        f'调用：{_join_or_none(summary.called_symbols)}',
        f'所属子项目：{subproject_root or "无"}',
        f'摘要：{summary.summary}',
    ]
    return KnowledgeDocument(
        doc_id=(
            f'{result.repo_info.repo_model.full_name}::api_route_summary::'
            f'{summary.source_path}::{methods_text}::{summary.route_path}'
        ),
        repo_id=result.repo_info.repo_model.full_name,
        doc_type='api_route_summary',
        title=f'{result.repo_info.repo_model.full_name}::{methods_text} {summary.route_path} 接口摘要',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_function_summary_document(
    summary: FunctionSummary,
    result: AnalysisRunResult,
) -> KnowledgeDocument:
    """把函数级摘要转换成可检索知识文档。"""
    metadata = _build_common_metadata(result)
    subproject_root, subproject_markers = _resolve_subproject_context(result, summary.source_path)
    relation_edges = [(summary.source_path, summary.qualified_name, 'contain_symbol')]
    if summary.owner_class:
        relation_edges.append((summary.owner_class, summary.qualified_name, 'define_method'))
    relation_edges.extend(
        (summary.qualified_name, called_symbol, infer_call_relation_type(called_symbol))
        for called_symbol in summary.called_symbols
    )
    metadata.update(
        {
            'source_path': summary.source_path,
            'summary_kind': 'function',
            'symbol_name': summary.name,
            'qualified_name': summary.qualified_name,
            'owner_class': summary.owner_class or '',
            'language_scope': summary.language_scope,
            'line_start': summary.line_start or 0,
            'line_end': summary.line_end or 0,
            'signature': summary.signature,
            'decorators': summary.decorators,
            'parameters': summary.parameters,
            'called_symbols': summary.called_symbols,
            'return_signals': summary.return_signals,
            'subproject_root': subproject_root or '',
            'subproject_markers': subproject_markers,
            'code_entity_names': [summary.name],
            'code_entity_kinds': ['function'],
            'code_entity_refs': [summary.qualified_name],
        }
    )
    metadata.update(_build_relation_metadata(relation_edges))
    lines = [
        f'仓库 {result.repo_info.repo_model.full_name} 中的函数/方法实现摘要。',
        f'来源文件：{summary.source_path}',
        f'符号名称：{summary.name}',
        f'限定名：{summary.qualified_name}',
        f'语言范围：{summary.language_scope}',
        f'代码位置：{_format_line_range(summary.line_start, summary.line_end)}',
        f'所属类：{summary.owner_class or "无"}',
        f'函数签名：{summary.signature}',
        f'装饰器：{_join_or_none(summary.decorators)}',
        f'参数：{_join_or_none(summary.parameters)}',
        f'调用：{_join_or_none(summary.called_symbols)}',
        f'返回线索：{_join_or_none(summary.return_signals)}',
        f'所属子项目：{subproject_root or "无"}',
        f'摘要：{summary.summary}',
    ]
    return KnowledgeDocument(
        doc_id=f'{result.repo_info.repo_model.full_name}::function_summary::{summary.source_path}::{summary.qualified_name}',
        repo_id=result.repo_info.repo_model.full_name,
        doc_type='function_summary',
        title=f'{result.repo_info.repo_model.full_name}::{summary.qualified_name} 函数摘要',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_class_summary_document(
    summary: ClassSummary,
    result: AnalysisRunResult,
) -> KnowledgeDocument:
    """把类级摘要转换成可检索知识文档。"""
    metadata = _build_common_metadata(result)
    subproject_root, subproject_markers = _resolve_subproject_context(result, summary.source_path)
    method_refs = _resolve_class_method_refs(summary, result)
    relation_edges = [(summary.source_path, summary.qualified_name, 'contain_symbol')]
    relation_edges.extend((summary.qualified_name, method_ref, 'define_method') for method_ref in method_refs)
    relation_edges.extend((summary.qualified_name, base_name, 'inherit') for base_name in summary.bases)
    metadata.update(
        {
            'source_path': summary.source_path,
            'summary_kind': 'class',
            'symbol_name': summary.name,
            'qualified_name': summary.qualified_name,
            'language_scope': summary.language_scope,
            'line_start': summary.line_start or 0,
            'line_end': summary.line_end or 0,
            'bases': summary.bases,
            'decorators': summary.decorators,
            'class_methods': summary.methods,
            'subproject_root': subproject_root or '',
            'subproject_markers': subproject_markers,
            'code_entity_names': [summary.name] + summary.methods,
            'code_entity_kinds': ['class'] + (['function'] * len(summary.methods)),
            'code_entity_refs': [summary.qualified_name] + method_refs,
        }
    )
    metadata.update(_build_relation_metadata(relation_edges))
    lines = [
        f'仓库 {result.repo_info.repo_model.full_name} 中的类实现摘要。',
        f'来源文件：{summary.source_path}',
        f'类名称：{summary.name}',
        f'限定名：{summary.qualified_name}',
        f'语言范围：{summary.language_scope}',
        f'代码位置：{_format_line_range(summary.line_start, summary.line_end)}',
        f'继承：{_join_or_none(summary.bases)}',
        f'装饰器：{_join_or_none(summary.decorators)}',
        f'方法：{_join_or_none(summary.methods)}',
        f'所属子项目：{subproject_root or "无"}',
        f'摘要：{summary.summary}',
    ]
    return KnowledgeDocument(
        doc_id=f'{result.repo_info.repo_model.full_name}::class_summary::{summary.source_path}::{summary.qualified_name}',
        repo_id=result.repo_info.repo_model.full_name,
        doc_type='class_summary',
        title=f'{result.repo_info.repo_model.full_name}::{summary.qualified_name} 类摘要',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_function_body_chunk_document(
    summary: FunctionSummary,
    result: AnalysisRunResult,
    source_lookup: dict[str, KeyFileContent],
) -> KnowledgeDocument | None:
    """把函数源码切成更接近实现细节的代码块文档。"""
    source_item = source_lookup.get(summary.source_path)
    if source_item is None:
        return None

    chunk_text, chunk_line_start, chunk_line_end = _extract_line_range_chunk(
        source_item.content,
        line_start=summary.line_start,
        line_end=summary.line_end,
        context_before=1,
        context_after=2,
        max_lines=MAX_CODE_CHUNK_LINES,
        max_chars=MAX_CODE_CHUNK_CHARS,
    )
    if not chunk_text:
        return None

    metadata = _build_common_metadata(result)
    subproject_root, subproject_markers = _resolve_subproject_context(result, summary.source_path)
    relation_edges = [(summary.source_path, summary.qualified_name, 'contain_symbol')]
    if summary.owner_class:
        relation_edges.append((summary.owner_class, summary.qualified_name, 'define_method'))
    relation_edges.extend(
        (summary.qualified_name, called_symbol, infer_call_relation_type(called_symbol))
        for called_symbol in summary.called_symbols
    )
    metadata.update(
        {
            'source_path': summary.source_path,
            'summary_kind': 'function_body_chunk',
            'code_chunk_kind': 'function_body',
            'symbol_name': summary.name,
            'qualified_name': summary.qualified_name,
            'owner_class': summary.owner_class or '',
            'language_scope': summary.language_scope,
            'line_start': chunk_line_start,
            'line_end': chunk_line_end,
            'signature': summary.signature,
            'decorators': summary.decorators,
            'parameters': summary.parameters,
            'called_symbols': summary.called_symbols,
            'return_signals': summary.return_signals,
            'subproject_root': subproject_root or '',
            'subproject_markers': subproject_markers,
            'code_entity_names': [summary.name],
            'code_entity_kinds': ['function'],
            'code_entity_refs': [summary.qualified_name],
        }
    )
    metadata.update(_build_relation_metadata(relation_edges))
    lines = [
        f'仓库 {result.repo_info.repo_model.full_name} 中的函数源码片段。',
        f'来源文件：{summary.source_path}',
        f'符号名称：{summary.name}',
        f'限定名：{summary.qualified_name}',
        f'语言范围：{summary.language_scope}',
        f'代码位置：{_format_line_range(chunk_line_start, chunk_line_end)}',
        f'所属类：{summary.owner_class or "无"}',
        f'函数签名：{summary.signature}',
        f'调用：{_join_or_none(summary.called_symbols)}',
        f'返回线索：{_join_or_none(summary.return_signals)}',
        f'所属子项目：{subproject_root or "无"}',
        f'摘要：{summary.summary}',
        '源码片段：',
        '```text',
        chunk_text,
        '```',
    ]
    return KnowledgeDocument(
        doc_id=f'{result.repo_info.repo_model.full_name}::function_body_chunk::{summary.source_path}::{summary.qualified_name}',
        repo_id=result.repo_info.repo_model.full_name,
        doc_type='function_body_chunk',
        title=f'{result.repo_info.repo_model.full_name}::{summary.qualified_name} 函数源码片段',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_class_body_chunk_document(
    summary: ClassSummary,
    result: AnalysisRunResult,
    source_lookup: dict[str, KeyFileContent],
) -> KnowledgeDocument | None:
    """把类定义切成更适合实现类问题召回的代码块文档。"""
    source_item = source_lookup.get(summary.source_path)
    if source_item is None:
        return None

    chunk_text, chunk_line_start, chunk_line_end = _extract_line_range_chunk(
        source_item.content,
        line_start=summary.line_start,
        line_end=summary.line_end,
        context_before=1,
        context_after=2,
        max_lines=MAX_CODE_CHUNK_LINES,
        max_chars=MAX_CODE_CHUNK_CHARS,
    )
    if not chunk_text:
        return None

    metadata = _build_common_metadata(result)
    subproject_root, subproject_markers = _resolve_subproject_context(result, summary.source_path)
    method_refs = _resolve_class_method_refs(summary, result)
    relation_edges = [(summary.source_path, summary.qualified_name, 'contain_symbol')]
    relation_edges.extend((summary.qualified_name, method_ref, 'define_method') for method_ref in method_refs)
    relation_edges.extend((summary.qualified_name, base_name, 'inherit') for base_name in summary.bases)
    metadata.update(
        {
            'source_path': summary.source_path,
            'summary_kind': 'class_body_chunk',
            'code_chunk_kind': 'class_body',
            'symbol_name': summary.name,
            'qualified_name': summary.qualified_name,
            'language_scope': summary.language_scope,
            'line_start': chunk_line_start,
            'line_end': chunk_line_end,
            'bases': summary.bases,
            'decorators': summary.decorators,
            'class_methods': summary.methods,
            'subproject_root': subproject_root or '',
            'subproject_markers': subproject_markers,
            'code_entity_names': [summary.name] + summary.methods,
            'code_entity_kinds': ['class'] + (['function'] * len(summary.methods)),
            'code_entity_refs': [summary.qualified_name] + method_refs,
        }
    )
    metadata.update(_build_relation_metadata(relation_edges))
    lines = [
        f'仓库 {result.repo_info.repo_model.full_name} 中的类源码片段。',
        f'来源文件：{summary.source_path}',
        f'类名称：{summary.name}',
        f'限定名：{summary.qualified_name}',
        f'语言范围：{summary.language_scope}',
        f'代码位置：{_format_line_range(chunk_line_start, chunk_line_end)}',
        f'继承：{_join_or_none(summary.bases)}',
        f'方法：{_join_or_none(summary.methods)}',
        f'所属子项目：{subproject_root or "无"}',
        f'摘要：{summary.summary}',
        '源码片段：',
        '```text',
        chunk_text,
        '```',
    ]
    return KnowledgeDocument(
        doc_id=f'{result.repo_info.repo_model.full_name}::class_body_chunk::{summary.source_path}::{summary.qualified_name}',
        repo_id=result.repo_info.repo_model.full_name,
        doc_type='class_body_chunk',
        title=f'{result.repo_info.repo_model.full_name}::{summary.qualified_name} 类源码片段',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_route_handler_chunk_document(
    summary: ApiRouteSummary,
    result: AnalysisRunResult,
    source_lookup: dict[str, KeyFileContent],
) -> KnowledgeDocument | None:
    """把路由处理附近的源码切成接口实现代码块文档。"""
    source_item = source_lookup.get(summary.source_path)
    if source_item is None:
        return None

    chunk_text, chunk_line_start, chunk_line_end = _extract_line_window_chunk(
        source_item.content,
        line_number=summary.line_number,
        context_before=MAX_ROUTE_CHUNK_CONTEXT_BEFORE,
        context_after=MAX_ROUTE_CHUNK_CONTEXT_AFTER,
        max_lines=MAX_CODE_CHUNK_LINES,
        max_chars=MAX_CODE_CHUNK_CHARS,
    )
    if not chunk_text:
        return None

    metadata = _build_common_metadata(result)
    subproject_root, subproject_markers = _resolve_subproject_context(result, summary.source_path)
    methods_text = '/'.join(summary.http_methods) if summary.http_methods else 'HTTP'
    route_ref = f'{methods_text} {summary.route_path}'
    relation_edges = [
        (summary.source_path, route_ref, 'contain_route'),
        (route_ref, summary.handler_qualified_name, 'handle_route'),
        *[
            (summary.handler_qualified_name, called_symbol, infer_call_relation_type(called_symbol))
            for called_symbol in summary.called_symbols
        ],
    ]
    metadata.update(
        {
            'source_path': summary.source_path,
            'summary_kind': 'route_handler_chunk',
            'code_chunk_kind': 'route_handler',
            'route_path': summary.route_path,
            'http_methods': summary.http_methods,
            'framework': summary.framework or '',
            'handler_name': summary.handler_name,
            'handler_qualified_name': summary.handler_qualified_name,
            'owner_class': summary.owner_class or '',
            'language_scope': summary.language_scope,
            'line_start': chunk_line_start,
            'line_end': chunk_line_end,
            'line_number': summary.line_number or 0,
            'decorators': summary.decorators,
            'called_symbols': summary.called_symbols,
            'subproject_root': subproject_root or '',
            'subproject_markers': subproject_markers,
            'code_entity_names': [route_ref, summary.handler_name],
            'code_entity_kinds': ['api_route', 'function'],
            'code_entity_refs': [route_ref, summary.handler_qualified_name],
        }
    )
    metadata.update(_build_relation_metadata(relation_edges))
    lines = [
        f'仓库 {result.repo_info.repo_model.full_name} 中的接口处理源码片段。',
        f'来源文件：{summary.source_path}',
        f'路由路径：{summary.route_path}',
        f'HTTP 方法：{methods_text}',
        f'处理函数：{summary.handler_name}',
        f'处理限定名：{summary.handler_qualified_name}',
        f'框架线索：{summary.framework or "未知"}',
        f'代码位置：{_format_line_range(chunk_line_start, chunk_line_end)}',
        f'调用：{_join_or_none(summary.called_symbols)}',
        f'所属子项目：{subproject_root or "无"}',
        f'摘要：{summary.summary}',
        '源码片段：',
        '```text',
        chunk_text,
        '```',
    ]
    return KnowledgeDocument(
        doc_id=(
            f'{result.repo_info.repo_model.full_name}::route_handler_chunk::'
            f'{summary.source_path}::{methods_text}::{summary.route_path}'
        ),
        repo_id=result.repo_info.repo_model.full_name,
        doc_type='route_handler_chunk',
        title=f'{result.repo_info.repo_model.full_name}::{methods_text} {summary.route_path} 接口源码片段',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _build_config_chunk_document(
    summary: ConfigSummary,
    result: AnalysisRunResult,
    source_lookup: dict[str, KeyFileContent],
) -> KnowledgeDocument | None:
    """把配置文件原文切出一份配置代码块文档，便于环境与脚本类问题召回。"""
    source_item = source_lookup.get(summary.source_path)
    if source_item is None:
        return None

    chunk_text, chunk_line_start, chunk_line_end = _extract_line_window_chunk(
        source_item.content,
        line_number=1,
        context_before=0,
        context_after=MAX_CONFIG_CHUNK_LINES - 1,
        max_lines=MAX_CONFIG_CHUNK_LINES,
        max_chars=MAX_CODE_CHUNK_CHARS,
    )
    if not chunk_text:
        return None

    metadata = _build_common_metadata(result)
    metadata.update(
        {
            'source_path': summary.source_path,
            'summary_kind': 'config_chunk',
            'code_chunk_kind': 'config',
            'config_kind': summary.config_kind,
            'language_scope': summary.language_scope,
            'config_key_points': summary.key_points,
            'config_scripts_or_commands': summary.scripts_or_commands,
            'config_service_dependencies': summary.service_dependencies,
            'config_env_vars': summary.env_vars,
            'config_related_paths': summary.related_paths,
            'subproject_root': summary.subproject_root or '',
            'subproject_markers': summary.subproject_markers,
            'line_start': chunk_line_start,
            'line_end': chunk_line_end,
            'code_symbols': summary.code_symbols,
            'code_symbol_names': _extract_names_from_formatted_items(summary.code_symbols),
            'module_relations': summary.module_relations,
            'module_relation_targets': _extract_targets_from_formatted_relations(summary.module_relations),
        }
    )
    lines = [
        f'仓库 {summary.repo_id} 中的配置文件片段。',
        f'来源文件：{summary.source_path}',
        f'配置类型：{summary.config_kind}',
        f'语言范围：{summary.language_scope}',
        f'所属子项目：{summary.subproject_root or "无"}',
        f'关键结论：{_join_or_none(summary.key_points)}',
        f'脚本或命令：{_join_or_none(summary.scripts_or_commands)}',
        f'环境变量：{_join_or_none(summary.env_vars)}',
        f'外部服务依赖：{_join_or_none(summary.service_dependencies)}',
        f'摘要：{summary.summary}',
        '配置片段：',
        '```text',
        chunk_text,
        '```',
    ]
    return KnowledgeDocument(
        doc_id=f'{summary.repo_id}::config_chunk::{summary.source_path}',
        repo_id=summary.repo_id,
        doc_type='config_chunk',
        title=f'{summary.repo_id}::{summary.source_path} 配置片段',
        content='\n'.join(lines),
        source_path=summary.source_path,
        metadata=metadata,
    )


def _extract_line_range_chunk(
    content: str,
    *,
    line_start: int | None,
    line_end: int | None,
    context_before: int,
    context_after: int,
    max_lines: int,
    max_chars: int,
) -> tuple[str, int, int]:
    """按行号范围切出源码片段，并保留少量上下文。"""
    lines = content.splitlines()
    if not lines:
        return '', 0, 0

    if line_start is None and line_end is None:
        return _extract_line_window_chunk(
            content,
            line_number=1,
            context_before=0,
            context_after=max_lines - 1,
            max_lines=max_lines,
            max_chars=max_chars,
        )

    actual_start = max(1, (line_start or line_end or 1) - context_before)
    actual_end = min(len(lines), max(line_end or line_start or 1, line_start or 1) + context_after)
    return _slice_lines_with_numbers(
        lines,
        actual_start=actual_start,
        actual_end=actual_end,
        max_lines=max_lines,
        max_chars=max_chars,
    )


def _extract_line_window_chunk(
    content: str,
    *,
    line_number: int | None,
    context_before: int,
    context_after: int,
    max_lines: int,
    max_chars: int,
) -> tuple[str, int, int]:
    """按中心行号截取一段窗口，用于路由和配置等场景。"""
    lines = content.splitlines()
    if not lines:
        return '', 0, 0

    center_line = max(1, line_number or 1)
    actual_start = max(1, center_line - context_before)
    actual_end = min(len(lines), center_line + context_after)
    return _slice_lines_with_numbers(
        lines,
        actual_start=actual_start,
        actual_end=actual_end,
        max_lines=max_lines,
        max_chars=max_chars,
    )


def _slice_lines_with_numbers(
    lines: list[str],
    *,
    actual_start: int,
    actual_end: int,
    max_lines: int,
    max_chars: int,
) -> tuple[str, int, int]:
    """把指定行范围转成带行号的文本，并控制最大长度。"""
    bounded_end = min(actual_end, actual_start + max_lines - 1)
    numbered_lines = [
        f'L{line_number}: {lines[line_number - 1]}'
        for line_number in range(actual_start, bounded_end + 1)
    ]
    chunk_text = '\n'.join(numbered_lines)
    if len(chunk_text) > max_chars:
        chunk_text = chunk_text[: max_chars - 3].rstrip() + '...'
    return chunk_text, actual_start, bounded_end


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
        'code_entity_names': _collect_repo_code_entity_names(result),
        'code_entity_kinds': _collect_repo_code_entity_kinds(result),
        'code_entity_refs': _collect_repo_code_entity_refs(result),
        'code_relation_sources': _collect_repo_code_relation_sources(result),
        'code_relation_targets': _collect_repo_code_relation_targets(result),
        'code_relation_types': _collect_repo_code_relation_types(result),
        'api_route_paths': _collect_repo_api_route_paths(result),
        'api_handler_names': _collect_repo_api_handler_names(result),
        'function_names': _collect_repo_function_names(result),
        'class_names': _collect_repo_class_names(result),
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
    code_entities = [
        _format_code_entity_text(item.entity_kind, item.name, item.qualified_name, item.location)
        for item in result.project_profile.code_entities
        if item.source_path == source_path
    ]
    code_relation_edges = [
        _format_code_relation_edge_text(item.source_ref, item.target_ref, item.relation_type, item.line_number)
        for item in result.project_profile.code_relation_edges
        if item.source_path == source_path
    ]
    return {
        'subproject_root': subproject_root or '',
        'subproject_markers': subproject_markers,
        'code_symbols': code_symbols,
        'code_symbol_names': [item.name for item in result.project_profile.code_symbols if item.source_path == source_path],
        'module_relations': module_relations,
        'module_relation_targets': [item.target for item in result.project_profile.module_relations if item.source_path == source_path],
        'code_entities': code_entities,
        'code_entity_names': [item.name for item in result.project_profile.code_entities if item.source_path == source_path],
        'code_entity_kinds': [item.entity_kind for item in result.project_profile.code_entities if item.source_path == source_path],
        'code_entity_refs': [
            item.qualified_name or item.name
            for item in result.project_profile.code_entities
            if item.source_path == source_path
        ],
        'code_relation_edges': code_relation_edges,
        'code_relation_sources': [
            item.source_ref for item in result.project_profile.code_relation_edges if item.source_path == source_path
        ],
        'code_relation_targets': [
            item.target_ref for item in result.project_profile.code_relation_edges if item.source_path == source_path
        ],
        'code_relation_types': [
            item.relation_type for item in result.project_profile.code_relation_edges if item.source_path == source_path
        ],
        'api_route_paths': [item.route_path for item in result.project_profile.api_route_summaries if item.source_path == source_path],
        'api_handler_names': [item.handler_qualified_name for item in result.project_profile.api_route_summaries if item.source_path == source_path],
        'function_names': [item.qualified_name for item in result.project_profile.function_summaries if item.source_path == source_path],
        'class_names': [item.qualified_name for item in result.project_profile.class_summaries if item.source_path == source_path],
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


def _collect_repo_function_names(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级函数限定名，便于代码实现类问题检索。"""
    return _unique_keep_order([item.qualified_name for item in result.project_profile.function_summaries])[:40]


def _collect_repo_api_route_paths(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级路由路径，便于接口问题检索。"""
    return _unique_keep_order([item.route_path for item in result.project_profile.api_route_summaries])[:40]


def _collect_repo_api_handler_names(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级接口处理函数，便于接口问题检索。"""
    return _unique_keep_order([item.handler_qualified_name for item in result.project_profile.api_route_summaries])[:40]


def _collect_repo_class_names(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级类限定名，便于职责与实现类问题检索。"""
    return _unique_keep_order([item.qualified_name for item in result.project_profile.class_summaries])[:30]


def _collect_repo_module_targets(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级模块依赖目标，便于架构和技术栈检索。"""
    return _unique_keep_order([item.target for item in result.project_profile.module_relations])[:30]


def _collect_repo_code_entity_names(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级统一代码实体名称。"""
    return _unique_keep_order([item.name for item in result.project_profile.code_entities])[:40]


def _collect_repo_code_entity_kinds(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级统一代码实体类型。"""
    return _unique_keep_order([item.entity_kind for item in result.project_profile.code_entities])[:20]


def _collect_repo_code_entity_refs(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级统一代码实体引用名。"""
    return _unique_keep_order(
        [item.qualified_name or item.name for item in result.project_profile.code_entities]
    )[:50]


def _collect_repo_code_relation_sources(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级统一关系边源引用。"""
    return _unique_keep_order([item.source_ref for item in result.project_profile.code_relation_edges])[:50]


def _collect_repo_code_relation_targets(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级统一关系边目标引用。"""
    return _unique_keep_order([item.target_ref for item in result.project_profile.code_relation_edges])[:50]


def _collect_repo_code_relation_types(result: AnalysisRunResult) -> list[str]:
    """汇总仓库级统一关系边类型。"""
    return _unique_keep_order([item.relation_type for item in result.project_profile.code_relation_edges])[:20]




def _build_relation_metadata(
    relation_edges: list[tuple[str, str, str]],
) -> dict[str, list[str]]:
    """把关系边压平成对齐的 metadata 数组，便于后续还原成图结构。"""
    return {
        'code_relation_sources': [source_ref for source_ref, _, _ in relation_edges],
        'code_relation_targets': [target_ref for _, target_ref, _ in relation_edges],
        'code_relation_types': [relation_type for _, _, relation_type in relation_edges],
    }


def _resolve_class_method_refs(
    summary: ClassSummary,
    result: AnalysisRunResult,
) -> list[str]:
    """把类中的方法名映射为限定名，方便统一关系追踪。"""
    method_refs: list[str] = []
    for item in result.project_profile.function_summaries:
        if item.source_path != summary.source_path:
            continue
        if item.owner_class not in {summary.name, summary.qualified_name}:
            continue
        method_refs.append(item.qualified_name)

    if len(method_refs) < len(summary.methods):
        for method_name in summary.methods:
            matched_ref = next(
                (
                    item.qualified_name
                    for item in result.project_profile.function_summaries
                    if item.source_path == summary.source_path and item.name == method_name
                ),
                f'{summary.qualified_name}.{method_name}',
            )
            method_refs.append(matched_ref)
    return _unique_keep_order(method_refs)


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


def _format_line_range(line_start: int | None, line_end: int | None) -> str:
    """把起止行号格式化成更适合展示的文本。"""
    if line_start is None and line_end is None:
        return '未知'
    if line_start is None:
        return f'L{line_end}'
    if line_end is None or line_end == line_start:
        return f'L{line_start}'
    return f'L{line_start}-L{line_end}'


def _format_relation_text(target: str, relation_type: str, line_number: int | None) -> str:
    """把模块依赖关系格式化为简洁的结构化文本。"""
    if line_number is None:
        return f'{relation_type} {target}'
    return f'{relation_type} {target} @L{line_number}'


def _format_code_entity_text(
    entity_kind: str,
    name: str,
    qualified_name: str | None,
    location: str | None,
) -> str:
    """把统一代码实体格式化为简洁文本。"""
    ref = qualified_name or name
    if location:
        return f'{entity_kind} {ref} @{location}'
    return f'{entity_kind} {ref}'


def _format_code_relation_edge_text(
    source_ref: str,
    target_ref: str,
    relation_type: str,
    line_number: int | None,
) -> str:
    """把统一代码关系边格式化为简洁文本。"""
    text = f'{relation_type} {source_ref} -> {target_ref}'
    if line_number is not None:
        return f'{text} @L{line_number}'
    return text


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
