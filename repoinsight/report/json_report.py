import json
from datetime import datetime
from pathlib import Path
from typing import Any

from repoinsight.models.analysis_model import AnalysisRunResult, KeyFileContent


def generate_json_report_payload(
    result: AnalysisRunResult,
    max_content_chars: int = 4000,
) -> dict[str, Any]:
    """生成适合程序消费的 JSON 报告结构。"""
    repo = result.repo_info.repo_model
    profile = result.project_profile
    stats = result.scan_result.stats

    return {
        'repo': {
            'owner': repo.owner,
            'name': repo.name,
            'full_name': repo.full_name,
            'html_url': repo.html_url,
            'description': repo.description,
            'default_branch': repo.default_branch,
            'primary_language': repo.primary_language,
            'languages': repo.languages,
            'topics': repo.topics,
            'license_name': repo.license_name,
            'stargazers_count': repo.stargazers_count,
            'forks_count': repo.forks_count,
            'watchers_count': repo.watchers_count,
            'open_issues_count': repo.open_issues_count,
            'created_at': _format_datetime(repo.created_at),
            'updated_at': _format_datetime(repo.updated_at),
            'pushed_at': _format_datetime(repo.pushed_at),
        },
        'local': {
            'clone_path': result.clone_path,
        },
        'insights': {
            'project_type': result.project_type,
            'project_type_evidence': result.project_type_evidence,
            'observations': result.observations,
            'strengths': result.strengths,
            'risks': result.risks,
        },
        'project_profile': {
            'primary_language': profile.primary_language,
            'languages': profile.languages,
            'runtimes': profile.runtimes,
            'frameworks': profile.frameworks,
            'build_tools': profile.build_tools,
            'package_managers': profile.package_managers,
            'test_tools': profile.test_tools,
            'ci_cd_tools': profile.ci_cd_tools,
            'deploy_tools': profile.deploy_tools,
            'entrypoints': profile.entrypoints,
            'project_markers': profile.project_markers,
            'subprojects': [
                {
                    'root_path': item.root_path,
                    'language_scope': item.language_scope,
                    'project_kind': item.project_kind,
                    'config_paths': item.config_paths,
                    'entrypoint_paths': item.entrypoint_paths,
                    'markers': item.markers,
                }
                for item in profile.subprojects
            ],
            'code_symbols': [
                {
                    'name': item.name,
                    'symbol_type': item.symbol_type,
                    'source_path': item.source_path,
                    'line_number': item.line_number,
                }
                for item in profile.code_symbols
            ],
            'module_relations': [
                {
                    'source_path': item.source_path,
                    'target': item.target,
                    'relation_type': item.relation_type,
                    'line_number': item.line_number,
                }
                for item in profile.module_relations
            ],
            'function_summaries': [
                {
                    'name': item.name,
                    'qualified_name': item.qualified_name,
                    'source_path': item.source_path,
                    'language_scope': item.language_scope,
                    'line_start': item.line_start,
                    'line_end': item.line_end,
                    'signature': item.signature,
                    'owner_class': item.owner_class,
                    'is_async': item.is_async,
                    'decorators': item.decorators,
                    'parameters': item.parameters,
                    'called_symbols': item.called_symbols,
                    'return_signals': item.return_signals,
                    'summary': item.summary,
                }
                for item in profile.function_summaries
            ],
            'class_summaries': [
                {
                    'name': item.name,
                    'qualified_name': item.qualified_name,
                    'source_path': item.source_path,
                    'language_scope': item.language_scope,
                    'line_start': item.line_start,
                    'line_end': item.line_end,
                    'bases': item.bases,
                    'decorators': item.decorators,
                    'methods': item.methods,
                    'summary': item.summary,
                }
                for item in profile.class_summaries
            ],
            'api_route_summaries': [
                {
                    'route_path': item.route_path,
                    'http_methods': item.http_methods,
                    'source_path': item.source_path,
                    'language_scope': item.language_scope,
                    'framework': item.framework,
                    'handler_name': item.handler_name,
                    'handler_qualified_name': item.handler_qualified_name,
                    'owner_class': item.owner_class,
                    'line_number': item.line_number,
                    'decorators': item.decorators,
                    'called_symbols': item.called_symbols,
                    'summary': item.summary,
                }
                for item in profile.api_route_summaries
            ],
            'confirmed_signals': [
                {
                    'name': item.name,
                    'category': item.category,
                    'evidence': item.evidence,
                    'evidence_level': item.evidence_level,
                    'evidence_source': item.evidence_source,
                    'source_path': item.source_path,
                }
                for item in profile.confirmed_signals
            ],
            'weak_signals': [
                {
                    'name': item.name,
                    'category': item.category,
                    'evidence': item.evidence,
                    'evidence_level': item.evidence_level,
                    'evidence_source': item.evidence_source,
                    'source_path': item.source_path,
                }
                for item in profile.weak_signals
            ],
        },
        'tech_stack': [
            {
                'name': item.name,
                'category': item.category,
                'evidence': item.evidence,
                'evidence_level': item.evidence_level,
                'evidence_source': item.evidence_source,
                'source_path': item.source_path,
            }
            for item in result.tech_stack
        ],
        'scan_summary': {
            'root_path': result.scan_result.root_path,
            'total_seen': stats.total_seen,
            'kept_count': stats.kept_count,
            'ignored_count': stats.ignored_count,
            'key_file_count': stats.key_file_count,
            'tree_preview': result.scan_result.tree_preview,
        },
        'key_files': [
            {
                'path': item.path,
                'name': item.name,
                'size_bytes': item.size_bytes,
                'extension': item.extension,
                'parent_dir': item.parent_dir,
                'subproject_root': _match_subproject_root(result, item.path),
                'code_symbols': _get_file_symbol_payload(result, item.path),
                'module_relations': _get_file_relation_payload(result, item.path),
                'function_summaries': _get_file_function_payload(result, item.path),
                'class_summaries': _get_file_class_payload(result, item.path),
                'api_route_summaries': _get_file_api_route_payload(result, item.path),
            }
            for item in result.scan_result.key_files
        ],
        'key_file_contents': [
            _serialize_key_file_content(item, result=result, max_content_chars=max_content_chars)
            for item in result.key_file_contents
        ],
    }


def generate_json_report_text(
    result: AnalysisRunResult,
    max_content_chars: int = 4000,
) -> str:
    """把 JSON 报告结构转为格式化字符串。"""
    payload = generate_json_report_payload(result, max_content_chars=max_content_chars)
    return json.dumps(payload, ensure_ascii=False, indent=2) + '\n'


def save_json_report(
    result: AnalysisRunResult,
    output_dir: str = 'reports',
    max_content_chars: int = 4000,
) -> Path:
    """把 JSON 报告写入本地文件，并返回文件路径。"""
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    report_root.mkdir(parents=True, exist_ok=True)

    repo = result.repo_info.repo_model
    report_file = report_root / f'{repo.owner}__{repo.name}.json'
    report_file.write_text(
        generate_json_report_text(result, max_content_chars=max_content_chars),
        encoding='utf-8',
    )
    return report_file


def get_json_report_path(repo_id: str, output_dir: str = 'reports') -> Path:
    """根据仓库标识返回 JSON 报告路径。"""
    owner, repo = _parse_repo_id(repo_id)
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    return report_root / f'{owner}__{repo}.json'


def remove_json_report(repo_id: str, output_dir: str = 'reports') -> bool:
    """删除 JSON 报告；成功返回 True，不存在返回 False。"""
    report_path = get_json_report_path(repo_id, output_dir=output_dir)
    if not report_path.exists():
        return False

    report_path.unlink()
    return True


def _serialize_key_file_content(
    item: KeyFileContent,
    result: AnalysisRunResult,
    max_content_chars: int,
) -> dict[str, Any]:
    """把关键文件内容裁剪后转成 JSON 结构。"""
    content = item.content
    content_truncated = item.truncated
    if len(content) > max_content_chars:
        content = content[:max_content_chars]
        content_truncated = True

    return {
        'path': item.path,
        'size_bytes': item.size_bytes,
        'content': content,
        'truncated': content_truncated,
        'subproject_root': _match_subproject_root(result, item.path),
        'code_symbols': _get_file_symbol_payload(result, item.path),
        'module_relations': _get_file_relation_payload(result, item.path),
        'function_summaries': _get_file_function_payload(result, item.path),
        'class_summaries': _get_file_class_payload(result, item.path),
        'api_route_summaries': _get_file_api_route_payload(result, item.path),
    }


def _match_subproject_root(result: AnalysisRunResult, source_path: str) -> str | None:
    """根据文件路径找到最匹配的子项目根目录。"""
    matched_root: str | None = None
    for item in result.project_profile.subprojects:
        root_path = item.root_path
        if root_path == '.':
            if matched_root is None:
                matched_root = root_path
            continue
        if source_path == root_path or source_path.startswith(f'{root_path}/'):
            if matched_root is None or len(root_path) > len(matched_root):
                matched_root = root_path
    return matched_root


def _get_file_symbol_payload(result: AnalysisRunResult, source_path: str) -> list[dict[str, Any]]:
    """提取某个文件对应的符号列表。"""
    payload: list[dict[str, Any]] = []
    for item in result.project_profile.code_symbols:
        if item.source_path != source_path:
            continue
        payload.append(
            {
                'name': item.name,
                'symbol_type': item.symbol_type,
                'line_number': item.line_number,
            }
        )
    return payload


def _get_file_function_payload(result: AnalysisRunResult, source_path: str) -> list[dict[str, Any]]:
    """提取某个文件对应的函数级摘要列表。"""
    payload: list[dict[str, Any]] = []
    for item in result.project_profile.function_summaries:
        if item.source_path != source_path:
            continue
        payload.append(
            {
                'name': item.name,
                'qualified_name': item.qualified_name,
                'line_start': item.line_start,
                'line_end': item.line_end,
                'signature': item.signature,
                'owner_class': item.owner_class,
                'called_symbols': item.called_symbols,
                'summary': item.summary,
            }
        )
    return payload


def _get_file_class_payload(result: AnalysisRunResult, source_path: str) -> list[dict[str, Any]]:
    """提取某个文件对应的类级摘要列表。"""
    payload: list[dict[str, Any]] = []
    for item in result.project_profile.class_summaries:
        if item.source_path != source_path:
            continue
        payload.append(
            {
                'name': item.name,
                'qualified_name': item.qualified_name,
                'line_start': item.line_start,
                'line_end': item.line_end,
                'bases': item.bases,
                'methods': item.methods,
                'summary': item.summary,
            }
        )
    return payload


def _get_file_api_route_payload(result: AnalysisRunResult, source_path: str) -> list[dict[str, Any]]:
    """提取某个文件对应的接口级摘要列表。"""
    payload: list[dict[str, Any]] = []
    for item in result.project_profile.api_route_summaries:
        if item.source_path != source_path:
            continue
        payload.append(
            {
                'route_path': item.route_path,
                'http_methods': item.http_methods,
                'framework': item.framework,
                'handler_name': item.handler_name,
                'handler_qualified_name': item.handler_qualified_name,
                'line_number': item.line_number,
                'called_symbols': item.called_symbols,
                'summary': item.summary,
            }
        )
    return payload


def _get_file_relation_payload(result: AnalysisRunResult, source_path: str) -> list[dict[str, Any]]:
    """提取某个文件对应的模块依赖列表。"""
    payload: list[dict[str, Any]] = []
    for item in result.project_profile.module_relations:
        if item.source_path != source_path:
            continue
        payload.append(
            {
                'target': item.target,
                'relation_type': item.relation_type,
                'line_number': item.line_number,
            }
        )
    return payload


def _format_datetime(value: datetime | None) -> str | None:
    """把 datetime 转成稳定字符串，便于后续 JSON 或 LLM 使用。"""
    if value is None:
        return None
    return value.strftime('%Y-%m-%d %H:%M:%S')


def _parse_repo_id(repo_id: str) -> tuple[str, str]:
    """把 owner/repo 形式的仓库标识拆分成 owner 和 repo。"""
    normalized = repo_id.strip().strip('/')
    parts = [part for part in normalized.split('/') if part]
    if len(parts) != 2:
        raise ValueError('仓库标识格式应为 owner/repo')

    owner, repo = parts
    if owner in {'.', '..'} or repo in {'.', '..'}:
        raise ValueError('仓库标识不合法')

    return owner, repo
