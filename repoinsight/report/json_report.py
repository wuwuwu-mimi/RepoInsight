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
        },
        'tech_stack': [
            {
                'name': item.name,
                'category': item.category,
                'evidence': item.evidence,
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
            }
            for item in result.scan_result.key_files
        ],
        'key_file_contents': [
            _serialize_key_file_content(item, max_content_chars=max_content_chars)
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
    }


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
