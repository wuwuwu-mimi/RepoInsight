from pathlib import Path

from repoinsight.models.analysis_model import AnalysisRunResult, KeyFileContent


# 为了给后续大模型留出上下文空间，单个文件内容默认只保留前 3000 个字符。
DEFAULT_MAX_FILE_CHARS = 3000

# 默认最多带入 8 个关键文件，避免 prompt 过长。
DEFAULT_MAX_FILES = 8


def build_llm_context_payload(
    result: AnalysisRunResult,
    max_files: int = DEFAULT_MAX_FILES,
    max_chars_per_file: int = DEFAULT_MAX_FILE_CHARS,
) -> dict[str, object]:
    """构造适合直接传给 LLM 的结构化上下文。"""
    repo = result.repo_info.repo_model
    profile = result.project_profile
    stats = result.scan_result.stats

    return {
        'task': 'analyze_repository',
        'repository': {
            'full_name': repo.full_name,
            'description': repo.description,
            'html_url': repo.html_url,
            'default_branch': repo.default_branch,
            'primary_language': repo.primary_language,
            'topics': repo.topics,
            'license_name': repo.license_name,
            'stars': repo.stargazers_count,
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
        'current_rule_analysis': {
            'project_type': result.project_type,
            'project_type_evidence': result.project_type_evidence,
            'observations': result.observations,
            'strengths': result.strengths,
            'risks': result.risks,
        },
        'scan_summary': {
            'total_seen': stats.total_seen,
            'kept_count': stats.kept_count,
            'ignored_count': stats.ignored_count,
            'key_file_count': stats.key_file_count,
            'tree_preview': result.scan_result.tree_preview[:50],
        },
        'tech_stack_signals': [
            {
                'name': item.name,
                'category': item.category,
                'evidence': item.evidence,
            }
            for item in result.tech_stack
        ],
        'key_file_snippets': [
            _build_file_snippet(item, max_chars=max_chars_per_file)
            for item in result.key_file_contents[:max_files]
        ],
    }


def build_llm_context_text(
    result: AnalysisRunResult,
    max_files: int = DEFAULT_MAX_FILES,
    max_chars_per_file: int = DEFAULT_MAX_FILE_CHARS,
) -> str:
    """把分析结果转换成适合直接拼进 prompt 的文本。"""
    payload = build_llm_context_payload(
        result=result,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
    )

    repository = payload['repository']
    project_profile = payload['project_profile']
    rule_analysis = payload['current_rule_analysis']
    scan_summary = payload['scan_summary']
    tech_stack_signals = payload['tech_stack_signals']
    key_file_snippets = payload['key_file_snippets']

    lines: list[str] = [
        '你将基于以下仓库结构化上下文继续做高层分析。',
        '',
        '[仓库基础信息]',
        f"- full_name: {repository['full_name']}",
        f"- description: {repository['description'] or '无'}",
        f"- html_url: {repository['html_url']}",
        f"- default_branch: {repository['default_branch']}",
        f"- primary_language: {repository['primary_language'] or '未识别'}",
        f"- topics: {_join_or_none(repository['topics'])}",
        f"- license_name: {repository['license_name'] or '无'}",
        f"- stars: {repository['stars']}",
        '',
        '[项目画像]',
        f"- primary_language: {project_profile['primary_language'] or '未识别'}",
        f"- languages: {_join_or_none(project_profile['languages'])}",
        f"- runtimes: {_join_or_none(project_profile['runtimes'])}",
        f"- frameworks: {_join_or_none(project_profile['frameworks'])}",
        f"- build_tools: {_join_or_none(project_profile['build_tools'])}",
        f"- package_managers: {_join_or_none(project_profile['package_managers'])}",
        f"- test_tools: {_join_or_none(project_profile['test_tools'])}",
        f"- ci_cd_tools: {_join_or_none(project_profile['ci_cd_tools'])}",
        f"- deploy_tools: {_join_or_none(project_profile['deploy_tools'])}",
        f"- entrypoints: {_join_or_none(project_profile['entrypoints'])}",
        f"- project_markers: {_join_or_none(project_profile['project_markers'])}",
        '',
        '[当前规则分析]',
        f"- project_type: {rule_analysis['project_type'] or '暂未明确识别'}",
        f"- project_type_evidence: {rule_analysis['project_type_evidence'] or '无'}",
        f"- observations: {_join_or_none(rule_analysis['observations'])}",
        f"- strengths: {_join_or_none(rule_analysis['strengths'])}",
        f"- risks: {_join_or_none(rule_analysis['risks'])}",
        '',
        '[扫描摘要]',
        f"- total_seen: {scan_summary['total_seen']}",
        f"- kept_count: {scan_summary['kept_count']}",
        f"- ignored_count: {scan_summary['ignored_count']}",
        f"- key_file_count: {scan_summary['key_file_count']}",
        f"- tree_preview: {_join_or_none(scan_summary['tree_preview'])}",
        '',
        '[技术栈信号]',
    ]

    if tech_stack_signals:
        for item in tech_stack_signals:
            lines.append(f"- {item['name']} ({item['category']}): {item['evidence']}")
    else:
        lines.append('- 无')

    lines.extend(['', '[关键文件片段]'])
    if key_file_snippets:
        for item in key_file_snippets:
            lines.append(f"## {item['path']}")
            lines.append(f"- size_bytes: {item['size_bytes']}")
            lines.append(f"- truncated: {'是' if item['truncated'] else '否'}")
            lines.append('```text')
            lines.append(item['content'] or '<空内容>')
            lines.append('```')
            lines.append('')
    else:
        lines.append('- 无')

    return '\n'.join(lines).strip() + '\n'


def save_llm_context_text(
    result: AnalysisRunResult,
    output_dir: str = 'reports',
    max_files: int = DEFAULT_MAX_FILES,
    max_chars_per_file: int = DEFAULT_MAX_FILE_CHARS,
) -> Path:
    """把 LLM 上下文文本保存到本地，便于后续直接读取喂给模型。"""
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    report_root.mkdir(parents=True, exist_ok=True)

    repo = result.repo_info.repo_model
    report_file = report_root / f'{repo.owner}__{repo.name}.llm.txt'
    report_file.write_text(
        build_llm_context_text(
            result=result,
            max_files=max_files,
            max_chars_per_file=max_chars_per_file,
        ),
        encoding='utf-8',
    )
    return report_file


def get_llm_context_path(repo_id: str, output_dir: str = 'reports') -> Path:
    """根据仓库标识返回 LLM 上下文文件路径。"""
    owner, repo = _parse_repo_id(repo_id)
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    return report_root / f'{owner}__{repo}.llm.txt'


def remove_llm_context_text(repo_id: str, output_dir: str = 'reports') -> bool:
    """删除 LLM 上下文文件；成功返回 True，不存在返回 False。"""
    report_path = get_llm_context_path(repo_id, output_dir=output_dir)
    if not report_path.exists():
        return False

    report_path.unlink()
    return True


def _build_file_snippet(item: KeyFileContent, max_chars: int) -> dict[str, object]:
    """提取适合 LLM 上下文使用的文件片段。"""
    content = item.content
    truncated = item.truncated
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True

    return {
        'path': item.path,
        'size_bytes': item.size_bytes,
        'content': content,
        'truncated': truncated,
    }


def _join_or_none(items: object) -> str:
    """把列表转换成更适合 prompt 的文本。"""
    if not isinstance(items, list) or not items:
        return '无'
    return ', '.join(str(item) for item in items)


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
