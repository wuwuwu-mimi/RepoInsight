from datetime import datetime
from pathlib import Path

from repoinsight.models.analysis_model import AnalysisRunResult


def generate_markdown_report(result: AnalysisRunResult) -> str:
    """根据分析结果生成 Markdown 报告文本。"""
    repo = result.repo_info.repo_model
    profile = result.project_profile
    stats = result.scan_result.stats
    readme_success = bool(result.repo_info.readme and result.repo_info.readme.strip())

    lines: list[str] = [
        f'# RepoInsight Report: {repo.full_name}',
        '',
        '## 1. 项目概览',
        f'- 仓库：`{repo.full_name}`',
        f'- 描述：{repo.description or "无"}',
        f'- Stars：{repo.stargazers_count}',
        f'- 默认分支：`{repo.default_branch}`',
        f'- 主要语言：{repo.primary_language or "未指定"}',
        f'- 最后更新时间：{_format_datetime(repo.updated_at)}',
        f'- 许可证：{repo.license_name or "无"}',
        f'- 仓库地址：{repo.html_url}',
        f'- 本地路径：`{result.clone_path}`',
        f'- README：{"已获取" if readme_success else "未获取"}',
        '',
        '## 2. 项目类型与初步结论',
    ]

    if result.project_type:
        lines.append(f'- 项目类型：{result.project_type}')
        lines.append(f'- 判断依据：{result.project_type_evidence or "无"}')
    else:
        lines.append('- 项目类型：暂未明确识别')

    lines.extend(['', '### 初步观察'])
    if result.observations:
        for item in result.observations:
            lines.append(f'- {item}')
    else:
        lines.append('- 暂无')

    lines.extend(['', '### 优势'])
    if result.strengths:
        for item in result.strengths:
            lines.append(f'- {item}')
    else:
        lines.append('- 暂无')

    lines.extend(['', '### 风险'])
    if result.risks:
        for item in result.risks:
            lines.append(f'- {item}')
    else:
        lines.append('- 暂无')

    lines.extend(['', '## 3. 项目画像'])
    lines.append(f'- 主语言：{profile.primary_language or "未识别"}')
    lines.append(f'- 语言：{_join_or_none(profile.languages)}')
    lines.append(f'- 运行时：{_join_or_none(profile.runtimes)}')
    lines.append(f'- 框架：{_join_or_none(profile.frameworks)}')
    lines.append(f'- 构建工具：{_join_or_none(profile.build_tools)}')
    lines.append(f'- 包管理器：{_join_or_none(profile.package_managers)}')
    lines.append(f'- 测试工具：{_join_or_none(profile.test_tools)}')
    lines.append(f'- CI/CD：{_join_or_none(profile.ci_cd_tools)}')
    lines.append(f'- 部署工具：{_join_or_none(profile.deploy_tools)}')
    lines.append(f'- 入口文件：{_join_or_none(profile.entrypoints)}')
    lines.append(f'- 项目标记：{_join_or_none(profile.project_markers)}')

    lines.extend(['', '## 4. 子项目与结构关系'])
    if profile.subprojects:
        for subproject in profile.subprojects:
            lines.append(f'### `{subproject.root_path}`')
            lines.append(f'- 语言范围：{subproject.language_scope}')
            lines.append(f'- 子项目类型：{subproject.project_kind}')
            lines.append(f'- 标记：{_join_or_none(subproject.markers)}')
            lines.append(f'- 配置文件：{_join_or_none(subproject.config_paths)}')
            lines.append(f'- 入口文件：{_join_or_none(subproject.entrypoint_paths)}')
            lines.append('')
    else:
        lines.append('- 未识别到明确的子项目结构')

    lines.extend(['', '## 5. 代码结构证据'])
    if profile.code_symbols:
        lines.append('### 关键符号')
        for item in profile.code_symbols[:40]:
            lines.append(
                f'- `{item.source_path}` · {item.symbol_type} `{item.name}`'
                f'{_format_line_suffix(item.line_number)}'
            )
    else:
        lines.append('- 未抽取到关键符号')

    if profile.module_relations:
        lines.extend(['', '### 模块依赖'])
        for item in profile.module_relations[:40]:
            lines.append(
                f'- `{item.source_path}` · {item.relation_type} `{item.target}`'
                f'{_format_line_suffix(item.line_number)}'
            )
    else:
        lines.extend(['', '### 模块依赖', '- 未抽取到模块依赖'])

    if profile.function_summaries:
        lines.extend(['', '### 函数级摘要'])
        for item in profile.function_summaries[:20]:
            lines.append(
                f'- `{item.source_path}` · `{item.qualified_name}`'
                f'（{_format_line_range(item.line_start, item.line_end)}）：{item.summary}'
            )
    else:
        lines.extend(['', '### 函数级摘要', '- 未抽取到函数级摘要'])

    if profile.class_summaries:
        lines.extend(['', '### 类级摘要'])
        for item in profile.class_summaries[:12]:
            lines.append(
                f'- `{item.source_path}` · `{item.qualified_name}`'
                f'（{_format_line_range(item.line_start, item.line_end)}）：{item.summary}'
            )
    else:
        lines.extend(['', '### 类级摘要', '- 未抽取到类级摘要'])

    if profile.api_route_summaries:
        lines.extend(['', '### 接口级摘要'])
        for item in profile.api_route_summaries[:20]:
            methods = '/'.join(item.http_methods) if item.http_methods else 'HTTP'
            lines.append(
                f'- `{item.source_path}` -> `{methods} {item.route_path}`'
                f'{_format_line_suffix(item.line_number)}：{item.summary}'
            )
    else:
        lines.extend(['', '### 接口级摘要', '- 未抽取到接口级摘要'])

    lines.extend(['', '## 6. 技术栈推断'])
    if result.tech_stack:
        for item in result.tech_stack:
            level = _translate_evidence_level(item.evidence_level)
            source = item.evidence_source or 'unknown'
            lines.append(f'- {item.name}（{item.category}｜证据强度：{level}｜来源：{source}）：{item.evidence}')
    else:
        lines.append('- 暂未推断出明确的技术栈')

    if result.project_profile.weak_signals:
        lines.extend(['', '### 弱证据候选'])
        for item in result.project_profile.weak_signals[:20]:
            lines.append(f'- {item.name}（{item.category}｜来源：{item.evidence_source}）：{item.evidence}')

    lines.extend(
        [
            '',
            '## 7. 扫描统计',
            f'- 扫描文件总数：{stats.total_seen}',
            f'- 候选文件数：{stats.kept_count}',
            f'- 忽略路径数：{stats.ignored_count}',
            f'- 关键文件数：{stats.key_file_count}',
            '',
            '## 8. 目录树预览',
        ]
    )

    if result.scan_result.tree_preview:
        for item in result.scan_result.tree_preview:
            lines.append(f'- `{item}`')
    else:
        lines.append('- 暂无目录树预览')

    lines.extend(['', '## 9. 关键文件列表'])
    if result.scan_result.key_files:
        for item in result.scan_result.key_files:
            lines.append(f'- `{item.path}`（{_format_size(item.size_bytes)}）')
    else:
        lines.append('- 未识别到关键文件')

    lines.extend(['', '## 10. 关键文件内容预览'])
    if result.key_file_contents:
        for item in result.key_file_contents:
            lines.append(f'### `{item.path}`')
            lines.append(f'- 文件大小：{_format_size(item.size_bytes)}')
            lines.append(f'- 是否截断：{"是" if item.truncated else "否"}')
            lines.append(f'- 所属子项目：{_match_subproject_root(result, item.path) or "无"}')
            lines.append(f'- 关键符号：{_join_or_none(_get_file_symbol_lines(result, item.path))}')
            lines.append(f'- 模块依赖：{_join_or_none(_get_file_relation_lines(result, item.path))}')
            lines.append('')
            lines.append('```text')
            lines.append(item.content or '<空内容>')
            lines.append('```')
            lines.append('')
    else:
        lines.append('- 未读取到关键文件内容')

    return '\n'.join(lines).strip() + '\n'


def save_markdown_report(
    result: AnalysisRunResult,
    output_dir: str = 'reports',
) -> Path:
    """把 Markdown 报告保存到本地文件，并返回保存路径。"""
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    report_root.mkdir(parents=True, exist_ok=True)

    repo = result.repo_info.repo_model
    report_file = report_root / f'{repo.owner}__{repo.name}.md'
    report_file.write_text(generate_markdown_report(result), encoding='utf-8')
    return report_file


def get_report_path(repo_id: str, output_dir: str = 'reports') -> Path:
    """根据仓库标识返回对应的 Markdown 报告路径。"""
    owner, repo = _parse_repo_id(repo_id)
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    return report_root / f'{owner}__{repo}.md'


def remove_markdown_report(repo_id: str, output_dir: str = 'reports') -> bool:
    """删除指定仓库的 Markdown 报告；成功返回 True，不存在返回 False。"""
    report_path = get_report_path(repo_id, output_dir=output_dir)
    if not report_path.exists():
        return False

    report_path.unlink()
    return True


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


def _get_file_symbol_lines(result: AnalysisRunResult, source_path: str) -> list[str]:
    """提取某个文件对应的关键符号摘要。"""
    items: list[str] = []
    for symbol in result.project_profile.code_symbols:
        if symbol.source_path != source_path:
            continue
        items.append(f'{symbol.symbol_type} {symbol.name}{_format_line_suffix(symbol.line_number)}')
    return items[:8]


def _get_file_relation_lines(result: AnalysisRunResult, source_path: str) -> list[str]:
    """提取某个文件对应的模块依赖摘要。"""
    items: list[str] = []
    for relation in result.project_profile.module_relations:
        if relation.source_path != source_path:
            continue
        items.append(f'{relation.relation_type} {relation.target}{_format_line_suffix(relation.line_number)}')
    return items[:8]


def _format_line_range(line_start: int | None, line_end: int | None) -> str:
    """把起止行号格式化成更适合摘要展示的文本。"""
    if line_start is None and line_end is None:
        return '未知'
    if line_start is None:
        return f'L{line_end}'
    if line_end is None or line_end == line_start:
        return f'L{line_start}'
    return f'L{line_start}-L{line_end}'


def _format_line_suffix(line_number: int | None) -> str:
    """把可选行号转换成展示后缀。"""
    if line_number is None:
        return ''
    return f'（L{line_number}）'


def _format_datetime(value: datetime | None) -> str:
    """把时间格式化成更适合写入报告的文本。"""
    if value is None:
        return '未知'
    return value.strftime('%Y-%m-%d %H:%M:%S')


def _format_size(size_bytes: int | None) -> str:
    """把字节数格式化为易读的大小文本。"""
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


def _join_or_none(items: list[str]) -> str:
    """把列表拼接成更适合报告展示的文本。"""
    return ', '.join(items) if items else '无'


def _translate_evidence_level(level: str) -> str:
    """把证据强度枚举翻译成中文。"""
    mapping = {
        'strong': '强',
        'medium': '中',
        'weak': '弱',
    }
    return mapping.get(level, level)
