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
        f"# RepoInsight Report: {repo.full_name}",
        "",
        "## 1. 项目概览",
        f"- 仓库：`{repo.full_name}`",
        f"- 描述：{repo.description or '无'}",
        f"- Stars：{repo.stargazers_count}",
        f"- 默认分支：`{repo.default_branch}`",
        f"- 主要语言：{repo.primary_language or '未指定'}",
        f"- 最后更新时间：{_format_datetime(repo.updated_at)}",
        f"- 许可证：{repo.license_name or '无'}",
        f"- 仓库地址：{repo.html_url}",
        f"- 本地路径：`{result.clone_path}`",
        f"- README：{'已获取' if readme_success else '未获取'}",
        "",
        "## 2. 项目类型与初步结论",
    ]

    if result.project_type:
        lines.append(f"- 项目类型：{result.project_type}")
        lines.append(f"- 判断依据：{result.project_type_evidence or '无'}")
    else:
        lines.append("- 项目类型：暂未明确识别")

    if result.observations:
        lines.append("")
        lines.append("### 初步观察")
        for item in result.observations:
            lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("### 初步观察")
        lines.append("- 暂无")

    if result.strengths:
        lines.append("")
        lines.append("### 优势")
        for item in result.strengths:
            lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("### 优势")
        lines.append("- 暂无")

    if result.risks:
        lines.append("")
        lines.append("### 风险")
        for item in result.risks:
            lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("### 风险")
        lines.append("- 暂无")

    lines.extend(["", "## 3. 项目画像"])
    lines.append(f"- 主语言：{profile.primary_language or '未识别'}")
    lines.append(f"- 语言：{_join_or_none(profile.languages)}")
    lines.append(f"- 运行时：{_join_or_none(profile.runtimes)}")
    lines.append(f"- 框架：{_join_or_none(profile.frameworks)}")
    lines.append(f"- 构建工具：{_join_or_none(profile.build_tools)}")
    lines.append(f"- 包管理器：{_join_or_none(profile.package_managers)}")
    lines.append(f"- 测试工具：{_join_or_none(profile.test_tools)}")
    lines.append(f"- CI/CD：{_join_or_none(profile.ci_cd_tools)}")
    lines.append(f"- 部署工具：{_join_or_none(profile.deploy_tools)}")
    lines.append(f"- 入口文件：{_join_or_none(profile.entrypoints)}")
    lines.append(f"- 项目标记：{_join_or_none(profile.project_markers)}")

    lines.extend(["", "## 4. 技术栈推断"])
    if result.tech_stack:
        for item in result.tech_stack:
            lines.append(f"- {item.name}（{item.category}）：{item.evidence}")
    else:
        lines.append("- 暂未推断出明确的技术栈")

    lines.extend(
        [
            "",
            "## 5. 扫描统计",
            f"- 扫描文件总数：{stats.total_seen}",
            f"- 候选文件数：{stats.kept_count}",
            f"- 忽略路径数：{stats.ignored_count}",
            f"- 关键文件数：{stats.key_file_count}",
            "",
            "## 6. 目录树预览",
        ]
    )

    if result.scan_result.tree_preview:
        for item in result.scan_result.tree_preview:
            lines.append(f"- `{item}`")
    else:
        lines.append("- 暂无目录树预览")

    lines.extend(["", "## 7. 关键文件列表"])
    if result.scan_result.key_files:
        for item in result.scan_result.key_files:
            lines.append(f"- `{item.path}`（{_format_size(item.size_bytes)}）")
    else:
        lines.append("- 未识别到关键文件")

    lines.extend(["", "## 8. 关键文件内容预览"])
    if result.key_file_contents:
        for item in result.key_file_contents:
            lines.append(f"### `{item.path}`")
            lines.append(f"- 文件大小：{_format_size(item.size_bytes)}")
            lines.append(f"- 是否截断：{'是' if item.truncated else '否'}")
            lines.append("")
            lines.append("```text")
            lines.append(item.content or "<空内容>")
            lines.append("```")
            lines.append("")
    else:
        lines.append("- 未读取到关键文件内容")

    return "\n".join(lines).strip() + "\n"


def save_markdown_report(
    result: AnalysisRunResult,
    output_dir: str = "reports",
) -> Path:
    """把 Markdown 报告保存到本地文件，并返回保存路径。"""
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    report_root.mkdir(parents=True, exist_ok=True)

    repo = result.repo_info.repo_model
    report_file = report_root / f"{repo.owner}__{repo.name}.md"
    report_file.write_text(generate_markdown_report(result), encoding="utf-8")
    return report_file


def get_report_path(repo_id: str, output_dir: str = "reports") -> Path:
    """根据仓库标识返回对应的 Markdown 报告路径。"""
    owner, repo = _parse_repo_id(repo_id)
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    return report_root / f"{owner}__{repo}.md"


def remove_markdown_report(repo_id: str, output_dir: str = "reports") -> bool:
    """删除指定仓库的 Markdown 报告；成功返回 True，不存在返回 False。"""
    report_path = get_report_path(repo_id, output_dir=output_dir)
    if not report_path.exists():
        return False

    report_path.unlink()
    return True


def _format_datetime(value: datetime | None) -> str:
    """把时间格式化成更适合写入报告的文本。"""
    if value is None:
        return "未知"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _format_size(size_bytes: int | None) -> str:
    """把字节数格式化为易读的大小文本。"""
    if size_bytes is None:
        return "未知"

    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{size_bytes} B"


def _parse_repo_id(repo_id: str) -> tuple[str, str]:
    """把 owner/repo 形式的仓库标识拆分成 owner 和 repo。"""
    normalized = repo_id.strip().strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) != 2:
        raise ValueError("仓库标识格式应为 owner/repo")

    owner, repo = parts
    if owner in {".", ".."} or repo in {".", ".."}:
        raise ValueError("仓库标识不合法")

    return owner, repo


def _join_or_none(items: list[str]) -> str:
    """把列表拼接成更适合报告展示的文本。"""
    return ', '.join(items) if items else '无'
