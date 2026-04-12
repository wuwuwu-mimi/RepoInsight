SECTION_TITLES = ('结论', '依据', '不确定点')


def format_structured_answer(
    conclusions: list[str],
    evidence: list[str],
    uncertainties: list[str],
) -> str:
    """把回答内容统一整理成三段式结构。"""
    normalized_sections = {
        '结论': _normalize_items(conclusions) or ['暂未提炼出直接结论。'],
        '依据': _normalize_items(evidence) or ['暂无可展示依据。'],
        '不确定点': _normalize_items(uncertainties) or ['无明显不确定点。'],
    }

    lines: list[str] = []
    for title in SECTION_TITLES:
        lines.append(f'{title}：')
        lines.extend(f'- {item}' for item in normalized_sections[title])
    return '\n'.join(lines).strip()


def normalize_structured_answer(message: str) -> str:
    """把模型输出收敛为固定的三段式结构。"""
    sections = {
        '结论': [],
        '依据': [],
        '不确定点': [],
    }

    current_section: str | None = None
    for raw_line in message.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matched_section, remainder = _match_section_header(line)
        if matched_section is not None:
            current_section = matched_section
            if remainder:
                sections[current_section].append(_normalize_bullet_line(remainder))
            continue

        if current_section is None:
            sections['结论'].append(_normalize_bullet_line(line))
            continue

        sections[current_section].append(_normalize_bullet_line(line))

    if not any(sections.values()):
        sections['结论'].append(_normalize_bullet_line(message.strip()))

    return format_structured_answer(
        conclusions=sections['结论'],
        evidence=sections['依据'] or ['未从模型输出中提炼出独立依据，请结合检索证据表查看。'],
        uncertainties=sections['不确定点'] or ['无明显不确定点。'],
    )


def _match_section_header(line: str) -> tuple[str | None, str]:
    """识别当前行是否是预期的标题行。"""
    normalized = (
        line.replace('*', '')
        .replace('#', '')
        .replace('：', ':')
        .strip()
    )
    for title in SECTION_TITLES:
        if normalized == title:
            return title, ''
        if normalized.startswith(f'{title}:'):
            return title, normalized[len(title) + 1:].strip()
    return None, ''


def _normalize_bullet_line(line: str) -> str:
    """去掉常见列表前缀，统一为纯文本。"""
    normalized = line.strip()
    for prefix in ('- ', '* ', '• ', '1. ', '2. ', '3. '):
        if normalized.startswith(prefix):
            return normalized[len(prefix):].strip()
    return normalized


def _normalize_items(items: list[str]) -> list[str]:
    """清理空项并保留稳定顺序。"""
    normalized_items: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_items.append(normalized)
    return normalized_items
