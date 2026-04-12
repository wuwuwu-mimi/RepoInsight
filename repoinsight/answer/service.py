import re
from collections.abc import Callable

from repoinsight.answer.formatter import format_structured_answer
from repoinsight.llm.config import get_llm_settings
from repoinsight.llm.service import LlmInvocationError, generate_answer_with_llm
from repoinsight.models.answer_model import AnswerEvidence, RepoAnswerResult
from repoinsight.models.rag_model import SearchHit
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR


QUESTION_FOCUS_KEYWORDS = {
    'startup': ('启动', '运行', '命令', 'run', 'start'),
    'entrypoint': ('入口', 'main', 'entrypoint'),
    'env': ('环境变量', 'env', '配置项'),
    'service': ('数据库', '缓存', 'redis', 'mysql', 'postgres', '服务依赖'),
    'config': ('配置', '构建', '打包', '依赖'),
    'architecture': ('架构', '模块', '依赖关系', '调用链', '子项目', 'monorepo', 'workspace'),
}

FOCUS_PREFIXES = {
    'startup': ('启动命令：', '启动提示：', '职责：', '摘要：'),
    'entrypoint': ('来源文件：', '入口类型：', '关联组件：', '摘要：'),
    'env': ('环境变量：', '关键结论：', '摘要：'),
    'service': ('外部服务依赖：', '关键结论：', '摘要：'),
    'config': ('配置类型：', '关键结论：', '相关路径：', '摘要：'),
    'architecture': ('子项目根目录：', '所属子项目：', '关键符号：', '模块依赖：', '摘要：'),
    'generic': ('摘要：', '关键结论：', '职责：', '关键符号：', '模块依赖：'),
}


def answer_repo_question(
    repo_id: str,
    question: str,
    top_k: int = 5,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
    use_llm: bool = True,
    llm_stream: bool = False,
    on_llm_chunk: Callable[[str], None] | None = None,
) -> RepoAnswerResult:
    """基于已分析仓库的知识文档，给出一个可解释的 MVP 回答。"""
    focus = _infer_question_focus(question)
    search_result = search_knowledge_base(
        query=question,
        top_k=max(top_k, 6) if focus == 'startup' else top_k,
        target_dir=target_dir,
        repo_id=repo_id,
    )

    if not search_result.hits:
        return RepoAnswerResult(
            repo_id=repo_id,
            question=question,
            answer=format_structured_answer(
                conclusions=['当前没有找到可用证据。'],
                evidence=['请先执行 analyze 建立该仓库的知识索引。'],
                uncertainties=['当前问题缺少可用知识支撑，或该仓库尚未完成分析。'],
            ),
            answer_mode='extractive',
            backend=search_result.backend,
            fallback_used=True,
            llm_enabled=use_llm,
            llm_attempted=False,
            llm_error=None,
            evidence=[],
        )

    prioritized_hits = _prioritize_hits_for_focus(search_result.hits, focus)
    selected_lines = _select_supporting_lines(question, prioritized_hits, focus)
    evidence = [
        AnswerEvidence(
            repo_id=hit.document.repo_id,
            doc_type=hit.document.doc_type,
            source_path=hit.document.source_path,
            snippet=hit.snippet,
        )
        for hit in prioritized_hits[:3]
    ]

    if focus == 'startup':
        extractive_answer = _build_startup_answer(prioritized_hits)
        evidence_lines = _build_evidence_lines(prioritized_hits[:4])
        llm_answer, llm_attempted, llm_error = _try_build_llm_answer(
            repo_id=repo_id,
            question=question,
            draft_answer=extractive_answer,
            evidence_lines=evidence_lines,
            use_llm=use_llm,
            llm_stream=llm_stream,
            on_llm_chunk=on_llm_chunk,
        )
        if llm_answer is not None:
            answer = llm_answer
            answer_mode = 'llm'
            fallback_used = False
        else:
            answer = extractive_answer
            answer_mode = 'extractive'
            fallback_used = llm_attempted
    elif selected_lines:
        extractive_answer = _build_answer_text(focus, selected_lines)
        evidence_lines = _build_evidence_lines(prioritized_hits[:3])
        llm_answer, llm_attempted, llm_error = _try_build_llm_answer(
            repo_id=repo_id,
            question=question,
            draft_answer=extractive_answer,
            evidence_lines=evidence_lines,
            use_llm=use_llm,
            llm_stream=llm_stream,
            on_llm_chunk=on_llm_chunk,
        )
        if llm_answer is not None:
            answer = llm_answer
            answer_mode = 'llm'
            fallback_used = False
        else:
            answer = extractive_answer
            answer_mode = 'extractive'
            fallback_used = llm_attempted
    else:
        answer = _build_fallback_answer(prioritized_hits[:3])
        answer_mode = 'extractive'
        fallback_used = True
        llm_attempted = False
        llm_error = None

    return RepoAnswerResult(
        repo_id=repo_id,
        question=question,
        answer=answer,
        answer_mode=answer_mode,
        backend=search_result.backend,
        fallback_used=fallback_used,
        llm_enabled=use_llm,
        llm_attempted=llm_attempted,
        llm_error=llm_error,
        evidence=evidence,
    )


def _infer_question_focus(question: str) -> str:
    """根据问题内容识别当前更像哪一类问题。"""
    lowered = question.lower()
    for focus, keywords in QUESTION_FOCUS_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return focus
    return 'generic'


def _select_supporting_lines(question: str, hits: list[SearchHit], focus: str, max_lines: int = 4) -> list[str]:
    """从命中文档中挑出最适合直接回答问题的几行文本。"""
    question_tokens = _tokenize(question)
    preferred_prefixes = FOCUS_PREFIXES.get(focus, FOCUS_PREFIXES['generic'])
    candidates: list[tuple[int, str]] = []

    for hit in hits[:4]:
        source = hit.document.source_path or '仓库级文档'
        for raw_line in hit.document.content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            score = 0
            if any(line.startswith(prefix) for prefix in preferred_prefixes):
                score += 4

            lowered_line = line.lower()
            token_overlap = sum(1 for token in question_tokens if token in lowered_line)
            score += token_overlap

            if score <= 0 and line.startswith('摘要：'):
                score = 1

            if score <= 0:
                continue

            candidates.append((score, f'[{hit.document.doc_type} | {source}] {line}'))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    seen: set[str] = set()
    for _, line in candidates:
        if line in seen:
            continue
        seen.add(line)
        selected.append(line)
        if len(selected) >= max_lines:
            break

    return selected


def _prioritize_hits_for_focus(hits: list[SearchHit], focus: str) -> list[SearchHit]:
    """在回答阶段再做一次轻量重排，强化专项问答的命中质量。"""
    doc_type_priority = {
        'startup': {
            'entrypoint_summary': 0,
            'config_summary': 1,
            'readme_summary': 2,
            'key_file_summary': 3,
        },
        'entrypoint': {
            'entrypoint_summary': 0,
            'key_file_summary': 1,
            'config_summary': 2,
        },
        'env': {
            'config_summary': 0,
            'entrypoint_summary': 1,
            'key_file_summary': 2,
        },
    }
    priority_map = doc_type_priority.get(focus, {})
    return sorted(
        hits,
        key=lambda item: (
            priority_map.get(item.document.doc_type, 9),
            -item.score,
            item.document.doc_id,
        ),
    )


def _build_answer_text(focus: str, selected_lines: list[str]) -> str:
    """把命中的关键信息拼成终端友好的回答文本。"""
    intro_map = {
        'startup': '可优先从这些启动线索开始理解和运行项目。',
        'entrypoint': '当前问题更接近入口定位，下面是最直接的入口线索。',
        'env': '当前问题更接近环境变量定位，下面是最直接的线索。',
        'service': '当前问题更接近外部服务依赖定位，下面是最直接的线索。',
        'config': '当前问题更接近配置定位，下面是最直接的线索。',
        'architecture': '当前问题更接近架构或模块关系定位，下面是最直接的线索。',
        'generic': '下面是基于当前知识文档提炼出的直接回答。',
    }
    conclusions = [intro_map.get(focus, intro_map['generic'])]
    conclusions.extend(_strip_source_prefix(line) for line in selected_lines[:2])
    evidence = selected_lines[:3]
    uncertainties = ['当前结论基于静态分析与检索证据，尚未实际运行仓库验证。']
    return format_structured_answer(
        conclusions=conclusions,
        evidence=evidence,
        uncertainties=uncertainties,
    )


def _build_startup_answer(hits: list[SearchHit]) -> str:
    """按固定模板回答启动类问题，避免只返回仓库概览。"""
    startup_commands = _collect_prefixed_lines(hits, '启动命令：', limit=4)
    startup_hints = _collect_prefixed_lines(hits, '启动提示：', limit=3)
    dependent_configs = _collect_prefixed_lines(hits, '依赖配置：', limit=3)
    entry_files = _collect_prefixed_lines(hits, '来源文件：', limit=4)
    responsibilities = _collect_prefixed_lines(hits, '职责：', limit=2)

    conclusions: list[str] = []
    if startup_commands:
        conclusions.append(f'可优先尝试这些启动命令：{_join_items(startup_commands)}。')
    else:
        conclusions.append('暂未抽取到完全确定的启动命令，可先从入口摘要和配置摘要联合确认。')

    if entry_files:
        conclusions.append(f'当前最相关的入口或配置文件：{_join_items(entry_files)}。')
    if dependent_configs:
        conclusions.append(f'启动前建议一起检查这些配置：{_join_items(dependent_configs)}。')
    if startup_hints:
        conclusions.append(f'额外启动提示：{_join_items(startup_hints)}。')
    if responsibilities:
        conclusions.append(f'入口职责线索：{_join_items(responsibilities)}。')

    evidence = []
    for prefix in ('启动命令：', '来源文件：', '依赖配置：', '启动提示：', '职责：'):
        evidence.extend(_collect_prefixed_lines(hits, prefix, limit=2, keep_prefix=True))

    uncertainties = ['当前结论基于静态分析与配置线索，尚未实际运行仓库验证。']
    if not startup_commands:
        uncertainties.append('没有在入口摘要中抽取到明确启动命令，可能需要结合 README 或脚本手动确认。')

    return format_structured_answer(
        conclusions=conclusions[:4],
        evidence=evidence[:6],
        uncertainties=uncertainties,
    )


def _collect_prefixed_lines(
    hits: list[SearchHit],
    prefix: str,
    *,
    limit: int,
    keep_prefix: bool = False,
) -> list[str]:
    """从命中文档中抽取指定前缀的证据行。"""
    collected: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        source = hit.document.source_path or '仓库级文档'
        for raw_line in hit.document.content.splitlines():
            line = raw_line.strip()
            if not line.startswith(prefix):
                continue
            value = line if keep_prefix else line[len(prefix):].strip()
            if keep_prefix:
                value = f'[{hit.document.doc_type} | {source}] {value}'
            if value in seen:
                continue
            seen.add(value)
            collected.append(value)
            if len(collected) >= limit:
                return collected
    return collected


def _join_items(items: list[str]) -> str:
    """把多个条目拼接成适合回答展示的短句。"""
    return '；'.join(items)


def _build_fallback_answer(hits: list[SearchHit]) -> str:
    """当没有抽取到明确回答句子时，退化为摘要式回答。"""
    evidence = []
    for hit in hits:
        source = hit.document.source_path or '仓库级文档'
        evidence.append(f'[{hit.document.doc_type} | {source}] {hit.snippet}')
    return format_structured_answer(
        conclusions=['暂时没有抽取出足够明确的直接结论。'],
        evidence=evidence,
        uncertainties=['当前回答主要基于检索片段，建议继续查看原始文件。'],
    )


def _try_build_llm_answer(
    repo_id: str,
    question: str,
    draft_answer: str,
    evidence_lines: list[str],
    use_llm: bool,
    llm_stream: bool,
    on_llm_chunk: Callable[[str], None] | None,
) -> tuple[str | None, bool, str | None]:
    """若已配置 LLM，则尝试生成更自然的最终回答。"""
    if not use_llm:
        return None, False, None

    if get_llm_settings() is None:
        return None, False, None

    try:
        return (
            generate_answer_with_llm(
                question=question,
                repo_id=repo_id,
                draft_answer=draft_answer,
                evidence_lines=evidence_lines,
                stream=llm_stream,
                on_chunk=on_llm_chunk,
            ),
            True,
            None,
        )
    except LlmInvocationError as exc:
        return None, True, str(exc)


def _build_evidence_lines(hits: list[SearchHit]) -> list[str]:
    """把检索命中转成适合发给 LLM 的证据行。"""
    evidence_lines: list[str] = []
    for hit in hits:
        source = hit.document.source_path or '仓库级文档'
        evidence_lines.append(f'{hit.document.doc_type} | {source} | {hit.snippet}')
    return evidence_lines


def _strip_source_prefix(line: str) -> str:
    """去掉内部拼接的来源前缀，保留更适合结论展示的正文。"""
    if '] ' in line:
        return line.split('] ', maxsplit=1)[1].strip()
    return line.strip()


def _tokenize(text: str) -> list[str]:
    """对中英文问题做轻量切分，便于提取相关回答句子。"""
    lowered = text.lower()
    latin_tokens = re.findall(r'[a-z0-9_\-\.]+', lowered)
    chinese_sequences = re.findall(r'[\u4e00-\u9fff]{2,}', lowered)

    tokens = list(latin_tokens)
    for sequence in chinese_sequences:
        tokens.append(sequence)
        tokens.extend(
            sequence[index:index + 2]
            for index in range(len(sequence) - 1)
        )

    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    return unique_tokens
