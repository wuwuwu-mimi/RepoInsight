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
    'overview': ('做什么', '是什么', '介绍', '概述', 'overview', 'purpose', '用途', '功能'),
    'startup': ('启动', '运行', '命令', 'run', 'start'),
    'entrypoint': ('入口', 'main', 'entrypoint'),
    'env': ('环境变量', 'env', '配置项'),
    'service': ('数据库', '缓存', 'redis', 'mysql', 'postgres', '服务依赖'),
    'config': ('配置', '构建', '打包', '依赖'),
    'architecture': ('架构', '模块', '依赖关系', '调用链', '子项目', 'monorepo', 'workspace'),
    'api': ('接口', '路由', 'api', 'endpoint', 'http', 'post /', 'get /', 'put /', 'delete /', 'patch /'),
    'implementation': ('实现', '实现逻辑', '怎么实现', '如何实现', '代码', '源码', '函数', '方法', '类', '具体功能'),
}

FOCUS_PREFIXES = {
    'overview': ('项目描述：', '项目类型：', '判断依据：', '摘要：'),
    'startup': ('启动命令：', '启动提示：', '职责：', '摘要：'),
    'entrypoint': ('来源文件：', '入口类型：', '关联组件：', '摘要：'),
    'env': ('环境变量：', '关键结论：', '摘要：'),
    'service': ('外部服务依赖：', '关键结论：', '摘要：'),
    'config': ('配置类型：', '关键结论：', '相关路径：', '摘要：'),
    'architecture': ('子项目根目录：', '所属子项目：', '关键符号：', '模块依赖：', '摘要：'),
    'api': ('路由路径：', 'HTTP 方法：', '处理函数：', '处理限定名：', '代码位置：', '调用：', '摘要：'),
    'implementation': ('符号名称：', '限定名：', '函数签名：', '代码位置：', '调用：', '方法：', '摘要：'),
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
    retrieval_top_k = _resolve_retrieval_top_k(focus, top_k)
    search_result = search_knowledge_base(
        query=question,
        top_k=retrieval_top_k,
        target_dir=target_dir,
        repo_id=repo_id,
    )

    if not search_result.hits:
        return _build_empty_answer_result(
            repo_id=repo_id,
            question=question,
            backend=search_result.backend,
            use_llm=use_llm,
        )

    prioritized_hits = _prioritize_hits_for_focus(search_result.hits, focus)
    selected_lines = _select_supporting_lines(question, prioritized_hits, focus)
    return _build_answer_result_from_context(
        repo_id=repo_id,
        question=question,
        focus=focus,
        backend=search_result.backend,
        prioritized_hits=prioritized_hits,
        selected_lines=selected_lines,
        extra_context_lines=None,
        use_llm=use_llm,
        llm_stream=llm_stream,
        on_llm_chunk=on_llm_chunk,
    )


def _build_empty_answer_result(
    repo_id: str,
    question: str,
    backend: str,
    use_llm: bool,
) -> RepoAnswerResult:
    """构建没有检索命中的降级回答。"""
    return RepoAnswerResult(
        repo_id=repo_id,
        question=question,
        answer=format_structured_answer(
            conclusions=['当前没有找到可用证据。'],
            evidence=['请先执行 analyze 建立该仓库的知识索引。'],
            uncertainties=['当前问题缺少可用知识支撑，或该仓库尚未完成分析。'],
        ),
        answer_mode='extractive',
        backend=backend,
        fallback_used=True,
        llm_enabled=use_llm,
        llm_attempted=False,
        llm_error=None,
        evidence=[],
    )


def _build_answer_result_from_context(
    repo_id: str,
    question: str,
    focus: str,
    backend: str,
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    extra_context_lines: list[str] | None,
    use_llm: bool,
    llm_stream: bool,
    on_llm_chunk: Callable[[str], None] | None,
) -> RepoAnswerResult:
    """基于已完成的路由与检索上下文构建最终回答结果。"""
    evidence = [
        AnswerEvidence(
            repo_id=hit.document.repo_id,
            doc_type=hit.document.doc_type,
            source_path=hit.document.source_path,
            snippet=hit.snippet,
        )
        for hit in prioritized_hits[:3]
    ]

    if focus == 'overview':
        extractive_answer = _build_overview_answer(prioritized_hits, selected_lines)
        evidence_lines = _merge_evidence_lines(
            _build_evidence_lines(prioritized_hits[:4]),
            extra_context_lines,
        )
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
    elif focus == 'startup':
        extractive_answer = _build_startup_answer(prioritized_hits)
        evidence_lines = _merge_evidence_lines(
            _build_evidence_lines(prioritized_hits[:4]),
            extra_context_lines,
        )
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
    elif focus == 'api':
        extractive_answer = _build_api_answer(prioritized_hits, selected_lines)
        extractive_answer = _append_extra_context_notes(extractive_answer, extra_context_lines)
        evidence_lines = _merge_evidence_lines(
            _build_evidence_lines(prioritized_hits[:4]),
            extra_context_lines,
        )
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
    elif focus == 'implementation':
        extractive_answer = _build_implementation_answer(prioritized_hits, selected_lines)
        extractive_answer = _append_extra_context_notes(extractive_answer, extra_context_lines)
        evidence_lines = _merge_evidence_lines(
            _build_evidence_lines(prioritized_hits[:4]),
            extra_context_lines,
        )
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
        evidence_lines = _merge_evidence_lines(
            _build_evidence_lines(prioritized_hits[:3]),
            extra_context_lines,
        )
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
        backend=backend,
        fallback_used=fallback_used,
        llm_enabled=use_llm,
        llm_attempted=llm_attempted,
        llm_error=llm_error,
        evidence=evidence,
    )


def _resolve_retrieval_top_k(focus: str, top_k: int) -> int:
    """根据问题焦点决定实际检索数量。"""
    if focus in {'overview', 'startup'}:
        return max(top_k, 6)
    return top_k


def _infer_question_focus(question: str) -> str:
    """根据问题内容识别当前更像哪一类问题。"""
    lowered = question.lower()
    if re.search(r'\b(get|post|put|delete|patch|options|head)\b\s+/[a-z0-9_\-/{}/:]+', lowered):
        return 'api'
    if re.search(r'/[a-z0-9_\-/{}/:]+', lowered) and any(token in lowered for token in ('接口', '路由', 'api', 'endpoint')):
        return 'api'
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

            score = _supporting_line_doc_type_bonus(hit.document.doc_type)
            if any(line.startswith(prefix) for prefix in preferred_prefixes):
                score += 4

            lowered_line = line.lower()
            token_overlap = sum(1 for token in question_tokens if token in lowered_line)
            score += token_overlap
            score += _score_precise_supporting_line(question_tokens, lowered_line)

            if len(line) > 180:
                score -= 1

            if score <= 0 and line.startswith('摘要：'):
                score = 1

            if score <= 0:
                continue

            candidates.append((score, f'[{hit.document.doc_type} | {source}] {line}'))

        if hit.snippet:
            snippet_score = _supporting_line_doc_type_bonus(hit.document.doc_type) + 1
            lowered_snippet = hit.snippet.lower()
            snippet_score += sum(1 for token in question_tokens if token in lowered_snippet)
            snippet_score += _score_precise_supporting_line(question_tokens, lowered_snippet)
            if snippet_score > 0:
                candidates.append((snippet_score, f'[{hit.document.doc_type} | {source}] {hit.snippet.strip()}'))

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


def _supporting_line_doc_type_bonus(doc_type: str) -> int:
    """为更适合直接回答的文档类型提供基础分。"""
    if doc_type in {'function_summary', 'api_route_summary'}:
        return 3
    if doc_type in {'class_summary', 'entrypoint_summary', 'config_summary'}:
        return 2
    if doc_type in {'readme_summary', 'repo_summary', 'subproject_summary'}:
        return 1
    return 0


def _score_precise_supporting_line(question_tokens: list[str], lowered_line: str) -> int:
    """为路径、符号、路由等显式命中提供额外分数。"""
    score = 0
    for token in question_tokens:
        if len(token) < 3:
            continue
        if not any(char in token for char in ('/', '.', '_')):
            continue
        if token in lowered_line:
            score += 2
    return score


def _prioritize_hits_for_focus(hits: list[SearchHit], focus: str) -> list[SearchHit]:
    """在回答阶段再做一次轻量重排，强化专项问答的命中质量。"""
    doc_type_priority = {
        'overview': {
            'readme_summary': 0,
            'repo_summary': 1,
            'repo_fact': 2,
            'entrypoint_summary': 3,
            'key_file_summary': 4,
            'function_summary': 5,
            'class_summary': 6,
        },
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
        'api': {
            'api_route_summary': 0,
            'function_summary': 1,
            'key_file_summary': 2,
            'entrypoint_summary': 3,
        },
        'implementation': {
            'function_summary': 0,
            'class_summary': 1,
            'key_file_summary': 2,
            'entrypoint_summary': 3,
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
        'overview': '当前问题更接近项目概览，下面优先给出仓库用途、定位和核心能力。',
        'startup': '可优先从这些启动线索开始理解和运行项目。',
        'entrypoint': '当前问题更接近入口定位，下面是最直接的入口线索。',
        'env': '当前问题更接近环境变量定位，下面是最直接的线索。',
        'service': '当前问题更接近外部服务依赖定位，下面是最直接的线索。',
        'config': '当前问题更接近配置定位，下面是最直接的线索。',
        'architecture': '当前问题更接近架构或模块关系定位，下面是最直接的线索。',
        'implementation': '当前问题更接近具体实现定位，下面是最直接的函数、类和调用线索。',
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


def _build_overview_answer(hits: list[SearchHit], selected_lines: list[str]) -> str:
    """优先基于 README 与仓库级摘要回答“这个项目是做什么的”。"""
    repo_descriptions = _clean_sentence_items(_collect_prefixed_lines(hits, '项目描述：', limit=2))
    project_types = _clean_sentence_items(_collect_prefixed_lines(hits, '项目类型：', limit=2))
    project_evidence = _collect_prefixed_lines(hits, '判断依据：', limit=2)
    frameworks = _clean_sentence_items(_collect_prefixed_lines(hits, '框架：', limit=2))
    runtimes = _clean_sentence_items(_collect_prefixed_lines(hits, '运行时：', limit=2))
    readme_points = _clean_sentence_items(_extract_readme_highlights(hits, limit=3))

    conclusions: list[str] = []
    if readme_points:
        conclusions.append(f'从 README 来看，这个项目主要是：{_join_items(readme_points[:2])}。')
    elif repo_descriptions:
        conclusions.append(f'从仓库描述来看，这个项目主要是：{_join_items(repo_descriptions[:2])}。')
    else:
        conclusions.append('当前已命中仓库级文档，但还没有抽取出足够明确的一句话项目介绍。')

    if project_types:
        conclusions.append(f'项目类型可归纳为：{_join_items(project_types[:2])}。')
    if frameworks or runtimes:
        stack_parts: list[str] = []
        if frameworks:
            stack_parts.append(f'框架包括 {_join_items(frameworks[:2])}')
        if runtimes:
            stack_parts.append(f'运行时包括 {_join_items(runtimes[:2])}')
        conclusions.append(f'从技术实现看，{ "，".join(stack_parts) }。')

    if not readme_points and selected_lines:
        conclusions.extend(_strip_source_prefix(line) for line in selected_lines[:2])

    evidence: list[str] = []
    for prefix in ('项目描述：', '项目类型：', '判断依据：', '框架：', '运行时：'):
        evidence.extend(_collect_prefixed_lines(hits, prefix, limit=2, keep_prefix=True))
    evidence.extend(_build_readme_evidence_lines(hits, limit=3))
    evidence.extend(selected_lines[:2])

    uncertainties = ['当前结论基于 README、仓库摘要和静态分析结果，尚未实际运行仓库验证。']
    if not readme_points:
        uncertainties.append('README 中未抽取到足够直接的项目介绍，当前概述更多依赖仓库级结构化摘要。')
    if not project_evidence:
        uncertainties.append('项目类型判断依据仍偏静态，后续可结合运行结果进一步修正。')

    return format_structured_answer(
        conclusions=conclusions[:4],
        evidence=evidence[:8],
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


def _extract_readme_highlights(hits: list[SearchHit], limit: int = 3) -> list[str]:
    """从 README 文档中提取更适合直接回答项目概览的问题句子。"""
    highlights: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.document.doc_type != 'readme_summary':
            continue
        for raw_line in hit.document.content.splitlines():
            if raw_line.strip().startswith('#'):
                continue
            line = raw_line.strip().lstrip('#').strip()
            if not line:
                continue
            if line.startswith('仓库 ') and 'README 摘要候选内容' in line:
                continue
            if line.startswith('以下内容来自 README 原文截断'):
                continue
            if line.startswith('```'):
                continue
            if len(line) < 8:
                continue
            normalized = line.rstrip('。；;')
            if normalized in seen:
                continue
            seen.add(normalized)
            highlights.append(normalized)
            if len(highlights) >= limit:
                return highlights
    return highlights


def _build_readme_evidence_lines(hits: list[SearchHit], limit: int = 3) -> list[str]:
    """把 README 高亮句转换成更适合证据展示的行文本。"""
    evidence: list[str] = []
    for item in _extract_readme_highlights(hits, limit=limit):
        evidence.append(f'[readme_summary | README] {item}')
    return evidence


def _clean_sentence_items(items: list[str]) -> list[str]:
    """清理回答句子里不适合直接展示的占位值和尾部标点。"""
    cleaned: list[str] = []
    for item in items:
        normalized = item.strip().strip('。；;，, ')
        if not normalized or normalized == '无':
            continue
        cleaned.append(normalized)
    return cleaned


def _join_items(items: list[str]) -> str:
    """把多个条目拼接成适合回答展示的短句。"""
    return '；'.join(items)


def _build_api_answer(hits: list[SearchHit], selected_lines: list[str]) -> str:
    """按接口视角组织路由、处理函数和调用线索。"""
    route_paths = _collect_prefixed_lines(hits, '路由路径：', limit=3)
    http_methods = _collect_prefixed_lines(hits, 'HTTP 方法：', limit=3)
    handler_names = _collect_prefixed_lines(hits, '处理函数：', limit=3)
    handler_qualified_names = _collect_prefixed_lines(hits, '处理限定名：', limit=3)
    locations = _collect_prefixed_lines(hits, '代码位置：', limit=3)
    called_symbols = _collect_prefixed_lines(hits, '调用：', limit=3)
    summaries = _collect_prefixed_lines(hits, '摘要：', limit=3)

    conclusions: list[str] = []
    if route_paths and http_methods:
        conclusions.append(f'当前最相关的接口包括：{_join_items([f"{method} {path}" for method, path in zip(http_methods, route_paths, strict=False)])}。')
    elif route_paths:
        conclusions.append(f'当前最相关的路由路径包括：{_join_items(route_paths)}。')
    elif summaries:
        conclusions.append(f'当前命中的接口线索是：{_join_items(summaries[:2])}。')
    else:
        conclusions.append('当前已命中接口相关文档，但还没有抽取到足够明确的路由摘要。')

    if handler_qualified_names:
        conclusions.append(f'优先关注这些处理函数：{_join_items(handler_qualified_names[:3])}。')
    elif handler_names:
        conclusions.append(f'优先关注这些处理函数：{_join_items(handler_names[:3])}。')
    if called_symbols:
        conclusions.append(f'处理流程进一步调用了：{_join_items(called_symbols[:3])}。')
    if locations:
        conclusions.append(f'可以先回到这些代码位置核对：{_join_items(locations[:3])}。')

    evidence = []
    for prefix in ('路由路径：', 'HTTP 方法：', '处理函数：', '处理限定名：', '代码位置：', '调用：', '摘要：'):
        evidence.extend(_collect_prefixed_lines(hits, prefix, limit=2, keep_prefix=True))
    evidence.extend(selected_lines[:2])

    uncertainties = ['当前结论基于静态代码摘要与检索证据，尚未结合运行时流量或请求样例验证。']
    if not called_symbols:
        uncertainties.append('当前尚未提炼出完整的下游调用链，复杂接口仍建议回到源文件继续展开查看。')

    return format_structured_answer(
        conclusions=conclusions[:5],
        evidence=evidence[:8],
        uncertainties=uncertainties,
    )


def _build_implementation_answer(hits: list[SearchHit], selected_lines: list[str]) -> str:
    """按实现视角组织函数、类和调用线索。"""
    symbol_names = _collect_prefixed_lines(hits, '符号名称：', limit=3)
    qualified_names = _collect_prefixed_lines(hits, '限定名：', limit=3)
    signatures = _collect_prefixed_lines(hits, '函数签名：', limit=3)
    locations = _collect_prefixed_lines(hits, '代码位置：', limit=3)
    called_symbols = _collect_prefixed_lines(hits, '调用：', limit=3)
    class_methods = _collect_prefixed_lines(hits, '方法：', limit=2)
    summaries = _collect_prefixed_lines(hits, '摘要：', limit=3)

    conclusions: list[str] = []
    if summaries:
        conclusions.append(f'当前最直接的实现线索是：{_join_items(summaries[:2])}。')
    elif signatures:
        conclusions.append(f'当前最相关的实现入口包括：{_join_items(signatures[:2])}。')
    else:
        conclusions.append('当前已命中与实现相关的文档，但还没有足够明确的职责摘要。')

    if qualified_names:
        conclusions.append(f'优先关注这些符号：{_join_items(qualified_names[:3])}。')
    elif symbol_names:
        conclusions.append(f'优先关注这些符号：{_join_items(symbol_names[:3])}。')

    if called_symbols:
        conclusions.append(f'它们进一步涉及的调用包括：{_join_items(called_symbols[:3])}。')
    if class_methods:
        conclusions.append(f'相关类的方法线索包括：{_join_items(class_methods[:2])}。')
    if locations:
        conclusions.append(f'可以先回到这些代码位置核对：{_join_items(locations[:3])}。')

    evidence = []
    for prefix in ('符号名称：', '限定名：', '函数签名：', '代码位置：', '调用：', '方法：', '摘要：'):
        evidence.extend(_collect_prefixed_lines(hits, prefix, limit=2, keep_prefix=True))
    evidence.extend(selected_lines[:2])

    uncertainties = ['当前结论基于静态代码摘要与检索证据，尚未结合运行时调用链做动态验证。']
    if not called_symbols:
        uncertainties.append('当前尚未提炼出足够完整的调用链，复杂功能仍建议回到源文件继续展开查看。')

    return format_structured_answer(
        conclusions=conclusions[:5],
        evidence=evidence[:8],
        uncertainties=uncertainties,
    )


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


def _merge_evidence_lines(
    evidence_lines: list[str],
    extra_context_lines: list[str] | None,
) -> list[str]:
    """把检索证据与额外上下文合并，避免重复。"""
    if not extra_context_lines:
        return evidence_lines

    merged = list(evidence_lines)
    for line in extra_context_lines:
        if line and line not in merged:
            merged.append(line)
    return merged


def _append_extra_context_notes(answer_text: str, extra_context_lines: list[str] | None) -> str:
    """把 code_agent 等额外线索附加到抽取式回答尾部。"""
    if not extra_context_lines:
        return answer_text

    note_lines = [line for line in extra_context_lines if line.strip()]
    if not note_lines:
        return answer_text

    formatted_lines = [line.removeprefix('[code_agent] ').strip() for line in note_lines]
    relation_chains = _extract_relation_chain_lines(formatted_lines)
    supplemental_lines = [line for line in formatted_lines if line not in relation_chains][:3]

    sections = [answer_text]
    if relation_chains:
        sections.append(
            '实现链路：\n'
            + '\n'.join(f'- {line.removeprefix("代表性关系链：").strip()}' for line in relation_chains[:2])
        )
    if supplemental_lines:
        sections.append(
            '补充线索：\n'
            + '\n'.join(f'- {line}' for line in supplemental_lines)
        )
    return '\n'.join(sections)


def _extract_relation_chain_lines(lines: list[str]) -> list[str]:
    """从额外上下文里提取关系链说明，避免只把它们塞进补充线索。"""
    relation_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue
        if normalized.startswith('代表性关系链：'):
            if normalized not in seen:
                seen.add(normalized)
                relation_lines.append(normalized)
            continue
        if '->' in normalized and ('关系链' in normalized or normalized.count('->') >= 2):
            if normalized not in seen:
                seen.add(normalized)
                relation_lines.append(normalized)
    return relation_lines


def _strip_source_prefix(line: str) -> str:
    """去掉内部拼接的来源前缀，保留更适合结论展示的正文。"""
    if '] ' in line:
        return line.split('] ', maxsplit=1)[1].strip()
    return line.strip()


def _tokenize(text: str) -> list[str]:
    """对中英文问题做轻量切分，便于提取相关回答句子。"""
    lowered = text.lower()
    latin_tokens = re.findall(r'[a-z0-9_\-\.]+', lowered)
    route_tokens = re.findall(r'/[a-z0-9_\-/{}/:]+', lowered)
    chinese_sequences = re.findall(r'[一-鿿]{2,}', lowered)

    tokens = list(latin_tokens) + list(route_tokens)
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
