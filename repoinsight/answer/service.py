from __future__ import annotations

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

CHUNK_DOC_TYPES = {
    'function_body_chunk',
    'class_body_chunk',
    'route_handler_chunk',
    'config_chunk',
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
    relation_chain_details: list[CodeRelationChain] | None = None,
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
        extractive_answer = _append_extra_context_notes(
            extractive_answer,
            extra_context_lines,
            focus=focus,
            relation_chain_details=relation_chain_details,
        )
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
        extractive_answer = _append_extra_context_notes(
            extractive_answer,
            extra_context_lines,
            focus=focus,
            relation_chain_details=relation_chain_details,
        )
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
    elif focus == 'architecture':
        extractive_answer = _build_architecture_answer(prioritized_hits, selected_lines)
        extractive_answer = _append_extra_context_notes(
            extractive_answer,
            extra_context_lines,
            focus=focus,
            relation_chain_details=relation_chain_details,
        )
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
    if focus == 'architecture':
        return max(top_k, 8)
    return top_k


def _infer_question_focus(question: str) -> str:
    """根据问题内容识别当前更像哪一类问题。"""
    lowered = question.lower()
    if re.search(r'\b(get|post|put|delete|patch|options|head)\b\s+/[a-z0-9_\-/{}/:]+', lowered):
        return 'api'
    if re.search(r'/[a-z0-9_\-/{}/:]+', lowered) and any(token in lowered for token in ('接口', '路由', 'api', 'endpoint')):
        return 'api'
    focus_match_order = (
        'implementation',
        'architecture',
        'startup',
        'entrypoint',
        'env',
        'service',
        'config',
        'overview',
    )
    for focus in focus_match_order:
        keywords = QUESTION_FOCUS_KEYWORDS.get(focus, ())
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
        if hit.document.doc_type in CHUNK_DOC_TYPES:
            for chunk_score, chunk_line in _collect_chunk_supporting_candidates(
                question_tokens=question_tokens,
                hit=hit,
            ):
                candidates.append((chunk_score, f'[{hit.document.doc_type} | {source}] {chunk_line}'))
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
    if doc_type in {'function_body_chunk', 'route_handler_chunk'}:
        return 4
    if doc_type in {'class_body_chunk', 'config_chunk'}:
        return 3
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
            'config_chunk': 0,
            'config_summary': 0,
            'entrypoint_summary': 1,
            'key_file_summary': 2,
        },
        'api': {
            'route_handler_chunk': 0,
            'api_route_summary': 1,
            'function_body_chunk': 2,
            'function_summary': 3,
            'key_file_summary': 4,
            'entrypoint_summary': 5,
        },
        'implementation': {
            'function_body_chunk': 0,
            'function_summary': 1,
            'class_body_chunk': 2,
            'class_summary': 3,
            'key_file_summary': 4,
            'entrypoint_summary': 5,
        },
        'architecture': {
            'subproject_summary': 0,
            'repo_summary': 1,
            'key_file_summary': 2,
            'class_summary': 3,
            'function_summary': 4,
            'api_route_summary': 5,
            'entrypoint_summary': 6,
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


def _collect_chunk_supporting_candidates(
    *,
    question_tokens: list[str],
    hit: SearchHit,
) -> list[tuple[int, str]]:
    """从 chunk 文档中挑出更适合直接展示的源码行。"""
    candidates: list[tuple[int, str]] = []
    for line in _extract_code_block_lines(hit.document.content):
        lowered_line = line.lower()
        score = _supporting_line_doc_type_bonus(hit.document.doc_type) + 2
        score += sum(1 for token in question_tokens if token in lowered_line)
        score += _score_precise_supporting_line(question_tokens, lowered_line)
        if any(marker in lowered_line for marker in ('def ', 'class ', 'return ', '@', 'function ', 'async ')):
            score += 2
        candidates.append((score, line))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:3]


def _extract_code_block_lines(content: str, *, max_lines: int = 4) -> list[str]:
    """从 ```text``` 代码块中提取几行源码预览。"""
    inside_block = False
    lines: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith('```'):
            if inside_block:
                break
            inside_block = True
            continue
        if not inside_block or not stripped:
            continue
        lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return lines


def _collect_chunk_locations(
    hits: list[SearchHit],
    allowed_doc_types: set[str],
    *,
    limit: int,
) -> list[str]:
    """收集 chunk 文档的源码位置，便于回答直接指路。"""
    chunk_hits = [hit for hit in hits if hit.document.doc_type in allowed_doc_types]
    return _collect_prefixed_lines(chunk_hits, '代码位置：', limit=limit)


def _collect_chunk_preview_lines(
    hits: list[SearchHit],
    allowed_doc_types: set[str],
    *,
    limit: int,
) -> list[str]:
    """收集 chunk 文档中的关键源码行。"""
    previews: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.document.doc_type not in allowed_doc_types:
            continue
        for line in _extract_code_block_lines(hit.document.content, max_lines=3):
            if line in seen:
                continue
            seen.add(line)
            previews.append(line)
            if len(previews) >= limit:
                return previews
    return previews


def _build_chunk_evidence_lines(
    hits: list[SearchHit],
    allowed_doc_types: set[str],
    *,
    limit: int,
) -> list[str]:
    """把 chunk 文档整理成 evidence 区可直接展示的证据行。"""
    evidence: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.document.doc_type not in allowed_doc_types:
            continue
        source = hit.document.source_path or '仓库级文档'
        location = _first_prefixed_line_value(hit.document.content, '代码位置：') or '未知位置'
        preview = ' | '.join(_extract_code_block_lines(hit.document.content, max_lines=2))
        line = f'[{hit.document.doc_type} | {source}] 代码位置：{location}'
        if preview:
            line += f'；源码片段：{preview}'
        if line in seen:
            continue
        seen.add(line)
        evidence.append(line)
        if len(evidence) >= limit:
            break
    return evidence


def _first_prefixed_line_value(content: str, prefix: str) -> str | None:
    """读取正文里第一条指定前缀的值。"""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            return value or None
    return None


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
    chunk_locations = _collect_chunk_locations(hits, {'route_handler_chunk', 'function_body_chunk'}, limit=3)
    chunk_previews = _collect_chunk_preview_lines(hits, {'route_handler_chunk', 'function_body_chunk'}, limit=3)

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
    if chunk_locations:
        conclusions.append(f'已命中更贴近实现的源码片段，可以先看：{_join_items(chunk_locations[:2])}。')
    if chunk_previews:
        conclusions.append(f'从源码片段可直接看到：{_join_items(chunk_previews[:2])}。')
    if locations:
        conclusions.append(f'可以先回到这些代码位置核对：{_join_items(locations[:3])}。')

    evidence = []
    evidence.extend(_build_chunk_evidence_lines(hits, {'route_handler_chunk', 'function_body_chunk'}, limit=4))
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
    chunk_locations = _collect_chunk_locations(hits, {'function_body_chunk', 'class_body_chunk'}, limit=3)
    chunk_previews = _collect_chunk_preview_lines(hits, {'function_body_chunk', 'class_body_chunk'}, limit=3)

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
    if chunk_locations:
        conclusions.append(f'已命中更贴近实现的源码片段，可以先看：{_join_items(chunk_locations[:2])}。')
    if chunk_previews:
        conclusions.append(f'从源码片段可直接看到：{_join_items(chunk_previews[:2])}。')
    if locations:
        conclusions.append(f'可以先回到这些代码位置核对：{_join_items(locations[:3])}。')

    evidence = []
    evidence.extend(_build_chunk_evidence_lines(hits, {'function_body_chunk', 'class_body_chunk'}, limit=4))
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


def _build_architecture_answer(hits: list[SearchHit], selected_lines: list[str]) -> str:
    """按架构视角组织子项目、模块依赖和关键实现符号。"""
    subproject_roots = _clean_sentence_items(_collect_prefixed_lines(hits, '子项目根目录：', limit=4))
    subproject_types = _clean_sentence_items(_collect_prefixed_lines(hits, '子项目类型：', limit=3))
    subproject_languages = _clean_sentence_items(_collect_prefixed_lines(hits, '语言范围：', limit=3))
    subproject_configs = _clean_sentence_items(_collect_prefixed_lines(hits, '配置文件：', limit=3))
    subproject_entrypoints = _clean_sentence_items(_collect_prefixed_lines(hits, '入口文件：', limit=3))
    owned_subprojects = _clean_sentence_items(_collect_prefixed_lines(hits, '所属子项目：', limit=4))
    code_symbols = _clean_sentence_items(_collect_prefixed_lines(hits, '关键符号：', limit=4))
    module_relations = _clean_sentence_items(_collect_prefixed_lines(hits, '模块依赖：', limit=4))
    repo_subprojects = _clean_sentence_items(_collect_prefixed_lines(hits, '子项目：', limit=2))
    repo_relation_counts = _clean_sentence_items(_collect_prefixed_lines(hits, '模块依赖关系数量：', limit=1))
    summaries = _clean_sentence_items(_collect_prefixed_lines(hits, '摘要：', limit=3))

    conclusions: list[str] = []
    if subproject_roots:
        conclusions.append(f'当前最相关的子项目或模块入口包括：{_join_items(subproject_roots[:3])}。')
    elif owned_subprojects:
        conclusions.append(f'当前命中的实现线索主要落在这些子项目：{_join_items(owned_subprojects[:3])}。')
    elif repo_subprojects:
        conclusions.append(f'仓库当前识别出的子项目概览包括：{_join_items(repo_subprojects[:2])}。')
    elif summaries:
        conclusions.append(f'当前最直接的架构线索是：{_join_items(summaries[:2])}。')
    else:
        conclusions.append('当前已命中架构相关文档，但还没有抽取到足够明确的子项目或模块摘要。')

    if module_relations:
        conclusions.append(f'当前最值得优先核对的模块依赖包括：{_join_items(module_relations[:3])}。')
    elif repo_relation_counts:
        conclusions.append(f'仓库级分析里已抽取到 {repo_relation_counts[0]} 条模块依赖关系，可继续沿命中文档展开。')

    if code_symbols:
        conclusions.append(f'关键实现符号主要集中在：{_join_items(code_symbols[:3])}。')

    stack_parts: list[str] = []
    if subproject_types:
        stack_parts.append(f'子项目类型包括 {_join_items(subproject_types[:2])}')
    if subproject_languages:
        stack_parts.append(f'语言范围包括 {_join_items(subproject_languages[:2])}')
    if stack_parts:
        conclusions.append(f'从模块形态看，{"；".join(stack_parts)}。')

    path_parts: list[str] = []
    if subproject_entrypoints:
        path_parts.append(f'入口文件有 {_join_items(subproject_entrypoints[:2])}')
    if subproject_configs:
        path_parts.append(f'配置文件有 {_join_items(subproject_configs[:2])}')
    if path_parts:
        conclusions.append(f'继续梳理时可以优先关注这些路径：{"；".join(path_parts)}。')

    if len(conclusions) < 2 and selected_lines:
        conclusions.extend(_strip_source_prefix(line) for line in selected_lines[:2])

    evidence = []
    for prefix in (
        '子项目根目录：',
        '子项目类型：',
        '语言范围：',
        '配置文件：',
        '入口文件：',
        '所属子项目：',
        '关键符号：',
        '模块依赖：',
        '子项目：',
        '模块依赖关系数量：',
        '摘要：',
    ):
        evidence.extend(_collect_prefixed_lines(hits, prefix, limit=2, keep_prefix=True))
    evidence.extend(selected_lines[:2])

    uncertainties = ['当前结论基于静态结构摘要与检索证据，尚未结合运行时调用或真实部署拓扑验证。']
    if not module_relations:
        uncertainties.append('当前尚未拿到足够完整的模块依赖明细，复杂项目仍建议继续沿实现链路和配置文件展开核对。')
    if not subproject_roots and not owned_subprojects:
        uncertainties.append('当前命中结果对子项目边界的描述仍然有限，后续可以补更多子项目级摘要。')

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


def _append_extra_context_notes(
    answer_text: str,
    extra_context_lines: list[str] | None,
    *,
    focus: str = 'implementation',
    relation_chain_details: list[CodeRelationChain] | None = None,
) -> str:
    """把 code_agent 等额外线索附加到抽取式回答尾部。"""
    if not extra_context_lines:
        return answer_text

    note_lines = [line for line in extra_context_lines if line.strip()]
    if not note_lines:
        return answer_text

    formatted_lines = [line.removeprefix('[code_agent] ').strip() for line in note_lines]
    relation_chains = _build_relation_chain_lines_from_details(relation_chain_details)
    extracted_relation_chains = _extract_relation_chain_lines(formatted_lines)
    if not relation_chains:
        relation_chains = extracted_relation_chains
    trace_step_lines = _extract_trace_step_lines(formatted_lines)
    supplemental_lines = [
        line for line in formatted_lines
        if line not in extracted_relation_chains and line not in trace_step_lines
    ][:3]
    relation_section_title, process_section_title, supplemental_section_title = _resolve_extra_context_section_titles(focus)

    sections = [answer_text]
    if relation_chains:
        sections.append(
            f'{relation_section_title}：\n'
            + '\n'.join(
                f'- {_format_relation_chain_line(line, focus=focus)}'
                for line in relation_chains[:2]
            )
        )
    if trace_step_lines:
        sections.append(
            f'{process_section_title}：\n'
            + '\n'.join(
                f'- {_format_trace_step_line(line, focus=focus)}'
                for line in trace_step_lines[:3]
            )
        )
    if supplemental_lines:
        sections.append(
            f'{supplemental_section_title}：\n'
            + '\n'.join(f'- {line}' for line in supplemental_lines)
        )
    return '\n'.join(sections)


def _build_relation_chain_lines_from_details(
    relation_chain_details: list[CodeRelationChain] | None,
) -> list[str]:
    """优先把结构化关系链转成稳定的展示行，供回答拼接直接复用。"""
    if not relation_chain_details:
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for item in relation_chain_details:
        if item.typed_text:
            line = f'关系链类型：{item.typed_text}'
        elif item.plain_text:
            line = f'代表性关系链：{item.plain_text}'
        else:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _extract_relation_chain_lines(lines: list[str]) -> list[str]:
    """从额外上下文里提取关系链说明，避免只把它们塞进补充线索。"""
    relation_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue
        if normalized.startswith('关系链类型：'):
            if normalized not in seen:
                seen.add(normalized)
                relation_lines.append(normalized)
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


def _extract_trace_step_lines(lines: list[str]) -> list[str]:
    """从额外上下文中提取可转成自然语言过程说明的追踪步骤。"""
    trace_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        if normalized.startswith('[depth=') and ' -> ' in normalized:
            seen.add(normalized)
            trace_lines.append(normalized)
    return trace_lines


def _resolve_extra_context_section_titles(focus: str) -> tuple[str, str, str]:
    """根据问题焦点决定额外上下文的区块标题。"""
    title_map = {
        'api': ('接口链路', '接口过程', '补充接口线索'),
        'implementation': ('实现链路', '实现过程', '补充实现线索'),
        'architecture': ('模块链路', '模块展开', '补充模块线索'),
    }
    return title_map.get(focus, ('实现链路', '实现过程', '补充线索'))


def _format_relation_chain_line(line: str, *, focus: str = 'implementation') -> str:
    """把关系链条目转换成更适合回答正文展示的文本。"""
    chain_text = line.removeprefix('关系链类型：').removeprefix('代表性关系链：').strip()
    typed_chain_steps = _parse_typed_relation_chain(chain_text)
    if typed_chain_steps is not None:
        return _format_typed_relation_chain(chain_text, typed_chain_steps, focus=focus)

    chain_parts = [item.strip() for item in chain_text.split('->') if item.strip()]
    if len(chain_parts) < 2:
        return chain_text

    if focus == 'architecture':
        if len(chain_parts) == 2:
            return (
                f'{chain_text}（模块依赖从 {chain_parts[0]} 指向 {chain_parts[1]}，'
                '可以先沿这条边界继续查看职责拆分）'
            )

        middle_text = f'，中间经过 {"、".join(chain_parts[1:-1])}' if len(chain_parts) > 2 else ''
        return (
            f'{chain_text}（模块链路从 {chain_parts[0]} 出发{middle_text}，'
            f'最终落到 {chain_parts[-1]}，可据此继续核对职责与依赖方向）'
        )

    if focus == 'api' and '/' in chain_parts[0]:
        tail_text = '，随后依次调用 ' + '、'.join(chain_parts[2:]) if len(chain_parts) > 2 else ''
        return f'{chain_text}（请求先进入 {chain_parts[0]}，再由 {chain_parts[1]} 处理{tail_text}）'

    tail_text = '，随后依次调用 ' + '、'.join(chain_parts[1:]) if len(chain_parts) > 1 else ''
    if focus == 'api':
        return f'{chain_text}（接口主处理链从 {chain_parts[0]} 开始{tail_text}）'
    return f'{chain_text}（主实现流程从 {chain_parts[0]} 开始{tail_text}）'


def _parse_typed_relation_chain(chain_text: str) -> list[tuple[str, str, str | None]] | None:
    """把带关系类型的链路文本解析成边列表。"""
    if '-[' not in chain_text:
        return None

    marker_pattern = re.compile(r'\s*-\[(?P<relation>[a-z_]+)\]->\s*')
    start_match = marker_pattern.search(chain_text)
    if start_match is None:
        return None

    current_label = chain_text[:start_match.start()].strip()
    if not current_label:
        return None
    parsed_steps: list[tuple[str, str, str | None]] = []
    current_position = start_match.start()
    while True:
        marker_match = marker_pattern.match(chain_text, current_position)
        if marker_match is None:
            break
        relation_type = marker_match.group('relation').strip()
        next_marker = marker_pattern.search(chain_text, marker_match.end())
        target_label = chain_text[marker_match.end(): next_marker.start() if next_marker else len(chain_text)].strip()
        if not target_label:
            return None
        parsed_steps.append((current_label, target_label, relation_type.strip()))
        current_label = target_label
        if next_marker is None:
            break
        current_position = next_marker.start()
    return parsed_steps


def _format_typed_relation_chain(
    chain_text: str,
    typed_steps: list[tuple[str, str, str | None]],
    *,
    focus: str,
) -> str:
    """把带关系类型的链路文本转成更自然的说明。"""
    if not typed_steps:
        return chain_text

    explanation_parts = [
        _build_relation_transition_text(source, target, relation_type or '', focus=focus)
        for source, target, relation_type in typed_steps
    ]
    return f'{chain_text}（{"；".join(explanation_parts)}）'


def _format_trace_step_line(line: str, *, focus: str = 'implementation') -> str:
    """把追踪步骤转成更自然的过程说明。"""
    trace_step = _parse_trace_step_line(line)
    if trace_step is None:
        return line

    label = trace_step['label']
    parent_label = trace_step['parent_label']
    location = trace_step['location']
    relation_type = trace_step['relation_type']
    summary = _clean_trace_step_summary(trace_step['summary'], label=label, parent_label=parent_label)
    location_text = f'（{location}）' if location else ''

    if focus == 'api':
        if parent_label:
            return f'{_build_relation_transition_text(parent_label, label, relation_type, focus=focus)}{location_text}，{summary}。'
        return f'接口入口先落到 {label}{location_text}，{summary}。'

    if focus == 'architecture':
        if parent_label:
            return f'{_build_relation_transition_text(parent_label, label, relation_type, focus=focus)}{location_text}，{summary}。'
        return f'当前模块链路先落在 {label}{location_text}，{summary}。'

    if parent_label:
        return f'{_build_relation_transition_text(parent_label, label, relation_type, focus=focus)}{location_text}，{summary}。'
    return f'先定位到 {label}{location_text}，{summary}。'


def _parse_trace_step_line(line: str) -> dict[str, str] | None:
    """解析 code_agent 输出的追踪步骤行。"""
    normalized = line.strip()
    depth_match = re.match(r'^\[depth=(?P<depth>\d+)\]\s*', normalized)
    if depth_match is None:
        return None

    body = normalized[depth_match.end():]
    relation_type = ''
    relation_match = re.match(r'^\[relation=(?P<relation>[a-z_]+)\]\s*', body)
    if relation_match is not None:
        relation_type = relation_match.group('relation')
        body = body[relation_match.end():]
    if ' -> ' not in body:
        return None

    head, summary = body.split(' -> ', maxsplit=1)
    location = ''
    if ' @ ' in head:
        head, location = head.split(' @ ', maxsplit=1)
        location = location.strip()

    label = head.strip()
    parent_label = ''
    if ' <- ' in head:
        label, parent_label = head.split(' <- ', maxsplit=1)
        label = label.strip()
        parent_label = parent_label.strip()

    return {
        'depth': depth_match.group('depth'),
        'label': label,
        'parent_label': parent_label,
        'location': location,
        'relation_type': relation_type,
        'summary': summary.strip(),
    }


def _clean_trace_step_summary(summary: str, *, label: str, parent_label: str) -> str:
    """尽量去掉追踪步骤摘要中的重复前缀，让正文更自然。"""
    cleaned = summary.strip().rstrip('。')
    if parent_label:
        cleaned = cleaned.removeprefix(f'作为 {parent_label} 的下一跳，')
    cleaned = cleaned.removeprefix(f'函数 {label} ')
    cleaned = cleaned.removeprefix(f'类 {label} ')
    cleaned = cleaned.removeprefix(f'接口 {label} 的')
    return cleaned or summary.strip().rstrip('。')


def _build_relation_transition_text(
    parent_label: str,
    label: str,
    relation_type: str,
    *,
    focus: str,
) -> str:
    """根据关系类型生成更稳定的过程说明语句。"""
    relation_type = relation_type or 'call'

    if focus == 'api':
        if relation_type == 'handle_route':
            return f'请求从 {parent_label} 路由到 {label}'
        if relation_type == 'delegate_service':
            return f'从 {parent_label} 继续把业务处理交给服务 {label}'
        if relation_type == 'delegate_repository':
            return f'从 {parent_label} 继续把数据访问交给 {label}'
        if relation_type == 'define_method':
            return f'从 {parent_label} 继续落到类方法 {label}'
        return f'从 {parent_label} 继续调用 {label}'

    if focus == 'architecture':
        if relation_type in {'import', 'import_module'}:
            return f'{parent_label} 继续依赖模块 {label}'
        if relation_type == 'delegate_service':
            return f'{parent_label} 继续把业务链路交给服务 {label}'
        if relation_type == 'delegate_repository':
            return f'{parent_label} 继续把持久化链路交给 {label}'
        if relation_type == 'handle_route':
            return f'{parent_label} 继续把入口流量路由到 {label}'
        if relation_type == 'define_method':
            return f'{parent_label} 继续展开到方法 {label}'
        return f'{parent_label} 继续依赖到 {label}'

    if relation_type == 'delegate_service':
        return f'从 {parent_label} 继续把业务处理交给服务 {label}'
    if relation_type == 'delegate_repository':
        return f'从 {parent_label} 继续把数据访问交给 {label}'
    if relation_type == 'define_method':
        return f'从 {parent_label} 继续展开到方法 {label}'
    if relation_type == 'handle_route':
        return f'从 {parent_label} 继续落到接口处理 {label}'
    return f'从 {parent_label} 继续跟到 {label}'


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
