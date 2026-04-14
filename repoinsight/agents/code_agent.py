import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from repoinsight.agents.models import CodeInvestigationResult, CodeTraceStep
from repoinsight.ingest.repo_cache import get_clone_path
from repoinsight.models.rag_model import KnowledgeDocument, SearchHit
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR, load_repo_documents


SUPPORTED_CODE_DOC_TYPES = {'function_summary', 'class_summary', 'api_route_summary'}
MAX_SOURCE_SNIPPET_LINES = 8
CODE_AGENT_CACHE_MAX_SIZE = 128
CODE_AGENT_MIN_CONFIDENCE_SCORE = 0.4
_CODE_AGENT_CACHE: OrderedDict[str, CodeInvestigationResult] = OrderedDict()
_CODE_AGENT_CACHE_LOCK = Lock()


@dataclass(slots=True)
class _TraceQueueItem:
    """表示等待展开的一条代码追踪任务。"""

    document: KnowledgeDocument
    depth: int
    parent_label: str | None
    step_kind: str



def investigate_code_hits(
    question: str,
    hits: list[SearchHit],
    focus: str,
    *,
    repo_id: str | None = None,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
    max_hits: int = 4,
    max_follow_steps: int = 6,
    max_follow_depth: int = 2,
) -> CodeInvestigationResult | None:
    """基于已召回的代码级文档，递归提炼跨文件调用链与源码落点。"""
    relevant_hits = [
        hit for hit in hits
        if hit.document.doc_type in SUPPORTED_CODE_DOC_TYPES
    ]
    if not relevant_hits:
        return None

    cache_key = _build_investigation_cache_key(
        question=question,
        focus=focus,
        repo_id=repo_id,
        relevant_hits=relevant_hits,
        target_dir=target_dir,
        max_hits=max_hits,
        max_follow_steps=max_follow_steps,
        max_follow_depth=max_follow_depth,
    )
    cached = _get_cached_investigation(cache_key)
    if cached is not None:
        return cached.model_copy(update={'cache_hit': True})

    question_tokens = _tokenize(question)
    ranked_hits = sorted(
        relevant_hits,
        key=lambda item: (
            -_score_hit_against_question(item, question_tokens),
            -item.score,
            item.document.doc_id,
        ),
    )
    selected_hits = ranked_hits[:max_hits]
    target_repo_id = repo_id or selected_hits[0].document.repo_id
    repo_documents = load_repo_documents(target_repo_id, target_dir=target_dir)
    lookup = _build_document_lookup(repo_documents)

    matched_symbols: list[str] = []
    matched_routes: list[str] = []
    source_paths: list[str] = []
    evidence_locations: list[str] = []
    called_symbols: list[str] = []
    implementation_notes: list[str] = []
    evidence_doc_types: list[str] = []
    trace_steps: list[CodeTraceStep] = []
    seen_doc_ids: set[str] = set()
    queued_doc_ids: set[str] = set()

    queue: deque[_TraceQueueItem] = deque()
    for hit in selected_hits:
        queue.append(
            _TraceQueueItem(
                document=hit.document,
                depth=0,
                parent_label=None,
                step_kind='entry',
            )
        )
        queued_doc_ids.add(hit.document.doc_id)

    followed_step_count = 0
    while queue:
        item = queue.popleft()
        queued_doc_ids.discard(item.document.doc_id)
        if item.document.doc_id in seen_doc_ids:
            continue

        trace_step, step_called_symbols = _build_trace_step_from_document(
            document=item.document,
            repo_id=target_repo_id,
            step_kind=item.step_kind,
            depth=item.depth,
            parent_label=item.parent_label,
        )
        trace_steps.append(trace_step)
        seen_doc_ids.add(item.document.doc_id)
        evidence_doc_types.append(item.document.doc_type)
        called_symbols.extend(step_called_symbols)
        _collect_trace_step_fields(
            trace_step=trace_step,
            document=item.document,
            matched_symbols=matched_symbols,
            matched_routes=matched_routes,
            source_paths=source_paths,
            evidence_locations=evidence_locations,
            implementation_notes=implementation_notes,
        )

        if item.depth >= max_follow_depth:
            continue
        if followed_step_count >= max_follow_steps:
            continue

        next_documents = _resolve_follow_documents(
            called_symbols=step_called_symbols,
            lookup=lookup,
            seen_doc_ids=seen_doc_ids,
            queued_doc_ids=queued_doc_ids,
            current_source_path=item.document.source_path,
            max_follow_steps=max_follow_steps - followed_step_count,
        )
        for next_document in next_documents:
            queue.append(
                _TraceQueueItem(
                    document=next_document,
                    depth=item.depth + 1,
                    parent_label=trace_step.label,
                    step_kind='callee',
                )
            )
            queued_doc_ids.add(next_document.doc_id)
            followed_step_count += 1

    matched_symbols = _unique_keep_order(matched_symbols)
    matched_routes = _unique_keep_order(matched_routes)
    source_paths = _unique_keep_order(source_paths)
    evidence_locations = _unique_keep_order(evidence_locations)
    called_symbols = _unique_keep_order(called_symbols)
    implementation_notes = _unique_keep_order(implementation_notes)
    evidence_doc_types = _unique_keep_order(evidence_doc_types)
    relation_chains = _build_relation_chains(trace_steps)

    summary = _build_investigation_summary(
        focus=focus,
        matched_symbols=matched_symbols,
        matched_routes=matched_routes,
        source_paths=source_paths,
        evidence_locations=evidence_locations,
        called_symbols=called_symbols,
        trace_steps=trace_steps,
        relation_chains=relation_chains,
    )
    relevance_score, confidence_level, quality_notes = _evaluate_investigation_quality(
        question_tokens=question_tokens,
        focus=focus,
        matched_symbols=matched_symbols,
        matched_routes=matched_routes,
        source_paths=source_paths,
        evidence_locations=evidence_locations,
        called_symbols=called_symbols,
        trace_steps=trace_steps,
    )
    result = CodeInvestigationResult(
        focus=focus,
        summary=summary,
        matched_symbols=matched_symbols,
        matched_routes=matched_routes,
        source_paths=source_paths,
        evidence_locations=evidence_locations,
        called_symbols=called_symbols,
        trace_steps=trace_steps,
        relation_chains=relation_chains,
        implementation_notes=implementation_notes,
        evidence_doc_types=evidence_doc_types,
        relevance_score=relevance_score,
        confidence_level=confidence_level,
        quality_notes=quality_notes,
        cache_hit=False,
        recovery_attempted=False,
        recovery_improved=False,
    )
    _store_cached_investigation(cache_key, result)
    return result



def build_code_investigation_context_lines(
    investigation: CodeInvestigationResult,
    *,
    max_lines: int = 6,
) -> list[str]:
    """把代码调查结果转换为 synthesis_agent 可直接消费的补充上下文。"""
    quality_text = (
        f'置信度={investigation.confidence_level}，'
        f'相关性={investigation.relevance_score:.2f}'
    )
    if investigation.cache_hit:
        quality_text += '，命中缓存'
    lines = [f'[code_agent] {investigation.summary}（{quality_text}）']
    remaining_slots = max(0, max_lines - 1)
    if remaining_slots > 0 and investigation.relation_chains:
        lines.append(f'[code_agent] 代表性关系链：{investigation.relation_chains[0]}')
        remaining_slots -= 1
    for step in investigation.trace_steps[:remaining_slots]:
        parent_text = f' <- {step.parent_label}' if step.parent_label else ''
        depth_text = f'[depth={step.depth}] '
        location_text = f' @ {step.location}' if step.location else ''
        lines.append(f'[code_agent] {depth_text}{step.label}{parent_text}{location_text} -> {step.summary}')
    return _unique_keep_order(lines)


def should_use_code_investigation_context(result: CodeInvestigationResult | None) -> bool:
    """判断当前代码调查结果是否足够可靠，可直接注入最终回答上下文。"""
    if result is None:
        return False
    if result.confidence_level == 'low':
        return False
    return result.relevance_score >= CODE_AGENT_MIN_CONFIDENCE_SCORE


def clear_code_agent_cache() -> None:
    """清空 code_agent 的进程内缓存，便于测试或调试。"""
    with _CODE_AGENT_CACHE_LOCK:
        _CODE_AGENT_CACHE.clear()


def _build_investigation_cache_key(
    *,
    question: str,
    focus: str,
    repo_id: str | None,
    relevant_hits: list[SearchHit],
    target_dir: str,
    max_hits: int,
    max_follow_steps: int,
    max_follow_depth: int,
) -> str:
    """构造稳定的缓存键，避免同一问题重复展开代码追踪。"""
    doc_ids = [hit.document.doc_id for hit in relevant_hits[: max_hits * 2]]
    return '|'.join(
        [
            repo_id or '',
            target_dir,
            focus,
            question.strip().lower(),
            str(max_hits),
            str(max_follow_steps),
            str(max_follow_depth),
            ','.join(doc_ids),
        ]
    )


def _get_cached_investigation(cache_key: str) -> CodeInvestigationResult | None:
    """读取 code_agent 缓存，并维护最近使用顺序。"""
    with _CODE_AGENT_CACHE_LOCK:
        cached = _CODE_AGENT_CACHE.get(cache_key)
        if cached is None:
            return None
        _CODE_AGENT_CACHE.move_to_end(cache_key)
        return cached


def _store_cached_investigation(cache_key: str, result: CodeInvestigationResult) -> None:
    """写入 code_agent 缓存，并控制缓存大小。"""
    with _CODE_AGENT_CACHE_LOCK:
        _CODE_AGENT_CACHE[cache_key] = result
        _CODE_AGENT_CACHE.move_to_end(cache_key)
        while len(_CODE_AGENT_CACHE) > CODE_AGENT_CACHE_MAX_SIZE:
            _CODE_AGENT_CACHE.popitem(last=False)


def _evaluate_investigation_quality(
    *,
    question_tokens: list[str],
    focus: str,
    matched_symbols: list[str],
    matched_routes: list[str],
    source_paths: list[str],
    evidence_locations: list[str],
    called_symbols: list[str],
    trace_steps: list[CodeTraceStep],
) -> tuple[float, str, list[str]]:
    """对代码调查结果做轻量质量评估，供编排层决定是否吸收。"""
    score = 0.0
    notes: list[str] = []

    if trace_steps:
        score += min(len(trace_steps), 4) * 0.12
        notes.append(f'已展开 {len(trace_steps)} 个源码追踪步骤')
    if evidence_locations:
        score += min(len(evidence_locations), 3) * 0.08
        notes.append('包含精确源码位置')
    if source_paths:
        score += min(len(source_paths), 3) * 0.06
        notes.append(f'涉及 {len(source_paths)} 个源码文件')
    if called_symbols:
        score += min(len(called_symbols), 4) * 0.05
        notes.append('已提取下游调用线索')
    if any(step.snippet for step in trace_steps):
        score += 0.1
        notes.append('包含源码片段预览')

    overlap_score = _score_question_overlap(
        question_tokens=question_tokens,
        matched_symbols=matched_symbols,
        matched_routes=matched_routes,
        source_paths=source_paths,
        called_symbols=called_symbols,
    )
    if overlap_score > 0:
        score += min(overlap_score * 0.08, 0.28)
        notes.append('问题关键词与命中线索存在重合')

    if focus == 'api' and matched_routes:
        score += 0.12
        notes.append('接口问题已命中路由摘要')
    if focus == 'implementation' and matched_symbols:
        score += 0.12
        notes.append('实现问题已命中函数或类摘要')

    final_score = round(min(score, 1.0), 2)
    if final_score >= 0.75:
        confidence_level = 'high'
    elif final_score >= CODE_AGENT_MIN_CONFIDENCE_SCORE:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'
        if not notes:
            notes.append('缺少足够的符号、位置或调用链证据')
    return final_score, confidence_level, notes


def _score_question_overlap(
    *,
    question_tokens: list[str],
    matched_symbols: list[str],
    matched_routes: list[str],
    source_paths: list[str],
    called_symbols: list[str],
) -> int:
    """计算问题 token 与命中线索之间的重合度。"""
    lowered_text = ' '.join(matched_symbols + matched_routes + source_paths + called_symbols).lower()
    score = 0
    for token in question_tokens:
        if token in lowered_text:
            score += 1
    return score



def _build_trace_step_from_document(
    *,
    document: KnowledgeDocument,
    repo_id: str,
    step_kind: str,
    depth: int,
    parent_label: str | None,
) -> tuple[CodeTraceStep, list[str]]:
    """从单个知识文档构建一条源码追踪步骤。"""
    metadata = document.metadata
    label = _resolve_document_label(document)
    location = _build_location_text(document.source_path, metadata)
    snippet = _load_source_snippet(
        repo_id=repo_id,
        source_path=document.source_path,
        metadata=metadata,
        label=label,
    )
    summary = _build_step_summary(document, parent_label=parent_label)
    called_symbols = _resolve_follow_targets(document)
    return (
        CodeTraceStep(
            step_kind=step_kind,
            label=label,
            source_path=document.source_path,
            location=location,
            summary=summary,
            snippet=snippet,
            depth=depth,
            parent_label=parent_label,
        ),
        called_symbols,
    )



def _build_document_lookup(documents: list[KnowledgeDocument]) -> dict[str, list[KnowledgeDocument]]:
    """按符号名和路由路径建立轻量索引，便于补全调用链。"""
    lookup: dict[str, list[KnowledgeDocument]] = {}
    for document in documents:
        if document.doc_type not in SUPPORTED_CODE_DOC_TYPES:
            continue
        metadata = document.metadata
        candidate_keys = [
            _first_text(metadata.get('qualified_name')),
            _first_text(metadata.get('symbol_name')),
            _first_text(metadata.get('handler_qualified_name')),
            _first_text(metadata.get('handler_name')),
            _first_text(metadata.get('route_path')),
        ]
        candidate_keys.extend(_normalize_list(metadata.get('code_entity_refs')))
        candidate_keys.extend(_normalize_list(metadata.get('code_entity_names')))
        candidate_keys.extend(_normalize_list(metadata.get('class_methods')))
        candidate_keys.extend(_normalize_list(metadata.get('code_relation_targets')))
        for key in candidate_keys:
            for normalized_key in _expand_lookup_keys(key):
                _add_lookup_item(lookup, normalized_key, document)
    return lookup



def _expand_lookup_keys(raw_key: str) -> list[str]:
    """把一个原始符号名扩展成多种可匹配键。"""
    normalized = raw_key.strip().lower()
    if not normalized:
        return []

    keys = [normalized]
    for separator in ('.', ':', '/', '->'):
        if separator in normalized:
            keys.append(normalized.split(separator)[-1])
    if normalized.endswith('()'):
        keys.append(normalized[:-2])
    return _unique_keep_order(keys)



def _resolve_follow_documents(
    *,
    called_symbols: list[str],
    lookup: dict[str, list[KnowledgeDocument]],
    seen_doc_ids: set[str],
    queued_doc_ids: set[str],
    current_source_path: str | None,
    max_follow_steps: int,
) -> list[KnowledgeDocument]:
    """根据调用符号补出下一跳实现文档，支持跨文件继续展开。"""
    resolved: list[KnowledgeDocument] = []
    for called_symbol in called_symbols:
        candidates = _lookup_candidates_for_called_symbol(called_symbol, lookup)
        best = _pick_best_follow_document(
            candidates=candidates,
            seen_doc_ids=seen_doc_ids,
            queued_doc_ids=queued_doc_ids,
            current_source_path=current_source_path,
        )
        if best is None:
            continue
        resolved.append(best)
        queued_doc_ids.add(best.doc_id)
        if len(resolved) >= max_follow_steps:
            break
    return resolved



def _lookup_candidates_for_called_symbol(
    called_symbol: str,
    lookup: dict[str, list[KnowledgeDocument]],
) -> list[KnowledgeDocument]:
    """根据调用符号命中候选实现文档。"""
    merged: list[KnowledgeDocument] = []
    for lookup_key in _expand_lookup_keys(called_symbol):
        merged.extend(lookup.get(lookup_key, []))
    unique_candidates: list[KnowledgeDocument] = []
    seen_doc_ids: set[str] = set()
    for item in merged:
        if item.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(item.doc_id)
        unique_candidates.append(item)
    return unique_candidates



def _pick_best_follow_document(
    *,
    candidates: list[KnowledgeDocument],
    seen_doc_ids: set[str],
    queued_doc_ids: set[str],
    current_source_path: str | None,
) -> KnowledgeDocument | None:
    """从候选文档中选出最适合作为下一跳的一条。"""
    available = [
        item for item in candidates
        if item.doc_id not in seen_doc_ids and item.doc_id not in queued_doc_ids
    ]
    if not available:
        return None

    priority = {'function_summary': 0, 'api_route_summary': 1, 'class_summary': 2}
    return sorted(
        available,
        key=lambda item: (
            0 if item.source_path == current_source_path else 1,
            priority.get(item.doc_type, 9),
            item.doc_id,
        ),
    )[0]



def _collect_trace_step_fields(
    *,
    trace_step: CodeTraceStep,
    document: KnowledgeDocument,
    matched_symbols: list[str],
    matched_routes: list[str],
    source_paths: list[str],
    evidence_locations: list[str],
    implementation_notes: list[str],
) -> None:
    """把追踪步骤中的关键信息回填到聚合结果。"""
    metadata = document.metadata
    symbol_name = (
        _first_text(metadata.get('qualified_name'))
        or _first_text(metadata.get('symbol_name'))
        or _first_text(metadata.get('handler_qualified_name'))
        or _first_text(metadata.get('code_entity_refs'))
    )
    route_path = _first_text(metadata.get('route_path'))

    if symbol_name:
        matched_symbols.append(symbol_name)
    if route_path:
        matched_routes.append(route_path)
    if trace_step.source_path:
        source_paths.append(trace_step.source_path)
    if trace_step.location:
        evidence_locations.append(trace_step.location)

    note = trace_step.summary
    if trace_step.parent_label:
        note = f'{trace_step.parent_label} -> {trace_step.label}：{note}'
    if trace_step.location:
        note = f'{note} 位置：{trace_step.location}。'
    implementation_notes.append(note)
    if trace_step.snippet:
        implementation_notes.append(f'{trace_step.label} 对应源码片段：\n{trace_step.snippet}')



def _resolve_document_label(document: KnowledgeDocument) -> str:
    """提取适合作为追踪标签的名称。"""
    metadata = document.metadata
    if document.doc_type == 'api_route_summary':
        route_path = _first_text(metadata.get('route_path')) or '未知路由'
        methods = _normalize_list(metadata.get('http_methods'))
        method_text = '/'.join(methods) if methods else 'HTTP'
        return f'{method_text} {route_path}'

    return (
        _first_text(metadata.get('qualified_name'))
        or _first_text(metadata.get('symbol_name'))
        or _first_text(metadata.get('handler_qualified_name'))
        or _first_text(metadata.get('code_entity_refs'))
        or document.title
    )



def _build_step_summary(document: KnowledgeDocument, *, parent_label: str | None) -> str:
    """把文档摘要转换为更适合追踪展示的短说明。"""
    metadata = document.metadata
    called_symbols = _resolve_follow_targets(document)
    location = _build_location_text(document.source_path, metadata)

    if document.doc_type == 'api_route_summary':
        handler_name = _first_text(metadata.get('handler_qualified_name')) or _first_text(metadata.get('handler_name'))
        route_path = _first_text(metadata.get('route_path')) or '未知路由'
        summary = f'接口 {route_path} 的处理入口是 {handler_name or "未知处理函数"}'
    elif document.doc_type == 'function_summary':
        qualified_name = _first_text(metadata.get('qualified_name')) or _first_text(metadata.get('symbol_name'))
        summary = f'函数 {qualified_name or "未知函数"} 承担当前实现逻辑'
    else:
        qualified_name = _first_text(metadata.get('qualified_name')) or _first_text(metadata.get('symbol_name'))
        summary = f'类 {qualified_name or "未知类"} 提供相关职责上下文'

    if parent_label:
        summary = f'作为 {parent_label} 的下一跳，{summary}'
    if location:
        summary += f'，源码位置在 {location}'
    if called_symbols:
        summary += f'，继续调用 {_join_items(called_symbols[:3])}'
    raw_summary = _extract_prefixed_value(document.content, '摘要：')
    if raw_summary:
        summary += f'。摘要：{raw_summary}'
    return summary



def _build_investigation_summary(
    *,
    focus: str,
    matched_symbols: list[str],
    matched_routes: list[str],
    source_paths: list[str],
    evidence_locations: list[str],
    called_symbols: list[str],
    trace_steps: list[CodeTraceStep],
    relation_chains: list[str],
) -> str:
    """生成一段简短的代码调查摘要。"""
    path_text = _join_items(source_paths[:4]) if source_paths else '未知文件'
    location_text = _join_items(evidence_locations[:4]) if evidence_locations else '暂无精确行号'
    max_depth = max((step.depth for step in trace_steps), default=0)
    if focus == 'api' and matched_routes:
        route_text = _join_items(matched_routes[:2])
        handler_text = _join_items(matched_symbols[:3]) if matched_symbols else '未知处理函数'
        summary = (
            f'已定位到接口 {route_text} 的实现入口，核心处理函数链路包含 {handler_text}，'
            f'源码涉及 {path_text}，关键位置包括 {location_text}。'
        )
    else:
        symbol_text = _join_items(matched_symbols[:4]) if matched_symbols else '未知符号'
        summary = (
            f'已定位到与问题最相关的实现符号链路 {symbol_text}，源码涉及 {path_text}，'
            f'关键位置包括 {location_text}。'
        )

    if called_symbols:
        summary += f' 下游调用重点包括：{_join_items(called_symbols[:4])}。'
    if max_depth > 0:
        summary += f' 当前已展开到第 {max_depth} 层调用链。'
    if relation_chains:
        summary += f' 代表性关系链：{relation_chains[0]}。'
    return summary


def _build_relation_chains(trace_steps: list[CodeTraceStep], *, max_chains: int = 4) -> list[str]:
    """把 trace_steps 归纳成几条可读的调用/关系链。"""
    if not trace_steps:
        return []

    label_to_children: dict[str, list[str]] = {}
    roots: list[str] = []
    for step in trace_steps:
        if step.parent_label:
            label_to_children.setdefault(step.parent_label, [])
            if step.label not in label_to_children[step.parent_label]:
                label_to_children[step.parent_label].append(step.label)
        else:
            if step.label not in roots:
                roots.append(step.label)

    chains: list[str] = []

    def walk(path: list[str], seen_labels: set[str]) -> None:
        current = path[-1]
        children = label_to_children.get(current, [])
        next_children = [child for child in children if child not in seen_labels]
        if not next_children:
            chains.append(' -> '.join(path))
            return
        for child in next_children:
            if len(chains) >= max_chains:
                return
            walk(path + [child], seen_labels | {child})

    for root in roots:
        if len(chains) >= max_chains:
            break
        walk([root], {root})

    unique_chains = _unique_keep_order(chains)
    return sorted(
        unique_chains,
        key=lambda item: (-item.count('->'), unique_chains.index(item)),
    )[:max_chains]



def _build_location_text(
    source_path: str | None,
    metadata: dict[str, str | int | float | bool | list[str]],
) -> str | None:
    """根据 metadata 构建统一的源码位置文本。"""
    if not source_path:
        return None

    line_start = _coerce_positive_int(metadata.get('line_start'))
    line_end = _coerce_positive_int(metadata.get('line_end'))
    line_number = _coerce_positive_int(metadata.get('line_number'))
    if line_number is not None and line_start is None:
        line_start = line_number
        line_end = line_number

    if line_start is None and line_end is None:
        return source_path
    if line_start is None:
        return f'{source_path}:L{line_end}'
    if line_end is None or line_end == line_start:
        return f'{source_path}:L{line_start}'
    return f'{source_path}:L{line_start}-L{line_end}'



def _load_source_snippet(
    *,
    repo_id: str,
    source_path: str | None,
    metadata: dict[str, str | int | float | bool | list[str]],
    label: str,
) -> str | None:
    """从本地 clone 中截取对应源码片段。"""
    if not source_path:
        return None

    file_path = _resolve_repo_file_path(repo_id, source_path)
    if file_path is None or not file_path.exists() or not file_path.is_file():
        return None

    try:
        lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except OSError:
        return None

    line_start = _coerce_positive_int(metadata.get('line_start'))
    line_end = _coerce_positive_int(metadata.get('line_end'))
    line_number = _coerce_positive_int(metadata.get('line_number'))
    if line_number is not None and line_start is None:
        line_start = line_number
        line_end = line_number

    if line_start is None:
        inferred_line = _find_line_for_label(lines, metadata, label)
        if inferred_line is not None:
            line_start = inferred_line
            line_end = inferred_line

    if line_start is None:
        return None

    line_end = line_end or line_start
    end_line = min(len(lines), max(line_end, line_start) + 2)
    start_line = max(1, line_start - 1)
    if end_line - start_line + 1 > MAX_SOURCE_SNIPPET_LINES:
        end_line = start_line + MAX_SOURCE_SNIPPET_LINES - 1

    snippet_lines: list[str] = []
    for line_index in range(start_line, end_line + 1):
        snippet_lines.append(f'L{line_index}: {lines[line_index - 1]}')
    return '\n'.join(snippet_lines)



def _resolve_repo_file_path(repo_id: str, source_path: str) -> Path | None:
    """把仓库内相对路径映射为本地 clone 文件路径。"""
    try:
        clone_root = get_clone_path(repo_id)
    except Exception:
        return None
    normalized_source_path = source_path.replace('\\', '/').strip('/')
    return clone_root / Path(normalized_source_path)



def _find_line_for_label(
    lines: list[str],
    metadata: dict[str, str | int | float | bool | list[str]],
    label: str,
) -> int | None:
    """在缺少行号时，尝试根据符号名或签名推断大致位置。"""
    candidates = [
        _first_text(metadata.get('signature')),
        _first_text(metadata.get('qualified_name')),
        _first_text(metadata.get('symbol_name')),
        _first_text(metadata.get('handler_qualified_name')),
        _first_text(metadata.get('handler_name')),
        _first_text(metadata.get('route_path')),
        _first_text(metadata.get('code_entity_refs')),
        label,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        keyword = candidate.split('.')[-1]
        for index, line in enumerate(lines, start=1):
            if keyword and keyword in line:
                return index
    return None



def _score_hit_against_question(hit: SearchHit, question_tokens: list[str]) -> int:
    """根据问题 token 与结构化字段重合度做轻量排序。"""
    metadata_text_parts = [hit.document.title, hit.document.source_path or '', hit.snippet]
    for key in (
        'qualified_name',
        'symbol_name',
        'route_path',
        'handler_name',
        'handler_qualified_name',
        'code_entity_refs',
        'code_entity_names',
        'code_relation_targets',
        'code_relation_sources',
        'called_symbols',
        'class_methods',
        'signature',
    ):
        value = hit.document.metadata.get(key)
        if isinstance(value, list):
            metadata_text_parts.extend(str(item) for item in value)
        elif value is not None:
            metadata_text_parts.append(str(value))

    lowered_text = ' '.join(metadata_text_parts).lower()
    score = 0
    for token in question_tokens:
        if token in lowered_text:
            score += 1
    return score



def _extract_prefixed_value(content: str, prefix: str) -> str | None:
    """从结构化知识文档正文中提取某个前缀对应的值。"""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            return value or None
    return None


def _resolve_follow_targets(document: KnowledgeDocument) -> list[str]:
    """优先基于统一实体/关系元数据构建下一跳目标，减少对正文格式的依赖。"""
    metadata = document.metadata
    follow_targets: list[str] = []
    relation_edges = _extract_relation_edges(metadata)

    if document.doc_type == 'api_route_summary':
        follow_targets.extend(
            target_ref for _, target_ref, relation_type in relation_edges if relation_type == 'handle_route'
        )
        follow_targets.extend(
            target_ref for _, target_ref, relation_type in relation_edges if relation_type == 'call'
        )
        handler_ref = _first_text(metadata.get('handler_qualified_name')) or _first_text(metadata.get('handler_name'))
        if handler_ref:
            follow_targets.append(handler_ref)
    elif document.doc_type == 'class_summary':
        follow_targets.extend(
            target_ref for _, target_ref, relation_type in relation_edges if relation_type == 'define_method'
        )
    else:
        follow_targets.extend(
            target_ref for _, target_ref, relation_type in relation_edges if relation_type == 'call'
        )

    follow_targets.extend(_normalize_list(metadata.get('called_symbols')))
    if not follow_targets:
        follow_targets.extend(
            target_ref
            for _, target_ref, relation_type in relation_edges
            if relation_type in {'handle_route', 'call', 'define_method'}
        )
    return _unique_keep_order(follow_targets)


def _extract_relation_edges(
    metadata: dict[str, str | int | float | bool | list[str]],
) -> list[tuple[str, str, str]]:
    """把 metadata 中的关系列表还原为边，并兼容旧版未严格对齐的数据。"""
    sources = _normalize_list(metadata.get('code_relation_sources'))
    targets = _normalize_list(metadata.get('code_relation_targets'))
    relation_types = _normalize_list(metadata.get('code_relation_types'))
    aligned_count = min(len(sources), len(targets), len(relation_types))

    edges: list[tuple[str, str, str]] = [
        (sources[index], targets[index], relation_types[index])
        for index in range(aligned_count)
    ]
    if edges:
        return edges

    default_source = sources[0] if sources else ''
    default_type = relation_types[0] if relation_types else ''
    for index, target_ref in enumerate(targets):
        source_ref = sources[index] if index < len(sources) else default_source
        relation_type = relation_types[index] if index < len(relation_types) else default_type
        if not target_ref:
            continue
        edges.append((source_ref, target_ref, relation_type))
    return edges



def _add_lookup_item(
    lookup: dict[str, list[KnowledgeDocument]],
    key: str,
    document: KnowledgeDocument,
) -> None:
    """向查找表中追加文档，并避免重复。"""
    bucket = lookup.setdefault(key, [])
    if any(item.doc_id == document.doc_id for item in bucket):
        return
    bucket.append(document)



def _normalize_list(value: object) -> list[str]:
    """把 metadata 字段统一展开为字符串列表。"""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []



def _first_text(value: object) -> str:
    """取字符串值的第一个有效文本。"""
    items = _normalize_list(value)
    return items[0] if items else ''



def _coerce_positive_int(value: object) -> int | None:
    """把 metadata 中的数值安全转换为正整数。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted > 0 else None



def _join_items(items: list[str]) -> str:
    """把多个短语拼成适合展示的文本。"""
    return '、'.join(item for item in items if item)



def _unique_keep_order(items: list[str]) -> list[str]:
    """稳定去重并过滤空字符串。"""
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result



def _tokenize(text: str) -> list[str]:
    """对问题做简单切词，兼顾英文标识与中文短语。"""
    lowered = text.lower()
    tokens: list[str] = []
    current: list[str] = []

    for char in lowered:
        if char.isalnum() or char in {'_', '-', '.', '/'}:
            current.append(char)
            continue
        if current:
            tokens.append(''.join(current))
            current = []
    if current:
        tokens.append(''.join(current))

    chinese_chars = [char for char in lowered if '\u4e00' <= char <= '\u9fff']
    for index, char in enumerate(chinese_chars):
        tokens.append(char)
        if index + 1 < len(chinese_chars):
            tokens.append(char + chinese_chars[index + 1])

    route_tokens = re.findall(r'/[a-z0-9_\-/{}/:]+', lowered)
    tokens.extend(route_tokens)
    return _unique_keep_order([token for token in tokens if len(token) >= 2])
