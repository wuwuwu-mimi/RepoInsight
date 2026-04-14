from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from repoinsight.agents.answer_coordinator import (
    _build_code_agent_record,
    _build_recovery_record,
    _build_retrieval_record,
    _build_revision_record,
    _build_route_reason,
    _build_synthesis_record,
    _build_verifier_record,
    _recover_answer_by_issue_tags,
    _revise_answer_if_needed,
    _should_run_code_agent,
    _run_code_agent_pipeline,
    _verify_answer_consistency,
    build_default_answer_agent_plan,
)
from repoinsight.agents.code_agent import build_code_investigation_context_lines
from repoinsight.agents.code_agent import should_use_code_investigation_context
from repoinsight.agents.execution import execute_with_retry
from repoinsight.agents.models import AgentRunRecord, AnswerRouteDecision, CoordinatedAnswerResult
from repoinsight.answer.service import (
    _build_answer_result_from_context,
    _build_empty_answer_result,
    _infer_question_focus,
    _prioritize_hits_for_focus,
    _resolve_retrieval_top_k,
    _select_supporting_lines,
)
from repoinsight.models.rag_model import SearchResult
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR


class LangGraphUnavailableError(RuntimeError):
    """表示当前环境未安装 LangGraph。"""


class _AnswerGraphState(TypedDict, total=False):
    """LangGraph 问答图在节点间流转的状态。"""

    repo_id: str
    question: str
    top_k: int
    target_dir: str
    use_llm: bool
    llm_stream: bool
    on_llm_chunk: Callable[[str], None] | None
    focus: str
    retrieval_top_k: int
    route_decision: AnswerRouteDecision
    search_result: SearchResult
    prioritized_hits: list[Any]
    selected_lines: list[str]
    code_outcome: Any
    extra_context_lines: list[str] | None
    code_investigation: Any
    answer_result: Any
    verification_result: Any
    recovery_result: Any
    revision_applied: bool
    agent_plan: list[Any]
    agent_trace: list[AgentRunRecord]
    shared_context: dict[str, str | int | float | bool | list[str]]



def is_langgraph_available() -> bool:
    """返回当前环境是否可用 LangGraph。"""
    try:
        _load_langgraph_components()
    except LangGraphUnavailableError:
        return False
    return True



def run_langgraph_answer(
    repo_id: str,
    question: str,
    *,
    top_k: int = 5,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
    use_llm: bool = True,
    llm_stream: bool = False,
    on_llm_chunk: Callable[[str], None] | None = None,
) -> CoordinatedAnswerResult:
    """使用 LangGraph 编排问答多 Agent 流程。"""
    compiled_graph = build_answer_state_graph()
    final_state = compiled_graph.invoke(
        {
            'repo_id': repo_id,
            'question': question,
            'top_k': top_k,
            'target_dir': target_dir,
            'use_llm': use_llm,
            'llm_stream': llm_stream,
            'on_llm_chunk': on_llm_chunk,
            'agent_plan': build_default_answer_agent_plan(),
            'agent_trace': [],
            'shared_context': {'orchestrator': 'langgraph'},
        }
    )
    return CoordinatedAnswerResult(
        answer_result=final_state['answer_result'],
        agent_plan=final_state['agent_plan'],
        agent_trace=final_state['agent_trace'],
        route_decision=final_state['route_decision'],
        retrieval_backend=final_state['search_result'].backend,
        retrieval_hit_count=len(final_state['prioritized_hits']),
        retrieved_doc_types=[hit.document.doc_type for hit in final_state['prioritized_hits'][:5]],
        code_investigation=final_state.get('code_investigation'),
        verification_result=final_state.get('verification_result'),
        shared_context=final_state['shared_context'],
    )



def build_answer_state_graph():
    """构建 LangGraph 版问答状态图。"""
    StateGraph, END = _load_langgraph_components()
    graph = StateGraph(_AnswerGraphState)
    graph.add_node('router_agent', _router_node)
    graph.add_node('retrieval_agent', _retrieval_node)
    graph.add_node('code_agent', _code_node)
    graph.add_node('synthesis_agent', _synthesis_node)
    graph.add_node('verifier_agent', _verifier_node)
    graph.add_node('recovery_agent', _recovery_node)
    graph.add_node('revision_agent', _revision_node)
    graph.set_entry_point('router_agent')
    graph.add_edge('router_agent', 'retrieval_agent')
    graph.add_conditional_edges(
        'retrieval_agent',
        _after_retrieval_route,
        {
            'code_agent': 'code_agent',
            'synthesis_agent': 'synthesis_agent',
        },
    )
    graph.add_edge('code_agent', 'synthesis_agent')
    graph.add_edge('synthesis_agent', 'verifier_agent')
    graph.add_edge('verifier_agent', 'recovery_agent')
    graph.add_edge('recovery_agent', 'revision_agent')
    graph.add_edge('revision_agent', END)
    return graph.compile()



def _router_node(state: _AnswerGraphState) -> dict[str, Any]:
    focus = _infer_question_focus(state['question'])
    retrieval_top_k = _resolve_retrieval_top_k(focus, state['top_k'])
    route_decision = AnswerRouteDecision(
        repo_id=state['repo_id'],
        question=state['question'],
        focus=focus,
        retrieval_top_k=retrieval_top_k,
        reason=_build_route_reason(focus, retrieval_top_k),
    )
    agent_trace = list(state['agent_trace'])
    agent_trace.append(
        AgentRunRecord(
            role='router_agent',
            display_name='Router Agent',
            status='success',
            stage_names=['route_question'],
            completed_stage_names=['route_question'],
            detail=route_decision.reason,
            attempt_count=1,
            used_retry=False,
            duration_ms=0,
        )
    )
    shared_context = dict(state['shared_context'])
    shared_context['question_focus'] = focus
    shared_context['retrieval_top_k'] = retrieval_top_k
    return {
        'focus': focus,
        'retrieval_top_k': retrieval_top_k,
        'route_decision': route_decision,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }



def _retrieval_node(state: _AnswerGraphState) -> dict[str, Any]:
    retrieval_outcome = execute_with_retry(
        search_knowledge_base,
        query=state['question'],
        top_k=state['retrieval_top_k'],
        target_dir=state['target_dir'],
        repo_id=state['repo_id'],
        retries=2,
    )
    if retrieval_outcome.error is not None:
        raise retrieval_outcome.error

    search_result = retrieval_outcome.value
    prioritized_hits = _prioritize_hits_for_focus(search_result.hits, state['focus'])
    selected_lines = _select_supporting_lines(state['question'], prioritized_hits, state['focus'])
    agent_trace = list(state['agent_trace'])
    agent_trace.append(
        _build_retrieval_record(
            backend=search_result.backend,
            prioritized_hits=prioritized_hits,
            selected_lines=selected_lines,
            attempt_count=retrieval_outcome.attempt_count,
            used_retry=retrieval_outcome.used_retry,
            duration_ms=retrieval_outcome.duration_ms,
        )
    )
    shared_context = dict(state['shared_context'])
    shared_context['retrieval_backend'] = search_result.backend
    shared_context['retrieval_hit_count'] = len(prioritized_hits)
    shared_context['retrieval_doc_types'] = [hit.document.doc_type for hit in prioritized_hits[:5]]
    shared_context['selected_line_count'] = len(selected_lines)
    return {
        'search_result': search_result,
        'prioritized_hits': prioritized_hits,
        'selected_lines': selected_lines,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }



def _after_retrieval_route(state: _AnswerGraphState) -> str:
    if _should_run_code_agent(state['focus'], state['prioritized_hits']):
        return 'code_agent'
    return 'synthesis_agent'



def _code_node(state: _AnswerGraphState) -> dict[str, Any]:
    code_execution = _run_code_agent_pipeline(
        repo_id=state['repo_id'],
        question=state['question'],
        prioritized_hits=state['prioritized_hits'],
        focus=state['focus'],
        retrieval_top_k=state['retrieval_top_k'],
        target_dir=state['target_dir'],
    )
    code_outcome = code_execution.outcome
    code_investigation = code_outcome.value
    extra_context_lines = None
    code_context_used = False
    if code_investigation is not None and should_use_code_investigation_context(code_investigation):
        extra_context_lines = build_code_investigation_context_lines(code_investigation)
        code_context_used = True

    agent_trace = list(state['agent_trace'])
    agent_trace.append(_build_code_agent_record(code_outcome))
    shared_context = dict(state['shared_context'])
    shared_context['code_agent_enabled'] = True
    shared_context['code_agent_cache_hit'] = code_investigation.cache_hit if code_investigation is not None else False
    shared_context['code_agent_confidence'] = code_investigation.confidence_level if code_investigation is not None else 'none'
    shared_context['code_agent_relevance_score'] = code_investigation.relevance_score if code_investigation is not None else 0.0
    shared_context['code_context_used'] = code_context_used
    shared_context['code_recovery_attempted'] = code_execution.recovery_attempted
    shared_context['code_recovery_improved'] = code_execution.recovery_improved
    shared_context['code_recovery_hit_count'] = code_execution.recovery_hit_count
    shared_context['code_trace_count'] = len(code_investigation.trace_steps) if code_investigation is not None else 0
    shared_context['code_trace_locations'] = code_investigation.evidence_locations[:5] if code_investigation is not None else []
    return {
        'code_outcome': code_outcome,
        'code_investigation': code_investigation,
        'extra_context_lines': extra_context_lines,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }



def _synthesis_node(state: _AnswerGraphState) -> dict[str, Any]:
    code_investigation = state.get('code_investigation')
    agent_trace = list(state['agent_trace'])
    shared_context = dict(state['shared_context'])
    if 'code_agent_enabled' not in shared_context:
        shared_context['code_agent_enabled'] = False
    if 'code_agent_cache_hit' not in shared_context:
        shared_context['code_agent_cache_hit'] = False
    if 'code_agent_confidence' not in shared_context:
        shared_context['code_agent_confidence'] = 'none'
    if 'code_agent_relevance_score' not in shared_context:
        shared_context['code_agent_relevance_score'] = 0.0
    if 'code_context_used' not in shared_context:
        shared_context['code_context_used'] = False
    if 'code_recovery_attempted' not in shared_context:
        shared_context['code_recovery_attempted'] = False
    if 'code_recovery_improved' not in shared_context:
        shared_context['code_recovery_improved'] = False
    if 'code_recovery_hit_count' not in shared_context:
        shared_context['code_recovery_hit_count'] = 0

    if not state['search_result'].hits:
        answer_result = _build_empty_answer_result(
            repo_id=state['repo_id'],
            question=state['question'],
            backend=state['search_result'].backend,
            use_llm=state['use_llm'],
        )
    else:
        answer_result = _build_answer_result_from_context(
            repo_id=state['repo_id'],
            question=state['question'],
            focus=state['focus'],
            backend=state['search_result'].backend,
            prioritized_hits=state['prioritized_hits'],
            selected_lines=state['selected_lines'],
            extra_context_lines=state.get('extra_context_lines'),
            use_llm=state['use_llm'],
            llm_stream=state['llm_stream'],
            on_llm_chunk=state['on_llm_chunk'],
        )

    agent_trace.append(
        _build_synthesis_record(
            answer_result=answer_result,
            selected_lines=state['selected_lines'],
            prioritized_hits=state['prioritized_hits'],
            code_investigation=code_investigation,
            code_context_used=bool(state.get('extra_context_lines')),
        )
    )
    return {
        'answer_result': answer_result,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }


def _verifier_node(state: _AnswerGraphState) -> dict[str, Any]:
    verification_result = _verify_answer_consistency(
        answer_result=state['answer_result'],
        prioritized_hits=state['prioritized_hits'],
        selected_lines=state['selected_lines'],
        code_investigation=state.get('code_investigation'),
        code_context_used=bool(state.get('extra_context_lines')),
    )
    agent_trace = list(state['agent_trace'])
    agent_trace.append(_build_verifier_record(verification_result))
    shared_context = dict(state['shared_context'])
    shared_context['verification_verdict'] = verification_result.verdict
    shared_context['verification_support_score'] = verification_result.support_score
    shared_context['verification_supported_claim_count'] = verification_result.supported_claim_count
    shared_context['verification_checked_claim_count'] = verification_result.checked_claim_count
    shared_context['verification_issue_tags'] = verification_result.issue_tags
    return {
        'verification_result': verification_result,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }


def _revision_node(state: _AnswerGraphState) -> dict[str, Any]:
    answer_result, revision_applied = _revise_answer_if_needed(
        answer_result=state['answer_result'],
        verification_result=state['verification_result'],
        prioritized_hits=state['prioritized_hits'],
        selected_lines=state['selected_lines'],
        code_investigation=state.get('code_investigation'),
        code_context_used=bool(state.get('extra_context_lines')),
    )
    agent_trace = list(state['agent_trace'])
    agent_trace.append(_build_revision_record(revision_applied, state['verification_result']))
    shared_context = dict(state['shared_context'])
    shared_context['revision_applied'] = revision_applied
    return {
        'answer_result': answer_result,
        'revision_applied': revision_applied,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }


def _recovery_node(state: _AnswerGraphState) -> dict[str, Any]:
    initial_verification_result = state['verification_result']
    recovery_result = _recover_answer_by_issue_tags(
        repo_id=state['repo_id'],
        question=state['question'],
        focus=state['focus'],
        retrieval_top_k=state['retrieval_top_k'],
        target_dir=state['target_dir'],
        issue_tags=initial_verification_result.issue_tags,
        prioritized_hits=state['prioritized_hits'],
        selected_lines=state['selected_lines'],
        code_investigation=state.get('code_investigation'),
    )
    prioritized_hits = state['prioritized_hits']
    selected_lines = state['selected_lines']
    code_investigation = state.get('code_investigation')
    extra_context_lines = state.get('extra_context_lines')
    answer_result = state['answer_result']
    verification_result = initial_verification_result

    shared_context = dict(state['shared_context'])
    shared_context['issue_tag_recovery_attempted'] = recovery_result.attempted
    shared_context['issue_tag_recovery_improved'] = recovery_result.improved
    shared_context['issue_tag_recovery_actions'] = recovery_result.actions

    if recovery_result.improved:
        prioritized_hits = recovery_result.prioritized_hits
        selected_lines = recovery_result.selected_lines
        code_investigation = recovery_result.code_investigation
        extra_context_lines = recovery_result.extra_context_lines
        shared_context['retrieval_hit_count'] = len(prioritized_hits)
        shared_context['retrieval_doc_types'] = [hit.document.doc_type for hit in prioritized_hits[:5]]
        shared_context['selected_line_count'] = len(selected_lines)
        if recovery_result.code_execution is not None:
            shared_context['code_agent_enabled'] = True
            shared_context['code_recovery_attempted'] = recovery_result.code_execution.recovery_attempted
            shared_context['code_recovery_improved'] = recovery_result.code_execution.recovery_improved
            shared_context['code_recovery_hit_count'] = recovery_result.code_execution.recovery_hit_count
        if code_investigation is not None:
            shared_context['code_agent_cache_hit'] = code_investigation.cache_hit
            shared_context['code_agent_confidence'] = code_investigation.confidence_level
            shared_context['code_agent_relevance_score'] = code_investigation.relevance_score
            shared_context['code_trace_count'] = len(code_investigation.trace_steps)
            shared_context['code_trace_locations'] = code_investigation.evidence_locations[:5]
        else:
            shared_context['code_agent_cache_hit'] = False
            shared_context['code_agent_confidence'] = 'none'
            shared_context['code_agent_relevance_score'] = 0.0
            shared_context['code_trace_count'] = 0
            shared_context['code_trace_locations'] = []
        shared_context['code_context_used'] = recovery_result.code_context_used

        # 恢复阶段统一使用稳定的抽取式重建，避免在同一轮图执行里重复触发 LLM。
        if prioritized_hits:
            answer_result = _build_answer_result_from_context(
                repo_id=state['repo_id'],
                question=state['question'],
                focus=state['focus'],
                backend=state['search_result'].backend,
                prioritized_hits=prioritized_hits,
                selected_lines=selected_lines,
                extra_context_lines=extra_context_lines,
                use_llm=False,
                llm_stream=False,
                on_llm_chunk=None,
            )
        else:
            answer_result = _build_empty_answer_result(
                repo_id=state['repo_id'],
                question=state['question'],
                backend=state['search_result'].backend,
                use_llm=False,
            )

        verification_result = _verify_answer_consistency(
            answer_result=answer_result,
            prioritized_hits=prioritized_hits,
            selected_lines=selected_lines,
            code_investigation=code_investigation,
            code_context_used=bool(extra_context_lines),
        )
        shared_context['verification_verdict'] = verification_result.verdict
        shared_context['verification_support_score'] = verification_result.support_score
        shared_context['verification_supported_claim_count'] = verification_result.supported_claim_count
        shared_context['verification_checked_claim_count'] = verification_result.checked_claim_count
        shared_context['verification_issue_tags'] = verification_result.issue_tags

    agent_trace = list(state['agent_trace'])
    agent_trace.append(
        _build_recovery_record(
            initial_verification_result=initial_verification_result,
            recovery_result=recovery_result,
            final_verification_result=verification_result,
        )
    )
    return {
        'recovery_result': recovery_result,
        'prioritized_hits': prioritized_hits,
        'selected_lines': selected_lines,
        'code_investigation': code_investigation,
        'extra_context_lines': extra_context_lines,
        'answer_result': answer_result,
        'verification_result': verification_result,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }



def _load_langgraph_components():
    """按需导入 LangGraph，避免未安装时影响本地默认链路。"""
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:  # pragma: no cover - 仅在真实缺包环境触发
        raise LangGraphUnavailableError(
            '当前未安装 LangGraph，请先执行 `pip install langgraph` 或 `uv add langgraph`。'
        ) from exc
    return StateGraph, END
