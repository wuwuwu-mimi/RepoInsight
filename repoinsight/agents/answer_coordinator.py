from collections.abc import Callable
from dataclasses import dataclass, field

from repoinsight.agents.architecture_agent import (
    build_architecture_investigation_context_lines,
    investigate_architecture_hits,
    should_use_architecture_investigation_context,
)
from repoinsight.agents.code_agent import (
    build_code_investigation_context_lines,
    investigate_code_hits,
    should_use_code_investigation_context,
)
from repoinsight.agents.execution import ExecutionOutcome, execute_with_retry, run_parallel_tasks
from repoinsight.agents.models import (
    AgentEvidenceItem,
    AgentRunRecord,
    AgentStructuredOutput,
    AgentStepSpec,
    AnswerRouteDecision,
    CodeInvestigationResult,
    AnswerVerificationResult,
    CoordinatedAnswerResult,
)
from repoinsight.answer.service import (
    _build_answer_result_from_context,
    _build_empty_answer_result,
    _infer_question_focus,
    _prioritize_hits_for_focus,
    _resolve_retrieval_top_k,
    _select_supporting_lines,
)
from repoinsight.answer.formatter import format_structured_answer
from repoinsight.models.rag_model import SearchHit
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR


DEFAULT_ANSWER_AGENT_SPECS: tuple[dict[str, object], ...] = (
    {
        'role': 'router_agent',
        'display_name': 'Router Agent',
        'description': '负责识别问题焦点并决定检索策略。',
        'stage_names': ['route_question'],
        'depends_on': [],
        'can_run_in_parallel': False,
    },
    {
        'role': 'retrieval_agent',
        'display_name': 'Retrieval Agent',
        'description': '负责检索知识库、重排结果并筛选支持证据。',
        'stage_names': ['search_knowledge_base', 'prioritize_hits', 'select_supporting_lines'],
        'depends_on': ['router_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'code_agent',
        'display_name': 'Code Agent',
        'description': '在实现与接口类问题下提炼函数、类、路由和调用链等代码线索。',
        'stage_names': ['investigate_code_context'],
        'depends_on': ['retrieval_agent'],
        'can_run_in_parallel': True,
    },
    {
        'role': 'architecture_agent',
        'display_name': 'Architecture Agent',
        'description': '在架构类问题下提炼入口、模块依赖、跨文件调用链与关系链等结构线索。',
        'stage_names': ['investigate_architecture_context'],
        'depends_on': ['retrieval_agent'],
        'can_run_in_parallel': True,
    },
    {
        'role': 'synthesis_agent',
        'display_name': 'Synthesis Agent',
        'description': '负责组织最终回答，并按需调用 LLM 做自然语言润色。',
        'stage_names': ['build_answer_result'],
        'depends_on': ['retrieval_agent', 'code_agent', 'architecture_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'verifier_agent',
        'display_name': 'Verifier Agent',
        'description': '负责检查最终回答是否被当前证据充分支撑，并提示潜在风险。',
        'stage_names': ['verify_answer_consistency'],
        'depends_on': ['synthesis_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'recovery_agent',
        'display_name': 'Recovery Agent',
        'description': '根据 verifier 的失败标签执行补救检索、代码扩查与上下文重建。',
        'stage_names': ['recover_by_issue_tags'],
        'depends_on': ['verifier_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'revision_agent',
        'display_name': 'Revision Agent',
        'description': '当回答支撑不足时，负责收敛结论并输出更保守的修订版本。',
        'stage_names': ['revise_answer_if_needed'],
        'depends_on': ['recovery_agent'],
        'can_run_in_parallel': False,
    },
)


CODE_AGENT_FOCUSES = {'implementation', 'api'}
ARCHITECTURE_AGENT_FOCUSES = {'architecture'}
CODE_AGENT_RECOVERY_TOP_K_GROWTH = 4
CODE_AGENT_RECOVERY_TOP_K_CAP = 12
CODE_AGENT_RECOVERY_MAX_HITS = 8
CODE_AGENT_RECOVERY_MAX_FOLLOW_STEPS = 10
CODE_AGENT_RECOVERY_MAX_FOLLOW_DEPTH = 3
ANSWER_RECOVERY_TOP_K_GROWTH = 4
ANSWER_RECOVERY_TOP_K_CAP = 12
ANSWER_RECOVERY_RETRIEVAL_TAGS = {
    'retrieval_sparse',
    'supporting_lines_missing',
    'evidence_weak',
    'no_explicit_evidence',
    'claim_not_explicit',
}
ANSWER_RECOVERY_CODE_TAGS = {
    'code_confidence_low',
    'code_context_missing',
}

INVESTIGATION_AGENT_SPECS = {
    'code_agent': {
        'display_name': 'Code Agent',
        'stage_name': 'investigate_code_context',
        'failure_label': '代码调查',
        'empty_label': '实现线索',
        'trace_label': '代码链路线索',
    },
    'architecture_agent': {
        'display_name': 'Architecture Agent',
        'stage_name': 'investigate_architecture_context',
        'failure_label': '架构调查',
        'empty_label': '模块线索',
        'trace_label': '架构链路线索',
    },
}


@dataclass(slots=True)
class CodeAgentExecutionResult:
    """表示 code_agent 一次完整执行后的结果，包括恢复扩检状态。"""

    outcome: ExecutionOutcome
    recovery_attempted: bool = False
    recovery_improved: bool = False
    recovery_hit_count: int = 0


@dataclass(slots=True)
class AnswerRecoveryResult:
    """表示 recovery_agent 基于 issue_tags 执行恢复后的上下文。"""

    prioritized_hits: list[SearchHit]
    selected_lines: list[str]
    code_execution: CodeAgentExecutionResult | None = None
    code_investigation: CodeInvestigationResult | None = None
    extra_context_lines: list[str] | None = None
    code_context_used: bool = False
    attempted: bool = False
    improved: bool = False
    actions: list[str] = field(default_factory=list)



def build_default_answer_agent_plan() -> list[AgentStepSpec]:
    """构建一份默认的问答多 Agent 计划。"""
    return [AgentStepSpec.model_validate(item) for item in DEFAULT_ANSWER_AGENT_SPECS]


def _resolve_investigation_agent_role(focus: str) -> str | None:
    """根据问题焦点决定本轮应使用哪个调查 Agent。"""
    if focus in CODE_AGENT_FOCUSES:
        return 'code_agent'
    if focus in ARCHITECTURE_AGENT_FOCUSES:
        return 'architecture_agent'
    return None


def _get_investigation_agent_spec(role: str | None) -> dict[str, str]:
    """返回调查 Agent 的展示配置，缺省时回退到 code_agent 规格。"""
    if role is None:
        return INVESTIGATION_AGENT_SPECS['code_agent']
    return INVESTIGATION_AGENT_SPECS.get(role, INVESTIGATION_AGENT_SPECS['code_agent'])


def _should_use_investigation_context(
    focus: str,
    investigation: CodeInvestigationResult | None,
) -> bool:
    """按焦点选择对应的上下文吸收策略。"""
    if focus in ARCHITECTURE_AGENT_FOCUSES:
        return should_use_architecture_investigation_context(investigation)
    return should_use_code_investigation_context(investigation)


def _build_investigation_context_lines(
    focus: str,
    investigation: CodeInvestigationResult,
) -> list[str]:
    """按焦点生成对应调查 Agent 的补充上下文。"""
    if focus in ARCHITECTURE_AGENT_FOCUSES:
        return build_architecture_investigation_context_lines(investigation)
    return build_code_investigation_context_lines(investigation)


def run_multi_agent_answer(
    repo_id: str,
    question: str,
    *,
    top_k: int = 5,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
    use_llm: bool = True,
    llm_stream: bool = False,
    on_llm_chunk: Callable[[str], None] | None = None,
) -> CoordinatedAnswerResult:
    """以多 Agent 视角执行问答编排，并补充轻量并行、重试和状态流转。"""
    agent_plan = build_default_answer_agent_plan()
    agent_trace: list[AgentRunRecord] = []
    shared_context: dict[str, str | int | float | bool | list[str]] = {'orchestrator': 'local'}

    focus = _infer_question_focus(question)
    retrieval_top_k = _resolve_retrieval_top_k(focus, top_k)
    shared_context['question_focus'] = focus
    shared_context['retrieval_top_k'] = retrieval_top_k
    route_decision = AnswerRouteDecision(
        repo_id=repo_id,
        question=question,
        focus=focus,
        retrieval_top_k=retrieval_top_k,
        reason=_build_route_reason(focus, retrieval_top_k),
    )
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

    retrieval_outcome = execute_with_retry(
        search_knowledge_base,
        query=question,
        top_k=retrieval_top_k,
        target_dir=target_dir,
        repo_id=repo_id,
        retries=2,
    )
    if retrieval_outcome.error is not None:
        raise retrieval_outcome.error

    search_result = retrieval_outcome.value
    prioritized_hits = _prioritize_hits_for_focus(search_result.hits, focus)
    shared_context['retrieval_backend'] = search_result.backend
    shared_context['retrieval_hit_count'] = len(prioritized_hits)
    shared_context['retrieval_doc_types'] = [hit.document.doc_type for hit in prioritized_hits[:5]]

    investigation_agent_role = _resolve_investigation_agent_role(focus)
    parallel_tasks = {
        'selected_lines': lambda: _select_supporting_lines(question, prioritized_hits, focus),
    }
    if investigation_agent_role is not None and prioritized_hits:
        parallel_tasks['investigation_execution'] = lambda: _run_investigation_agent_pipeline(
            repo_id=repo_id,
            question=question,
            prioritized_hits=prioritized_hits,
            focus=focus,
            retrieval_top_k=retrieval_top_k,
            target_dir=target_dir,
        )

    parallel_results = run_parallel_tasks(parallel_tasks)
    selected_lines = parallel_results['selected_lines']
    code_execution = parallel_results.get('investigation_execution')
    code_outcome = code_execution.outcome if code_execution is not None else None
    shared_context['selected_line_count'] = len(selected_lines)
    shared_context['code_agent_enabled'] = investigation_agent_role == 'code_agent' and code_outcome is not None
    shared_context['architecture_agent_enabled'] = (
        investigation_agent_role == 'architecture_agent' and code_outcome is not None
    )
    shared_context['investigation_agent_role'] = investigation_agent_role or 'none'

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

    code_investigation: CodeInvestigationResult | None = None
    extra_context_lines: list[str] | None = None
    if code_outcome is not None:
        code_investigation = code_outcome.value
        should_absorb_code_context = _should_use_investigation_context(focus, code_investigation)
        shared_context['code_agent_cache_hit'] = code_investigation.cache_hit if code_investigation is not None else False
        shared_context['code_agent_confidence'] = code_investigation.confidence_level if code_investigation is not None else 'none'
        shared_context['code_agent_relevance_score'] = code_investigation.relevance_score if code_investigation is not None else 0.0
        shared_context['code_context_used'] = should_absorb_code_context
        shared_context['code_recovery_attempted'] = code_execution.recovery_attempted
        shared_context['code_recovery_improved'] = code_execution.recovery_improved
        shared_context['code_recovery_hit_count'] = code_execution.recovery_hit_count
        if code_investigation is not None:
            shared_context['code_trace_count'] = len(code_investigation.trace_steps)
            shared_context['code_trace_locations'] = code_investigation.evidence_locations[:5]
            if should_absorb_code_context:
                extra_context_lines = _build_investigation_context_lines(focus, code_investigation)
        else:
            shared_context['code_trace_count'] = 0
            shared_context['code_trace_locations'] = []
        agent_trace.append(
            _build_investigation_agent_record(
                code_outcome,
                role=investigation_agent_role or 'code_agent',
                focus=focus,
            )
        )
    else:
        shared_context['code_agent_cache_hit'] = False
        shared_context['code_agent_confidence'] = 'none'
        shared_context['code_agent_relevance_score'] = 0.0
        shared_context['code_context_used'] = False
        shared_context['code_recovery_attempted'] = False
        shared_context['code_recovery_improved'] = False
        shared_context['code_recovery_hit_count'] = 0
        shared_context['code_trace_count'] = 0
        shared_context['code_trace_locations'] = []

    if not search_result.hits:
        answer_result = _build_empty_answer_result(
            repo_id=repo_id,
            question=question,
            backend=search_result.backend,
            use_llm=use_llm,
        )
    else:
        answer_result = _build_answer_result_from_context(
            repo_id=repo_id,
            question=question,
            focus=focus,
            backend=search_result.backend,
            prioritized_hits=prioritized_hits,
            selected_lines=selected_lines,
            extra_context_lines=extra_context_lines,
            use_llm=use_llm,
            llm_stream=llm_stream,
            on_llm_chunk=on_llm_chunk,
            relation_chain_details=(
                code_investigation.relation_chain_details if code_investigation is not None else None
            ),
        )

    agent_trace.append(
        _build_synthesis_record(
            answer_result=answer_result,
            selected_lines=selected_lines,
            prioritized_hits=prioritized_hits,
            code_investigation=code_investigation,
            code_context_used=bool(extra_context_lines),
            investigation_agent_role=investigation_agent_role,
        )
    )
    verification_result = _verify_answer_consistency(
        answer_result=answer_result,
        prioritized_hits=prioritized_hits,
        selected_lines=selected_lines,
        code_investigation=code_investigation,
        code_context_used=bool(extra_context_lines),
        investigation_agent_role=investigation_agent_role,
    )
    shared_context['verification_verdict'] = verification_result.verdict
    shared_context['verification_support_score'] = verification_result.support_score
    shared_context['verification_supported_claim_count'] = verification_result.supported_claim_count
    shared_context['verification_checked_claim_count'] = verification_result.checked_claim_count
    shared_context['verification_issue_tags'] = verification_result.issue_tags
    agent_trace.append(_build_verifier_record(verification_result))
    initial_verification_result = verification_result
    recovery_result = _recover_answer_by_issue_tags(
        repo_id=repo_id,
        question=question,
        focus=focus,
        retrieval_top_k=retrieval_top_k,
        target_dir=target_dir,
        issue_tags=verification_result.issue_tags,
        prioritized_hits=prioritized_hits,
        selected_lines=selected_lines,
        code_investigation=code_investigation,
        investigation_agent_role=investigation_agent_role,
    )
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
            shared_context['code_agent_enabled'] = investigation_agent_role == 'code_agent'
            shared_context['architecture_agent_enabled'] = investigation_agent_role == 'architecture_agent'
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
        # 恢复阶段优先稳定补证据，不重复触发 LLM，避免同一轮问答二次生成拖慢响应。
        if prioritized_hits:
            answer_result = _build_answer_result_from_context(
                repo_id=repo_id,
                question=question,
                focus=focus,
                backend=search_result.backend,
                prioritized_hits=prioritized_hits,
                selected_lines=selected_lines,
                extra_context_lines=extra_context_lines,
                use_llm=False,
                llm_stream=False,
                on_llm_chunk=None,
                relation_chain_details=(
                    code_investigation.relation_chain_details if code_investigation is not None else None
                ),
            )
        else:
            answer_result = _build_empty_answer_result(
                repo_id=repo_id,
                question=question,
                backend=search_result.backend,
                use_llm=False,
            )
        verification_result = _verify_answer_consistency(
            answer_result=answer_result,
            prioritized_hits=prioritized_hits,
            selected_lines=selected_lines,
            code_investigation=code_investigation,
            code_context_used=bool(extra_context_lines),
            investigation_agent_role=investigation_agent_role,
        )
        shared_context['verification_verdict'] = verification_result.verdict
        shared_context['verification_support_score'] = verification_result.support_score
        shared_context['verification_supported_claim_count'] = verification_result.supported_claim_count
        shared_context['verification_checked_claim_count'] = verification_result.checked_claim_count
        shared_context['verification_issue_tags'] = verification_result.issue_tags
    agent_trace.append(
        _build_recovery_record(
            initial_verification_result=initial_verification_result,
            recovery_result=recovery_result,
            final_verification_result=verification_result,
        )
    )
    answer_result, revision_applied = _revise_answer_if_needed(
        answer_result=answer_result,
        verification_result=verification_result,
        prioritized_hits=prioritized_hits,
        selected_lines=selected_lines,
        code_investigation=code_investigation,
        code_context_used=bool(extra_context_lines),
        investigation_agent_role=investigation_agent_role,
    )
    shared_context['revision_applied'] = revision_applied
    agent_trace.append(_build_revision_record(revision_applied, verification_result))

    return CoordinatedAnswerResult(
        answer_result=answer_result,
        agent_plan=agent_plan,
        agent_trace=agent_trace,
        route_decision=route_decision,
        retrieval_backend=search_result.backend,
        retrieval_hit_count=len(prioritized_hits),
        retrieved_doc_types=[hit.document.doc_type for hit in prioritized_hits[:5]],
        code_investigation=code_investigation,
        verification_result=verification_result,
        shared_context=shared_context,
    )



def _should_run_code_agent(focus: str, prioritized_hits: list[SearchHit]) -> bool:
    """仅在实现类或接口类问题下触发 code_agent。"""
    return focus in CODE_AGENT_FOCUSES and bool(prioritized_hits)


def _should_run_architecture_agent(focus: str, prioritized_hits: list[SearchHit]) -> bool:
    """仅在架构类问题下触发 architecture_agent。"""
    return focus in ARCHITECTURE_AGENT_FOCUSES and bool(prioritized_hits)



def _build_route_reason(focus: str, retrieval_top_k: int) -> str:
    """构建 router_agent 的路由说明。"""
    focus_labels = {
        'overview': '项目概览问题',
        'startup': '启动问题',
        'entrypoint': '入口定位问题',
        'env': '环境变量问题',
        'service': '服务依赖问题',
        'config': '配置问题',
        'architecture': '架构问题',
        'api': '接口问题',
        'implementation': '实现细节问题',
        'generic': '通用问题',
    }
    return f'问题被路由为 {focus_labels.get(focus, focus)}，实际检索 top_k={retrieval_top_k}。'


def _build_agent_structured_output(
    *,
    conclusions: list[str] | None = None,
    evidence: list[AgentEvidenceItem] | None = None,
    uncertainties: list[str] | None = None,
    next_actions: list[str] | None = None,
    metadata: dict[str, str | int | float | bool | list[str]] | None = None,
) -> AgentStructuredOutput:
    """构建统一的 Agent 结构化输出。"""
    return AgentStructuredOutput(
        conclusions=conclusions or [],
        evidence=evidence or [],
        uncertainties=uncertainties or [],
        next_actions=next_actions or [],
        metadata=metadata or {},
    )


def _build_text_evidence(
    label: str,
    *,
    kind: str = 'text',
    source_path: str | None = None,
    location: str | None = None,
    snippet: str | None = None,
) -> AgentEvidenceItem:
    """构建一条通用文本证据。"""
    return AgentEvidenceItem(
        kind=kind,
        label=label,
        source_path=source_path,
        location=location,
        snippet=snippet,
    )



def _build_retrieval_record(
    *,
    backend: str,
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    attempt_count: int,
    used_retry: bool,
    duration_ms: int,
) -> AgentRunRecord:
    """构建 retrieval_agent 的执行记录。"""
    evidence_items: list[AgentEvidenceItem] = []
    for hit in prioritized_hits[:3]:
        evidence_items.append(
            _build_text_evidence(
                f'{hit.document.doc_type}::{hit.document.source_path or "repo"}',
                kind='document',
                source_path=hit.document.source_path,
                snippet=hit.snippet,
            )
        )
    for line in selected_lines[:2]:
        evidence_items.append(_build_text_evidence(line, kind='supporting_line'))

    if not prioritized_hits:
        detail = f'检索后端为 {backend}，当前没有召回可用证据。'
    else:
        doc_types = ', '.join(hit.document.doc_type for hit in prioritized_hits[:3])
        detail = (
            f'检索后端为 {backend}，召回 {len(prioritized_hits)} 条结果；'
            f'前 3 个文档类型为 {doc_types}；'
            f'筛出 {len(selected_lines)} 条支持句。'
        )
    return AgentRunRecord(
        role='retrieval_agent',
        display_name='Retrieval Agent',
        status='success',
        stage_names=['search_knowledge_base', 'prioritize_hits', 'select_supporting_lines'],
        completed_stage_names=['search_knowledge_base', 'prioritize_hits', 'select_supporting_lines'],
        detail=detail,
        attempt_count=attempt_count,
        used_retry=used_retry,
        duration_ms=duration_ms,
        structured_output=_build_agent_structured_output(
            conclusions=(
                [f'当前使用 {backend} 检索后端，召回 {len(prioritized_hits)} 条候选证据。']
                + (
                    [f'当前首批命中文档类型集中在：{", ".join(hit.document.doc_type for hit in prioritized_hits[:3])}。']
                    if prioritized_hits
                    else []
                )
            ),
            evidence=evidence_items,
            uncertainties=(
                ['当前没有召回可用证据。']
                if not prioritized_hits
                else (['当前已召回文档，但还没筛出高质量支持句。'] if not selected_lines else [])
            ),
            next_actions=(
                ['建议改写问题关键词，或扩大 top_k 后再检索。']
                if not prioritized_hits
                else ['将命中结果交给 synthesis_agent 组织回答。']
            ),
            metadata={
                'backend': backend,
                'hit_count': len(prioritized_hits),
                'selected_line_count': len(selected_lines),
                'top_doc_types': [hit.document.doc_type for hit in prioritized_hits[:5]],
            },
        ),
    )



def _build_investigation_agent_record(
    code_outcome: ExecutionOutcome,
    *,
    role: str,
    focus: str,
) -> AgentRunRecord:
    """构建 code_agent / architecture_agent 的执行记录。"""
    spec = _get_investigation_agent_spec(role)
    code_investigation = code_outcome.value
    if code_outcome.error is not None:
        return AgentRunRecord(
            role=role,
            display_name=spec['display_name'],
            status='failed',
            stage_names=[spec['stage_name']],
            completed_stage_names=[],
            detail=f"{spec['failure_label']}执行失败，当前已回退到仅使用检索证据回答。",
            error_message=str(code_outcome.error),
            attempt_count=code_outcome.attempt_count,
            used_retry=code_outcome.used_retry,
            duration_ms=code_outcome.duration_ms,
            structured_output=_build_agent_structured_output(
                conclusions=[f"{spec['failure_label']}执行失败，当前回退为仅使用检索证据回答。"],
                uncertainties=[str(code_outcome.error)],
                next_actions=['由 synthesis_agent 基于已有检索证据生成保守回答。'],
                metadata={'attempt_count': code_outcome.attempt_count},
            ),
        )
    if code_investigation is None:
        return AgentRunRecord(
            role=role,
            display_name=spec['display_name'],
            status='success',
            stage_names=[spec['stage_name']],
            completed_stage_names=[spec['stage_name']],
            detail=f"当前已触发{spec['failure_label']}，但没有从召回结果中提炼出额外{spec['empty_label']}。",
            attempt_count=code_outcome.attempt_count,
            used_retry=code_outcome.used_retry,
            duration_ms=code_outcome.duration_ms,
            structured_output=_build_agent_structured_output(
                conclusions=[f"当前已执行{spec['failure_label']}，但没有提炼出额外{spec['empty_label']}。"],
                uncertainties=['代码文档与问题的贴合度不足，暂时无法形成稳定调用链。'],
                next_actions=['优先交给 synthesis_agent 使用检索证据回答。'],
                metadata={'attempt_count': code_outcome.attempt_count},
            ),
        )

    detail_parts = [code_investigation.summary]
    detail_parts.append(f'置信度：{code_investigation.confidence_level}')
    detail_parts.append(f'相关性评分：{code_investigation.relevance_score:.2f}')
    if code_investigation.cache_hit:
        detail_parts.append('命中缓存')
    if code_investigation.recovery_attempted:
        if code_investigation.recovery_improved:
            detail_parts.append('已执行自动扩检恢复并提升结果')
        else:
            detail_parts.append('已执行自动扩检恢复，但原结果仍更优')
    if code_investigation.evidence_locations:
        detail_parts.append(f'关键位置：{", ".join(code_investigation.evidence_locations[:3])}')
    if code_investigation.matched_routes:
        detail_parts.append(f'命中路由：{", ".join(code_investigation.matched_routes[:2])}')
    if code_investigation.matched_symbols:
        detail_parts.append(f'命中符号：{", ".join(code_investigation.matched_symbols[:3])}')
    if code_investigation.relation_chains:
        detail_parts.append(f'关系链：{code_investigation.relation_chains[0]}')
    if code_investigation.relation_chain_details:
        detail_parts.append(f'类型化关系链：{code_investigation.relation_chain_details[0].typed_text}')
    if code_investigation.quality_notes:
        detail_parts.append(f'质量说明：{"；".join(code_investigation.quality_notes[:2])}')
    return AgentRunRecord(
        role=role,
        display_name=spec['display_name'],
        status='success',
        stage_names=[spec['stage_name']],
        completed_stage_names=[spec['stage_name']],
        detail='；'.join(detail_parts),
        attempt_count=code_outcome.attempt_count,
        used_retry=code_outcome.used_retry,
        duration_ms=code_outcome.duration_ms,
        structured_output=_build_agent_structured_output(
            conclusions=[
                code_investigation.summary,
                f'当前代码调查置信度为 {code_investigation.confidence_level}，相关性评分为 {code_investigation.relevance_score:.2f}。',
            ],
            evidence=[
                *[
                    _build_text_evidence(location, kind='location', location=location)
                    for location in code_investigation.evidence_locations[:3]
                ],
                *[
                    _build_text_evidence(symbol, kind='symbol')
                    for symbol in code_investigation.matched_symbols[:3]
                ],
                *[
                    _build_text_evidence(chain.typed_text, kind='relation_chain')
                    for chain in code_investigation.relation_chain_details[:1]
                ],
                *[
                    _build_text_evidence(note, kind='quality_note')
                    for note in code_investigation.quality_notes[:2]
                ],
            ],
            uncertainties=(
                []
                if code_investigation.confidence_level == 'high'
                else [f"当前{spec['failure_label']}置信度为 {code_investigation.confidence_level}，仍建议结合源文件复核。"]
            ),
            next_actions=(
                [f"将{spec['trace_label']}交给 synthesis_agent 融入最终回答。"]
                if _should_use_investigation_context(focus, code_investigation)
                else ['如 verifier 判定支撑不足，可由 recovery_agent 再做扩检。']
            ),
            metadata={
                'focus': code_investigation.focus,
                'agent_role': role,
                'confidence_level': code_investigation.confidence_level,
                'relevance_score': round(code_investigation.relevance_score, 4),
                'cache_hit': code_investigation.cache_hit,
                'recovery_attempted': code_investigation.recovery_attempted,
                'recovery_improved': code_investigation.recovery_improved,
                'trace_step_count': len(code_investigation.trace_steps),
            },
        ),
    )



def _build_synthesis_record(
    *,
    answer_result,
    selected_lines: list[str],
    prioritized_hits: list[SearchHit],
    code_investigation: CodeInvestigationResult | None,
    code_context_used: bool,
    investigation_agent_role: str | None,
) -> AgentRunRecord:
    """构建 synthesis_agent 的执行记录。"""
    investigation_role_text = investigation_agent_role or 'code_agent'
    detail_parts = [
        f'回答模式为 {answer_result.answer_mode}',
        f'证据数 {len(answer_result.evidence)}',
        f'支持句 {len(selected_lines)} 条',
    ]
    if prioritized_hits:
        detail_parts.append(f'首条证据类型为 {prioritized_hits[0].document.doc_type}')
    if code_investigation is not None:
        if code_context_used:
            detail_parts.append(f'已吸收 {investigation_role_text} 线索 {len(code_investigation.trace_steps)} 步')
        else:
            detail_parts.append(
                f'{investigation_role_text} 已执行但未注入回答，上下文置信度为 {code_investigation.confidence_level}'
            )
    if answer_result.llm_attempted:
        detail_parts.append('已尝试调用 LLM')
    if answer_result.llm_error:
        detail_parts.append(f'LLM 回退原因：{answer_result.llm_error}')
    return AgentRunRecord(
        role='synthesis_agent',
        display_name='Synthesis Agent',
        status='success',
        stage_names=['build_answer_result'],
        completed_stage_names=['build_answer_result'],
        detail='；'.join(detail_parts),
        attempt_count=1,
        used_retry=False,
        duration_ms=None,
        structured_output=_build_agent_structured_output(
            conclusions=[
                f'当前回答模式为 {answer_result.answer_mode}，已组织 {len(answer_result.evidence)} 条证据。'
            ]
            + (
                [f'当前回答已吸收 {len(code_investigation.trace_steps)} 步 {investigation_role_text} 线索。']
                if code_investigation is not None and code_context_used
                else []
            ),
            evidence=[
                *[_build_text_evidence(line, kind='supporting_line') for line in selected_lines[:2]],
                *[
                    _build_text_evidence(
                        f'{item.doc_type}::{item.source_path or "repo"}',
                        kind='answer_evidence',
                        source_path=item.source_path,
                        snippet=item.snippet,
                    )
                    for item in answer_result.evidence[:3]
                ],
            ],
            uncertainties=(
                [answer_result.llm_error]
                if answer_result.llm_error
                else ([] if answer_result.evidence else ['当前回答可用证据仍然偏少。'])
            ),
            next_actions=['交给 verifier_agent 检查回答与证据是否一致。'],
            metadata={
                'answer_mode': answer_result.answer_mode,
                'evidence_count': len(answer_result.evidence),
                'selected_line_count': len(selected_lines),
                'code_context_used': code_context_used,
                'llm_attempted': answer_result.llm_attempted,
                'fallback_used': answer_result.fallback_used,
            },
        ),
    )



def _build_verifier_record(verification_result: AnswerVerificationResult) -> AgentRunRecord:
    """构建 verifier_agent 的执行记录。"""
    detail_parts = [
        f'验证结论为 {verification_result.verdict}',
        f'支撑评分 {verification_result.support_score:.2f}',
        f'已支撑 {verification_result.supported_claim_count}/{verification_result.checked_claim_count} 条结论',
    ]
    if verification_result.issues:
        detail_parts.append(f'风险提示：{"；".join(verification_result.issues[:2])}')
    if verification_result.issue_tags:
        detail_parts.append(f'原因标签：{", ".join(verification_result.issue_tags[:3])}')
    if verification_result.notes:
        detail_parts.append(f'验证说明：{"；".join(verification_result.notes[:2])}')
    return AgentRunRecord(
        role='verifier_agent',
        display_name='Verifier Agent',
        status='success',
        stage_names=['verify_answer_consistency'],
        completed_stage_names=['verify_answer_consistency'],
        detail='；'.join(detail_parts),
        attempt_count=1,
        used_retry=False,
        duration_ms=None,
        structured_output=_build_agent_structured_output(
            conclusions=[
                f'当前验证结论为 {verification_result.verdict}。',
                f'已有 {verification_result.supported_claim_count}/{verification_result.checked_claim_count} 条结论被证据支撑。',
            ],
            evidence=[
                *[
                    _build_text_evidence(issue_tag, kind='issue_tag')
                    for issue_tag in verification_result.issue_tags[:4]
                ],
                *[
                    _build_text_evidence(issue, kind='issue')
                    for issue in verification_result.issues[:3]
                ],
                *[
                    _build_text_evidence(note, kind='note')
                    for note in verification_result.notes[:2]
                ],
            ],
            uncertainties=verification_result.issues[:3],
            next_actions=(
                ['交给 recovery_agent 按 issue_tags 做补救恢复。']
                if verification_result.issue_tags
                else ['当前验证通过，可直接进入最终交付。']
            ),
            metadata={
                'verdict': verification_result.verdict,
                'support_score': round(verification_result.support_score, 4),
                'checked_claim_count': verification_result.checked_claim_count,
                'supported_claim_count': verification_result.supported_claim_count,
            },
        ),
    )


def _build_revision_record(
    revision_applied: bool,
    verification_result: AnswerVerificationResult,
) -> AgentRunRecord:
    """构建 revision_agent 的执行记录。"""
    if not revision_applied:
        return AgentRunRecord(
            role='revision_agent',
            display_name='Revision Agent',
            status='skipped',
            stage_names=['revise_answer_if_needed'],
            completed_stage_names=[],
            detail=f'当前验证结论为 {verification_result.verdict}，无需修订回答。',
            attempt_count=1,
            used_retry=False,
            duration_ms=None,
            structured_output=_build_agent_structured_output(
                conclusions=[f'当前验证结论为 {verification_result.verdict}，无需修订回答。'],
                next_actions=['保持当前回答作为最终输出。'],
                metadata={'revision_applied': False, 'verdict': verification_result.verdict},
            ),
        )
    detail = (
        f'根据 verifier 结论 {verification_result.verdict} 已收敛回答内容；'
        f'原验证支撑评分为 {verification_result.support_score:.2f}。'
    )
    return AgentRunRecord(
        role='revision_agent',
        display_name='Revision Agent',
        status='success',
        stage_names=['revise_answer_if_needed'],
        completed_stage_names=['revise_answer_if_needed'],
        detail=detail,
        attempt_count=1,
        used_retry=False,
        duration_ms=None,
        structured_output=_build_agent_structured_output(
            conclusions=['已根据 verifier 反馈收敛回答内容，输出更保守的最终版本。'],
            evidence=[
                _build_text_evidence(
                    f'原验证结论：{verification_result.verdict}',
                    kind='verification_verdict',
                )
            ],
            uncertainties=['修订后的回答会更偏保守，可能减少部分推断性结论。'],
            next_actions=['输出修订后的最终回答。'],
            metadata={
                'revision_applied': True,
                'verdict': verification_result.verdict,
                'support_score': round(verification_result.support_score, 4),
            },
        ),
    )


def _build_recovery_record(
    *,
    initial_verification_result: AnswerVerificationResult,
    recovery_result: AnswerRecoveryResult,
    final_verification_result: AnswerVerificationResult,
) -> AgentRunRecord:
    """构建 recovery_agent 的执行记录。"""
    if not recovery_result.attempted:
        return AgentRunRecord(
            role='recovery_agent',
            display_name='Recovery Agent',
            status='skipped',
            stage_names=['recover_by_issue_tags'],
            completed_stage_names=[],
            detail='当前原因标签不需要额外恢复动作，直接进入修订阶段。',
            attempt_count=1,
            used_retry=False,
            duration_ms=None,
            structured_output=_build_agent_structured_output(
                conclusions=['当前原因标签不需要额外恢复动作。'],
                next_actions=['直接进入 revision_agent。'],
                metadata={'attempted': False, 'improved': False},
            ),
        )

    detail_parts = []
    if recovery_result.actions:
        detail_parts.append(f'恢复动作：{"；".join(recovery_result.actions[:3])}')
    if recovery_result.improved:
        detail_parts.append(
            f'验证结论由 {initial_verification_result.verdict} 改善为 {final_verification_result.verdict}'
        )
        detail_parts.append(f'支撑评分提升到 {final_verification_result.support_score:.2f}')
    else:
        detail_parts.append('已尝试恢复，但没有拿到更强的证据上下文。')
    return AgentRunRecord(
        role='recovery_agent',
        display_name='Recovery Agent',
        status='success',
        stage_names=['recover_by_issue_tags'],
        completed_stage_names=['recover_by_issue_tags'],
        detail='；'.join(detail_parts),
        attempt_count=1,
        used_retry=False,
        duration_ms=None,
        structured_output=_build_agent_structured_output(
            conclusions=(
                [f'恢复后验证结论由 {initial_verification_result.verdict} 改善为 {final_verification_result.verdict}。']
                if recovery_result.improved
                else ['已尝试恢复，但没有拿到更强的证据上下文。']
            ),
            evidence=[
                *[
                    _build_text_evidence(
                        action,
                        kind='recovery_action',
                    )
                    for action in recovery_result.actions[:3]
                ],
            ],
            uncertainties=(
                []
                if recovery_result.improved
                else ['当前恢复没有提升回答支撑度，后续只能走保守修订。']
            ),
            next_actions=['将恢复结果交给 revision_agent 决定是否收敛回答。'],
            metadata={
                'attempted': recovery_result.attempted,
                'improved': recovery_result.improved,
                'initial_verdict': initial_verification_result.verdict,
                'final_verdict': final_verification_result.verdict,
                'action_count': len(recovery_result.actions),
            },
        ),
    )


def _verify_answer_consistency(
    *,
    answer_result,
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    code_investigation: CodeInvestigationResult | None,
    code_context_used: bool,
    investigation_agent_role: str | None = None,
) -> AnswerVerificationResult:
    """检查最终回答是否被当前证据充分支撑。"""
    checked_claim_count = 0
    supported_claim_count = 0
    notes: list[str] = []
    issues: list[str] = []
    issue_tags: list[str] = []

    answer_lines = [line.strip() for line in answer_result.answer.splitlines() if line.strip()]
    candidate_claims = [
        line.lstrip('- ').strip()
        for line in answer_lines
        if not line.startswith(('结论', '依据', '不确定点', '补充线索'))
    ]
    evidence_pool = _build_verifier_evidence_pool(
        prioritized_hits=prioritized_hits,
        selected_lines=selected_lines,
        code_investigation=code_investigation,
        code_context_used=code_context_used,
        answer_result=answer_result,
    )

    for claim in candidate_claims[:6]:
        claim_tokens = _tokenize_verifier_text(claim)
        if not claim_tokens:
            continue
        checked_claim_count += 1
        if _is_claim_supported(claim_tokens, evidence_pool):
            supported_claim_count += 1

    if checked_claim_count == 0:
        checked_claim_count = 1
        if answer_result.evidence:
            supported_claim_count = 1
            notes.append('未抽取到明确结论条目，按回答整体与证据是否存在做兜底校验。')
            issue_tags.append('claim_not_explicit')
        else:
            issues.append('当前回答没有可用于验证的显式证据。')
            issue_tags.append('no_explicit_evidence')

    support_score = round(supported_claim_count / max(checked_claim_count, 1), 2)
    if support_score >= 0.8:
        verdict = 'passed'
    elif support_score >= 0.45:
        verdict = 'warning'
        issues.append('部分结论缺少足够直接的证据支撑。')
        issue_tags.append('evidence_weak')
    else:
        verdict = 'failed'
        issues.append('回答中的主要结论与现有证据重合较弱，建议回看源码或重新提问。')
        issue_tags.append('claim_too_broad')

    if len(prioritized_hits) <= 1:
        issue_tags.append('retrieval_sparse')
    if not selected_lines:
        issue_tags.append('supporting_lines_missing')

    if prioritized_hits:
        notes.append(f'本次验证参考了 {len(prioritized_hits[:5])} 条主证据文档。')
    if code_investigation is not None:
        investigation_role_text = investigation_agent_role or _resolve_investigation_agent_role(code_investigation.focus) or 'code_agent'
        if code_context_used:
            notes.append(f'已纳入 {investigation_role_text} 的 {len(code_investigation.trace_steps)} 步代码线索。')
        else:
            notes.append(
                f'{investigation_role_text} 已执行，但其置信度为 {code_investigation.confidence_level}，未直接纳入。'
            )
            if code_investigation.confidence_level == 'low':
                issue_tags.append('code_confidence_low')
    if code_investigation is None and prioritized_hits and any(
        hit.document.doc_type in {'function_summary', 'class_summary', 'api_route_summary'}
        for hit in prioritized_hits
    ):
        issue_tags.append('code_context_missing')
    return AnswerVerificationResult(
        verdict=verdict,
        support_score=support_score,
        checked_claim_count=checked_claim_count,
        supported_claim_count=supported_claim_count,
        issues=_unique_preserve_order(issues),
        issue_tags=_unique_preserve_order(issue_tags),
        notes=_unique_preserve_order(notes),
    )


def _revise_answer_if_needed(
    *,
    answer_result,
    verification_result: AnswerVerificationResult,
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    code_investigation: CodeInvestigationResult | None,
    code_context_used: bool,
    investigation_agent_role: str | None = None,
):
    """在回答支撑不足时自动生成一个更保守的修订版本。"""
    if verification_result.verdict == 'passed':
        return answer_result, False

    sections = _parse_structured_answer_sections(answer_result.answer)
    supported_conclusions = _select_supported_conclusions(
        conclusions=sections['conclusions'],
        prioritized_hits=prioritized_hits,
        selected_lines=selected_lines,
        code_investigation=code_investigation,
        code_context_used=code_context_used,
        answer_result=answer_result,
    )
    if not supported_conclusions:
        supported_conclusions = _build_conservative_conclusions(
            prioritized_hits=prioritized_hits,
            selected_lines=selected_lines,
            code_investigation=code_investigation,
            code_context_used=code_context_used,
            issue_tags=verification_result.issue_tags,
            investigation_agent_role=investigation_agent_role,
        )

    revised_evidence = _unique_preserve_order(
        sections['evidence'] + selected_lines[:3] + [item.snippet for item in answer_result.evidence]
    )[:8]
    revised_uncertainties = _unique_preserve_order(
        sections['uncertainties']
        + verification_result.issues
        + (
            [f'失败原因标签：{", ".join(verification_result.issue_tags)}']
            if verification_result.issue_tags
            else []
        )
        + verification_result.notes
        + ['已根据 verifier 检查结果收敛回答，建议优先参考列出的证据与源码位置。']
    )[:6]
    revised_answer = format_structured_answer(
        conclusions=supported_conclusions[:4],
        evidence=revised_evidence or ['当前仅能确认有限证据，请结合源码继续核对。'],
        uncertainties=revised_uncertainties,
    )
    return answer_result.model_copy(update={'answer': revised_answer}), True


def _recover_answer_by_issue_tags(
    *,
    repo_id: str,
    question: str,
    focus: str,
    retrieval_top_k: int,
    target_dir: str,
    issue_tags: list[str],
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    code_investigation: CodeInvestigationResult | None,
    investigation_agent_role: str | None = None,
) -> AnswerRecoveryResult:
    """根据 verifier 标签做一次轻量恢复，优先补证据而不是直接改写结论。"""
    code_context_used = bool(
        code_investigation is not None and _should_use_investigation_context(focus, code_investigation)
    )
    recovery_result = AnswerRecoveryResult(
        prioritized_hits=list(prioritized_hits),
        selected_lines=list(selected_lines),
        code_investigation=code_investigation,
        extra_context_lines=(
            _build_investigation_context_lines(focus, code_investigation)
            if code_context_used and code_investigation is not None
            else None
        ),
        code_context_used=code_context_used,
    )
    issue_tag_set = set(issue_tags)
    retrieval_improved = False

    if issue_tag_set & ANSWER_RECOVERY_RETRIEVAL_TAGS:
        recovery_result.attempted = True
        expanded_top_k = min(
            ANSWER_RECOVERY_TOP_K_CAP,
            max(retrieval_top_k + ANSWER_RECOVERY_TOP_K_GROWTH, len(prioritized_hits) + 1, 4),
        )
        expanded_search_outcome = execute_with_retry(
            search_knowledge_base,
            query=question,
            top_k=expanded_top_k,
            target_dir=target_dir,
            repo_id=repo_id,
            retries=2,
        )
        if expanded_search_outcome.error is not None or expanded_search_outcome.value is None:
            recovery_result.actions.append('尝试扩大检索范围失败，暂时沿用原始召回结果。')
        else:
            candidate_hits = _prioritize_hits_for_focus(expanded_search_outcome.value.hits, focus)
            candidate_lines = _select_supporting_lines(question, candidate_hits, focus)
            if _score_retrieval_context(candidate_hits, candidate_lines) > _score_retrieval_context(
                recovery_result.prioritized_hits,
                recovery_result.selected_lines,
            ):
                recovery_result.prioritized_hits = candidate_hits
                recovery_result.selected_lines = candidate_lines
                recovery_result.improved = True
                retrieval_improved = True
                recovery_result.actions.append(
                    f'已将检索范围扩到 top_k={expanded_top_k}，补充到 {len(candidate_lines)} 条支持句。'
                )
            else:
                recovery_result.actions.append('已扩大检索范围，但新增证据没有明显提升。')

    investigation_spec = _get_investigation_agent_spec(
        investigation_agent_role or _resolve_investigation_agent_role(focus)
    )
    should_retry_code = (
        _resolve_investigation_agent_role(focus) is not None
        and bool(recovery_result.prioritized_hits)
        and (bool(issue_tag_set & ANSWER_RECOVERY_CODE_TAGS) or retrieval_improved)
    )
    if should_retry_code:
        recovery_result.attempted = True
        code_execution = _run_investigation_agent_pipeline(
            repo_id=repo_id,
            question=question,
            prioritized_hits=recovery_result.prioritized_hits,
            focus=focus,
            retrieval_top_k=max(retrieval_top_k, len(recovery_result.prioritized_hits)),
            target_dir=target_dir,
        )
        candidate_code = code_execution.outcome.value
        if code_execution.outcome.error is not None or candidate_code is None:
            recovery_result.actions.append(
                f"已重新执行{investigation_spec['failure_label']}，但没有拿到可用的{investigation_spec['empty_label']}。"
            )
        elif _is_better_code_recovery(recovery_result.code_investigation, candidate_code):
            recovery_result.code_execution = code_execution
            recovery_result.code_investigation = candidate_code
            recovery_result.code_context_used = _should_use_investigation_context(focus, candidate_code)
            recovery_result.extra_context_lines = (
                _build_investigation_context_lines(focus, candidate_code)
                if recovery_result.code_context_used
                else None
            )
            recovery_result.improved = True
            recovery_result.actions.append(
                f"已重新执行{investigation_spec['failure_label']}，当前置信度提升为 {candidate_code.confidence_level}。"
            )
        else:
            recovery_result.actions.append(
                f"已重新执行{investigation_spec['failure_label']}，但没有得到更可信的代码上下文。"
            )

    return recovery_result


def _build_verifier_evidence_pool(
    *,
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    code_investigation: CodeInvestigationResult | None,
    code_context_used: bool,
    answer_result,
) -> list[str]:
    """整理 verifier_agent 可用的证据文本池。"""
    evidence_pool: list[str] = []
    evidence_pool.extend(selected_lines)
    for hit in prioritized_hits[:5]:
        evidence_pool.append(hit.snippet)
        evidence_pool.append(hit.document.title)
        evidence_pool.append(hit.document.content[:300])
    for evidence in answer_result.evidence:
        evidence_pool.append(evidence.snippet)
        if evidence.source_path:
            evidence_pool.append(evidence.source_path)
    if code_investigation is not None:
        evidence_pool.append(code_investigation.summary)
        evidence_pool.extend(code_investigation.implementation_notes[:4])
        if code_context_used:
            evidence_pool.extend(code_investigation.evidence_locations[:5])
            evidence_pool.extend(code_investigation.matched_symbols[:5])
    return [item for item in evidence_pool if item]


def _is_claim_supported(claim_tokens: list[str], evidence_pool: list[str]) -> bool:
    """判断某条结论是否在证据池中有足够重合。"""
    if not evidence_pool:
        return False
    min_overlap = 2 if len(claim_tokens) >= 3 else 1
    for evidence_text in evidence_pool:
        lowered = evidence_text.lower()
        overlap = sum(1 for token in claim_tokens if token in lowered)
        if overlap >= min_overlap:
            return True
    return False


def _tokenize_verifier_text(text: str) -> list[str]:
    """对 verifier_agent 使用的文本做轻量切词。"""
    import re

    lowered = text.lower()
    latin_tokens = re.findall(r'[a-z0-9_\-\.\/]+', lowered)
    chinese_sequences = re.findall(r'[一-鿿]{2,}', lowered)
    tokens = list(latin_tokens)
    for sequence in chinese_sequences:
        tokens.append(sequence)
        tokens.extend(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return _unique_preserve_order([token for token in tokens if len(token) >= 2])


def _parse_structured_answer_sections(answer_text: str) -> dict[str, list[str]]:
    """解析三段式回答，便于 revision_agent 重组内容。"""
    sections = {
        'conclusions': [],
        'evidence': [],
        'uncertainties': [],
    }
    current_section = 'conclusions'
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('结论'):
            current_section = 'conclusions'
            continue
        if line.startswith('依据'):
            current_section = 'evidence'
            continue
        if line.startswith('不确定点'):
            current_section = 'uncertainties'
            continue
        sections[current_section].append(line.lstrip('- ').strip())
    return sections


def _select_supported_conclusions(
    *,
    conclusions: list[str],
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    code_investigation: CodeInvestigationResult | None,
    code_context_used: bool,
    answer_result,
) -> list[str]:
    """从原结论中挑出 verifier 认为更有证据支撑的部分。"""
    evidence_pool = _build_verifier_evidence_pool(
        prioritized_hits=prioritized_hits,
        selected_lines=selected_lines,
        code_investigation=code_investigation,
        code_context_used=code_context_used,
        answer_result=answer_result,
    )
    supported: list[str] = []
    for claim in conclusions:
        claim_tokens = _tokenize_verifier_text(claim)
        if claim_tokens and _is_claim_supported(claim_tokens, evidence_pool):
            supported.append(claim)
    return _unique_preserve_order(supported)


def _build_conservative_conclusions(
    *,
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
    code_investigation: CodeInvestigationResult | None,
    code_context_used: bool,
    issue_tags: list[str],
    investigation_agent_role: str | None = None,
) -> list[str]:
    """当原结论支撑不足时，生成更保守的替代结论。"""
    investigation_role_text = investigation_agent_role or (
        _resolve_investigation_agent_role(code_investigation.focus) if code_investigation is not None else None
    )
    conclusions: list[str] = []
    if 'retrieval_sparse' in issue_tags:
        conclusions.append('当前召回证据较少，回答仅能覆盖局部线索。')
    if 'code_confidence_low' in issue_tags:
        if investigation_role_text == 'architecture_agent':
            conclusions.append('架构调查置信度偏低，当前不建议把模块依赖路径当作最终定论。')
        else:
            conclusions.append('代码调查置信度偏低，当前不建议把代码路径当作最终定论。')
    if selected_lines:
        conclusions.append(_strip_verifier_source_prefix(selected_lines[0]))
    if prioritized_hits:
        first_hit = prioritized_hits[0]
        source = first_hit.document.source_path or first_hit.document.doc_type
        conclusions.append(f'当前最直接的证据来自 {source}，可先围绕该文件继续核对实现。')
    if code_investigation is not None and code_context_used and code_investigation.evidence_locations:
        if investigation_role_text == 'architecture_agent':
            conclusions.append(f'架构线索当前主要落在 {code_investigation.evidence_locations[0]}。')
        else:
            conclusions.append(f'代码线索当前主要落在 {code_investigation.evidence_locations[0]}。')
    if not conclusions:
        conclusions.append('当前只能确认部分局部证据，尚不足以支撑更强结论。')
    return _unique_preserve_order(conclusions)


def _strip_verifier_source_prefix(line: str) -> str:
    """去掉 selected_lines 内部来源前缀，保留正文。"""
    if '] ' in line:
        return line.split('] ', maxsplit=1)[1].strip()
    return line.strip()


def _score_retrieval_context(
    prioritized_hits: list[SearchHit],
    selected_lines: list[str],
) -> tuple[int, int, float]:
    """为恢复阶段的检索上下文生成一个轻量质量分值。"""
    return (
        len(selected_lines),
        min(len(prioritized_hits), 5),
        round(sum(hit.score for hit in prioritized_hits[:3]), 4),
    )


def _is_better_code_recovery(
    current: CodeInvestigationResult | None,
    candidate: CodeInvestigationResult,
) -> bool:
    """判断恢复阶段拿到的代码调查是否优于当前结果。"""
    if current is None:
        return True
    return _score_code_investigation_result(candidate) > _score_code_investigation_result(current)


def _unique_preserve_order(items: list[str]) -> list[str]:
    """稳定去重，保留原始顺序。"""
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _run_code_agent_with_retry(
    *,
    repo_id: str,
    question: str,
    prioritized_hits: list[SearchHit],
    focus: str,
    target_dir: str,
) -> ExecutionOutcome:
    """执行 code_agent，并在局部失败时做轻量重试。"""
    return execute_with_retry(
        investigate_code_hits,
        question,
        prioritized_hits,
        focus,
        repo_id=repo_id,
        target_dir=target_dir,
        retries=2,
    )


def _run_architecture_agent_with_retry(
    *,
    repo_id: str,
    question: str,
    prioritized_hits: list[SearchHit],
    target_dir: str,
) -> ExecutionOutcome:
    """执行 architecture_agent，并在局部失败时做轻量重试。"""
    return execute_with_retry(
        investigate_architecture_hits,
        question,
        prioritized_hits,
        repo_id=repo_id,
        target_dir=target_dir,
        retries=2,
    )


def _run_investigation_agent_pipeline(
    *,
    repo_id: str,
    question: str,
    prioritized_hits: list[SearchHit],
    focus: str,
    retrieval_top_k: int,
    target_dir: str,
) -> CodeAgentExecutionResult:
    """根据问题焦点分流到 code_agent 或 architecture_agent。"""
    if focus in ARCHITECTURE_AGENT_FOCUSES:
        return _run_architecture_agent_pipeline(
            repo_id=repo_id,
            question=question,
            prioritized_hits=prioritized_hits,
            focus=focus,
            retrieval_top_k=retrieval_top_k,
            target_dir=target_dir,
        )
    return _run_code_agent_pipeline(
        repo_id=repo_id,
        question=question,
        prioritized_hits=prioritized_hits,
        focus=focus,
        retrieval_top_k=retrieval_top_k,
        target_dir=target_dir,
    )


def _run_code_agent_pipeline(
    *,
    repo_id: str,
    question: str,
    prioritized_hits: list[SearchHit],
    focus: str,
    retrieval_top_k: int,
    target_dir: str,
) -> CodeAgentExecutionResult:
    """执行 code_agent，并在低置信度时自动进行一次扩检恢复。"""
    initial_outcome = _run_code_agent_with_retry(
        repo_id=repo_id,
        question=question,
        prioritized_hits=prioritized_hits,
        focus=focus,
        target_dir=target_dir,
    )
    return _recover_code_agent_if_needed(
        repo_id=repo_id,
        question=question,
        prioritized_hits=prioritized_hits,
        focus=focus,
        retrieval_top_k=retrieval_top_k,
        target_dir=target_dir,
        initial_outcome=initial_outcome,
        role='code_agent',
    )


def _run_architecture_agent_pipeline(
    *,
    repo_id: str,
    question: str,
    prioritized_hits: list[SearchHit],
    focus: str,
    retrieval_top_k: int,
    target_dir: str,
) -> CodeAgentExecutionResult:
    """执行 architecture_agent，并在低置信度时自动进行一次扩检恢复。"""
    initial_outcome = _run_architecture_agent_with_retry(
        repo_id=repo_id,
        question=question,
        prioritized_hits=prioritized_hits,
        target_dir=target_dir,
    )
    return _recover_code_agent_if_needed(
        repo_id=repo_id,
        question=question,
        prioritized_hits=prioritized_hits,
        focus=focus,
        retrieval_top_k=retrieval_top_k,
        target_dir=target_dir,
        initial_outcome=initial_outcome,
        role='architecture_agent',
    )


def _recover_code_agent_if_needed(
    *,
    repo_id: str,
    question: str,
    prioritized_hits: list[SearchHit],
    focus: str,
    retrieval_top_k: int,
    target_dir: str,
    initial_outcome: ExecutionOutcome,
    role: str = 'code_agent',
) -> CodeAgentExecutionResult:
    """当首次调查置信度不足时，尝试自动扩检并挑选更优结果。"""
    investigation_spec = _get_investigation_agent_spec(role)
    initial_value = initial_outcome.value
    if initial_outcome.error is not None or initial_value is None:
        return CodeAgentExecutionResult(outcome=initial_outcome)
    if _should_use_investigation_context(focus, initial_value):
        return CodeAgentExecutionResult(outcome=initial_outcome)

    expanded_hits = prioritized_hits
    recovery_hit_count = len(prioritized_hits)
    expanded_top_k = min(
        CODE_AGENT_RECOVERY_TOP_K_CAP,
        max(retrieval_top_k + CODE_AGENT_RECOVERY_TOP_K_GROWTH, len(prioritized_hits)),
    )
    if expanded_top_k > len(prioritized_hits):
        expanded_search_outcome = execute_with_retry(
            search_knowledge_base,
            query=question,
            top_k=expanded_top_k,
            target_dir=target_dir,
            repo_id=repo_id,
            retries=2,
        )
        if expanded_search_outcome.error is None and expanded_search_outcome.value is not None:
            expanded_hits = _prioritize_hits_for_focus(expanded_search_outcome.value.hits, focus)
            recovery_hit_count = len(expanded_hits)

    if role == 'architecture_agent':
        recovery_outcome = execute_with_retry(
            investigate_architecture_hits,
            question,
            expanded_hits,
            repo_id=repo_id,
            target_dir=target_dir,
            max_hits=min(max(len(expanded_hits), 1), CODE_AGENT_RECOVERY_MAX_HITS),
            max_follow_steps=CODE_AGENT_RECOVERY_MAX_FOLLOW_STEPS,
            max_follow_depth=CODE_AGENT_RECOVERY_MAX_FOLLOW_DEPTH,
            retries=2,
        )
    else:
        recovery_outcome = execute_with_retry(
            investigate_code_hits,
            question,
            expanded_hits,
            focus,
            repo_id=repo_id,
            target_dir=target_dir,
            max_hits=min(max(len(expanded_hits), 1), CODE_AGENT_RECOVERY_MAX_HITS),
            max_follow_steps=CODE_AGENT_RECOVERY_MAX_FOLLOW_STEPS,
            max_follow_depth=CODE_AGENT_RECOVERY_MAX_FOLLOW_DEPTH,
            retries=2,
        )
    if recovery_outcome.error is not None or recovery_outcome.value is None:
        return CodeAgentExecutionResult(
            outcome=_annotate_code_outcome(
                initial_outcome,
                recovery_attempted=True,
                recovery_improved=False,
                recovery_hit_count=recovery_hit_count,
                recovery_note=f"首次{investigation_spec['failure_label']}置信度较低，已尝试自动扩检，但未拿到更优结果。",
                total_attempt_count=initial_outcome.attempt_count + recovery_outcome.attempt_count,
                total_duration_ms=initial_outcome.duration_ms + recovery_outcome.duration_ms,
            ),
            recovery_attempted=True,
            recovery_improved=False,
            recovery_hit_count=recovery_hit_count,
        )

    selected_outcome, recovery_improved = _pick_preferred_code_outcome(
        initial_outcome=initial_outcome,
        recovery_outcome=recovery_outcome,
    )
    recovery_note = f"首次{investigation_spec['failure_label']}置信度较低，已自动扩检并获得更优结果。"
    if not recovery_improved:
        recovery_note = f"首次{investigation_spec['failure_label']}置信度较低，已自动扩检，但原结果仍然更优。"
    return CodeAgentExecutionResult(
        outcome=_annotate_code_outcome(
            selected_outcome,
            recovery_attempted=True,
            recovery_improved=recovery_improved,
            recovery_hit_count=recovery_hit_count,
            recovery_note=recovery_note,
            total_attempt_count=initial_outcome.attempt_count + recovery_outcome.attempt_count,
            total_duration_ms=initial_outcome.duration_ms + recovery_outcome.duration_ms,
        ),
        recovery_attempted=True,
        recovery_improved=recovery_improved,
        recovery_hit_count=recovery_hit_count,
    )


def _pick_preferred_code_outcome(
    *,
    initial_outcome: ExecutionOutcome,
    recovery_outcome: ExecutionOutcome,
) -> tuple[ExecutionOutcome, bool]:
    """从初次调查和扩检调查之间选出更优结果。"""
    initial_value = initial_outcome.value
    recovery_value = recovery_outcome.value
    if initial_value is None:
        return recovery_outcome, True
    if recovery_value is None:
        return initial_outcome, False

    initial_score = _score_code_investigation_result(initial_value)
    recovery_score = _score_code_investigation_result(recovery_value)
    if recovery_score > initial_score:
        return recovery_outcome, True
    return initial_outcome, False


def _score_code_investigation_result(result: CodeInvestigationResult) -> tuple[int, float, int, int, int]:
    """为代码调查结果生成可比较的质量分值。"""
    return (
        1 if _should_use_investigation_context(result.focus, result) else 0,
        result.relevance_score,
        len(result.trace_steps),
        len(result.evidence_locations),
        len(result.called_symbols),
    )


def _annotate_code_outcome(
    outcome: ExecutionOutcome,
    *,
    recovery_attempted: bool,
    recovery_improved: bool,
    recovery_hit_count: int,
    recovery_note: str,
    total_attempt_count: int | None = None,
    total_duration_ms: int | None = None,
) -> ExecutionOutcome:
    """给 code_agent 结果补充恢复扩检说明。"""
    value = outcome.value
    if value is None:
        return outcome

    notes = list(value.quality_notes)
    if recovery_note and recovery_note not in notes:
        notes.append(recovery_note)
    annotated_value = value.model_copy(
        update={
            'quality_notes': notes,
            'recovery_attempted': recovery_attempted,
            'recovery_improved': recovery_improved,
        }
    )
    if recovery_hit_count > len(value.source_paths):
        annotated_value = annotated_value.model_copy(
            update={
                'quality_notes': notes + [f'扩检阶段额外评估了 {recovery_hit_count} 条候选代码文档。'],
            }
        )
    return ExecutionOutcome(
        value=annotated_value,
        attempt_count=total_attempt_count or outcome.attempt_count,
        duration_ms=total_duration_ms or outcome.duration_ms,
        error=outcome.error,
    )
