from repoinsight.agents.execution import execute_with_retry
from repoinsight.agents.models import AgentRunRecord, AgentStepSpec, CoordinatedAnalysisResult
from repoinsight.analyze.pipeline import run_analysis
from repoinsight.models.analysis_model import StageTraceEntry
from repoinsight.models.rag_model import IndexResult
from repoinsight.storage.index_service import index_analysis_result


DEFAULT_ANALYSIS_AGENT_SPECS: tuple[dict[str, object], ...] = (
    {
        'role': 'metadata_agent',
        'display_name': 'Metadata Agent',
        'description': '负责 URL 校验、仓库元数据与 README 获取。',
        'stage_names': ['validate_analysis_url_stage', 'fetch_repo_metadata_stage'],
        'depends_on': [],
        'can_run_in_parallel': False,
    },
    {
        'role': 'structure_agent',
        'display_name': 'Structure Agent',
        'description': '负责克隆仓库、扫描目录并读取关键文件。',
        'stage_names': ['clone_repository_stage', 'scan_repository_stage', 'read_key_files_stage'],
        'depends_on': ['metadata_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'profile_agent',
        'display_name': 'Profile Agent',
        'description': '负责生成项目画像与技术栈信号。',
        'stage_names': ['build_project_profile_stage', 'build_tech_stack_stage'],
        'depends_on': ['structure_agent'],
        'can_run_in_parallel': True,
    },
    {
        'role': 'insight_agent',
        'display_name': 'Insight Agent',
        'description': '负责生成项目类型、优势、风险与观察结论。',
        'stage_names': ['build_repo_insights_stage'],
        'depends_on': ['profile_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'memory_agent',
        'display_name': 'Memory Agent',
        'description': '负责把分析结果写入知识文档与向量索引。',
        'stage_names': [],
        'depends_on': ['insight_agent'],
        'can_run_in_parallel': False,
    },
)

STAGE_TO_AGENT_ROLE = {
    'validate_analysis_url_stage': 'metadata_agent',
    'fetch_repo_metadata_stage': 'metadata_agent',
    'clone_repository_stage': 'structure_agent',
    'scan_repository_stage': 'structure_agent',
    'read_key_files_stage': 'structure_agent',
    'build_project_profile_stage': 'profile_agent',
    'build_tech_stack_stage': 'profile_agent',
    'build_repo_insights_stage': 'insight_agent',
}



def build_default_analysis_agent_plan(*, include_memory_agent: bool = True) -> list[AgentStepSpec]:
    """构建一份默认的分析多 Agent 计划。"""
    plan: list[AgentStepSpec] = []
    for item in DEFAULT_ANALYSIS_AGENT_SPECS:
        if not include_memory_agent and item['role'] == 'memory_agent':
            continue
        plan.append(AgentStepSpec.model_validate(item))
    return plan



def build_agent_trace_from_stage_trace(
    stage_trace: list[StageTraceEntry],
    *,
    include_memory_agent: bool = False,
) -> list[AgentRunRecord]:
    """把 stage trace 聚合成更高层的 Agent 执行记录。"""
    plan = build_default_analysis_agent_plan(include_memory_agent=include_memory_agent)
    records: list[AgentRunRecord] = []

    for step in plan:
        if step.role == 'memory_agent':
            records.append(
                AgentRunRecord(
                    role=step.role,
                    display_name=step.display_name,
                    status='pending',
                    stage_names=[],
                    completed_stage_names=[],
                    detail='尚未执行知识入库。',
                )
            )
            continue

        related_entries = [
            entry for entry in stage_trace if STAGE_TO_AGENT_ROLE.get(entry.stage_name) == step.role
        ]
        records.append(_build_agent_run_record(step, related_entries))

    return records



def run_multi_agent_analysis(
    url: str,
    *,
    persist_knowledge: bool = False,
) -> CoordinatedAnalysisResult:
    """在不改动现有 pipeline 的前提下，提供一层多 Agent 视角的分析编排。"""
    analysis_result = run_analysis(url)
    agent_plan = build_default_analysis_agent_plan(include_memory_agent=persist_knowledge)
    agent_trace = build_agent_trace_from_stage_trace(
        analysis_result.stage_trace,
        include_memory_agent=False,
    )

    shared_context = _build_analysis_shared_context(analysis_result)
    index_result: IndexResult | None = None
    if persist_knowledge:
        memory_outcome = execute_with_retry(index_analysis_result, analysis_result, retries=2)
        if memory_outcome.error is not None:
            raise memory_outcome.error
        index_result = memory_outcome.value
        agent_trace.append(_build_memory_agent_record(index_result, memory_outcome.attempt_count, memory_outcome.used_retry, memory_outcome.duration_ms))
        shared_context['memory_vector_indexed'] = index_result.vector_indexed
        shared_context['memory_vector_backend'] = index_result.vector_backend

    return CoordinatedAnalysisResult(
        analysis_result=analysis_result,
        agent_plan=agent_plan,
        agent_trace=agent_trace,
        index_result=index_result,
        shared_context=shared_context,
    )



def _build_agent_run_record(
    step: AgentStepSpec,
    related_entries: list[StageTraceEntry],
) -> AgentRunRecord:
    """根据某个 Agent 负责的 stage 列表，汇总执行状态。"""
    if not related_entries:
        return AgentRunRecord(
            role=step.role,
            display_name=step.display_name,
            status='pending',
            stage_names=step.stage_names,
            completed_stage_names=[],
            detail='对应 stage 尚未执行。',
        )

    failed_entries = [entry for entry in related_entries if entry.status == 'failed']
    completed_stage_names = [entry.stage_name for entry in related_entries if entry.status == 'success']

    if failed_entries:
        first_failure = failed_entries[0]
        return AgentRunRecord(
            role=step.role,
            display_name=step.display_name,
            status='failed',
            stage_names=step.stage_names,
            completed_stage_names=completed_stage_names,
            detail=_summarize_stage_details(related_entries),
            error_message=first_failure.error_message,
        )

    return AgentRunRecord(
        role=step.role,
        display_name=step.display_name,
        status='success',
        stage_names=step.stage_names,
        completed_stage_names=completed_stage_names,
        detail=_summarize_stage_details(related_entries),
    )



def _build_memory_agent_record(
    index_result: IndexResult,
    attempt_count: int,
    used_retry: bool,
    duration_ms: int,
) -> AgentRunRecord:
    """把知识入库结果包装成 memory_agent 的执行记录。"""
    detail_parts = [f'知识文档已保存到 {index_result.local_path}']
    if index_result.vector_indexed:
        if index_result.message:
            detail_parts.append(index_result.message)
        else:
            detail_parts.append(f'向量索引已写入 {index_result.vector_backend}')
    elif index_result.message:
        detail_parts.append(index_result.message)

    return AgentRunRecord(
        role='memory_agent',
        display_name='Memory Agent',
        status='success',
        stage_names=[],
        completed_stage_names=[],
        detail='；'.join(detail_parts),
        attempt_count=attempt_count,
        used_retry=used_retry,
        duration_ms=duration_ms,
    )



def _summarize_stage_details(entries: list[StageTraceEntry]) -> str:
    """把多个 stage 的 detail 合并成更紧凑的 Agent 层说明。"""
    details: list[str] = []
    for entry in entries:
        if entry.detail:
            details.append(entry.detail)
    if details:
        return '；'.join(details)
    return '已完成对应阶段执行。'



def _build_analysis_shared_context(result) -> dict[str, str | int | float | bool | list[str]]:
    """汇总分析链路中各 Agent 会共享的一组上下文。"""
    profile = result.project_profile
    return {
        'repo_id': result.repo_info.repo_model.full_name,
        'project_type': result.project_type or '',
        'primary_language': profile.primary_language or '',
        'entrypoints': profile.entrypoints,
        'frameworks': profile.frameworks,
        'function_summary_count': len(profile.function_summaries),
        'class_summary_count': len(profile.class_summaries),
        'api_route_summary_count': len(profile.api_route_summaries),
    }
