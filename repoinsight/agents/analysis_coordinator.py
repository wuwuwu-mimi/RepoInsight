from concurrent.futures import ThreadPoolExecutor

from repoinsight.agents.execution import execute_with_retry
from repoinsight.agents.models import (
    AgentEvidenceItem,
    AgentRunRecord,
    AgentStepSpec,
    AgentStructuredOutput,
    AgentTaskPacket,
    CoordinatedAnalysisResult,
)
from repoinsight.analyze.pipeline import run_analysis
from repoinsight.models.analysis_model import AnalysisRunResult, StageTraceEntry
from repoinsight.models.rag_model import IndexResult
from repoinsight.storage.index_service import index_analysis_result


DEFAULT_ANALYSIS_AGENT_SPECS: tuple[dict[str, object], ...] = (
    {
        'role': 'planner_agent',
        'display_name': 'Planner Agent',
        'description': '负责汇总分析任务顺序、依赖关系与跨 Agent 交接卡片。',
        'stage_names': [],
        'depends_on': [],
        'can_run_in_parallel': False,
    },
    {
        'role': 'repo_agent',
        'display_name': 'Repo Agent',
        'description': '负责 URL 校验与仓库基础元数据获取。',
        'stage_names': ['validate_analysis_url_stage', 'fetch_repo_metadata_stage'],
        'depends_on': ['planner_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'readme_agent',
        'display_name': 'Readme Agent',
        'description': '负责拉取 README 并补充仓库文档侧上下文。',
        'stage_names': ['fetch_repo_readme_stage'],
        'depends_on': ['repo_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'structure_agent',
        'display_name': 'Structure Agent',
        'description': '负责克隆仓库、扫描目录并读取关键文件。',
        'stage_names': ['clone_repository_stage', 'scan_repository_stage', 'read_key_files_stage'],
        'depends_on': ['readme_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'codebase_agent',
        'display_name': 'Codebase Agent',
        'description': '负责从关键文件与源码结构中提炼入口、模块、符号与项目画像。',
        'stage_names': ['build_project_profile_stage'],
        'depends_on': ['structure_agent'],
        'can_run_in_parallel': True,
    },
    {
        'role': 'profile_agent',
        'display_name': 'Profile Agent',
        'description': '负责汇总技术栈、运行时、框架与工程特征。',
        'stage_names': ['build_tech_stack_stage'],
        'depends_on': ['structure_agent'],
        'can_run_in_parallel': True,
    },
    {
        'role': 'insight_agent',
        'display_name': 'Insight Agent',
        'description': '负责生成项目类型、优势、风险与观察结论。',
        'stage_names': ['build_repo_insights_stage'],
        'depends_on': ['codebase_agent', 'profile_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'verifier_agent',
        'display_name': 'Verifier Agent',
        'description': '负责检查分析结果是否完整、是否具备可解释的关键证据。',
        'stage_names': [],
        'depends_on': ['insight_agent'],
        'can_run_in_parallel': False,
    },
    {
        'role': 'memory_agent',
        'display_name': 'Memory Agent',
        'description': '负责把分析结果写入知识文档与向量索引。',
        'stage_names': [],
        'depends_on': ['verifier_agent'],
        'can_run_in_parallel': False,
    },
)

ANALYSIS_AGENT_ROLE_ORDER: tuple[str, ...] = (
    'planner_agent',
    'repo_agent',
    'readme_agent',
    'structure_agent',
    'codebase_agent',
    'profile_agent',
    'insight_agent',
    'verifier_agent',
    'memory_agent',
)

STAGE_TO_AGENT_ROLE = {
    'validate_analysis_url_stage': 'repo_agent',
    'fetch_repo_metadata_stage': 'repo_agent',
    'fetch_repo_readme_stage': 'readme_agent',
    'clone_repository_stage': 'structure_agent',
    'scan_repository_stage': 'structure_agent',
    'read_key_files_stage': 'structure_agent',
    'build_project_profile_stage': 'codebase_agent',
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


def build_dynamic_analysis_agent_plan(
    analysis_result: AnalysisRunResult,
    *,
    include_memory_agent: bool = True,
) -> list[AgentStepSpec]:
    """基于仓库当前特征，动态裁剪一份更贴近实际的分析 Agent 计划。"""
    default_plan = build_default_analysis_agent_plan(include_memory_agent=include_memory_agent)
    step_by_role = {item.role: item for item in default_plan}
    enabled_roles = _resolve_enabled_analysis_roles(
        analysis_result,
        include_memory_agent=include_memory_agent,
    )
    dynamic_plan: list[AgentStepSpec] = []
    for role in ANALYSIS_AGENT_ROLE_ORDER:
        if role not in enabled_roles:
            continue
        step = step_by_role[role]
        depends_on = [item for item in step.depends_on if item in enabled_roles]
        if not depends_on and role != 'planner_agent':
            previous_enabled_role = _find_previous_enabled_role(role, enabled_roles)
            if previous_enabled_role is not None:
                depends_on = [previous_enabled_role]
        dynamic_plan.append(step.model_copy(update={'depends_on': depends_on}))
    return dynamic_plan


def build_analysis_task_packets(
    analysis_result: AnalysisRunResult,
    *,
    include_memory_agent: bool,
    agent_plan: list[AgentStepSpec] | None = None,
) -> dict[str, AgentTaskPacket]:
    """为分析链路中的各个 Agent 生成最小可用的任务卡片。"""
    profile = analysis_result.project_profile
    repo_model = analysis_result.repo_info.repo_model
    packets: list[AgentTaskPacket] = [
        AgentTaskPacket(
            target_role='repo_agent',
            title='确认仓库身份',
            objective='确认目标仓库的基础事实，作为后续所有分析的起点。',
            depends_on=['planner_agent'],
            required_inputs=['GitHub 仓库 URL'],
            expected_outputs=['仓库 full_name', '默认分支', 'stars / topics'],
            handoff_notes=[
                f'目标仓库：{repo_model.full_name}',
                f'默认分支优先关注：{repo_model.default_branch or "unknown"}',
            ],
        ),
        AgentTaskPacket(
            target_role='readme_agent',
            title='提炼仓库文档概览',
            objective='确认 README 是否可用，并输出最短项目概览。',
            depends_on=['repo_agent'],
            required_inputs=['仓库基础元数据', 'README 原文'],
            expected_outputs=['README 可用性', 'README 预览', 'README 长度'],
            handoff_notes=[
                '若 README 缺失，需要明确标记后续分析证据可能不足。',
            ],
        ),
        AgentTaskPacket(
            target_role='structure_agent',
            title='扫描仓库结构',
            objective='克隆仓库并输出目录结构、关键文件和子项目信息。',
            depends_on=['readme_agent'],
            required_inputs=['README 是否可用', '仓库 URL'],
            expected_outputs=['本地 clone 路径', '关键文件列表', '顶层目录 / 子项目'],
            handoff_notes=[
                f'当前关键文件数：{len(analysis_result.scan_result.key_files)}',
                f'当前候选文件数：{len(analysis_result.scan_result.all_files)}',
            ],
        ),
        AgentTaskPacket(
            target_role='codebase_agent',
            title='提炼代码结构',
            objective='从关键文件和源码结构中识别入口、符号、接口与关系边。',
            depends_on=['structure_agent'],
            required_inputs=['关键文件内容', '扫描结果', '子项目信息'],
            expected_outputs=['entrypoints', 'function/class/api 摘要', '模块 / 代码关系'],
            handoff_notes=[
                f'当前入口线索：{", ".join(profile.entrypoints[:3]) or "暂无"}',
                f'当前子项目数：{len(profile.subprojects)}',
            ],
        ),
        AgentTaskPacket(
            target_role='profile_agent',
            title='归纳技术栈画像',
            objective='聚合运行时、框架、构建工具与工程配套信号。',
            depends_on=['structure_agent'],
            required_inputs=['project_profile', 'tech_stack 结果'],
            expected_outputs=['confirmed / weak signals', '框架/运行时/工具链画像'],
            handoff_notes=[
                f'主语言候选：{profile.primary_language or "未知"}',
                f'框架候选：{", ".join(profile.frameworks[:3]) or "暂无"}',
            ],
        ),
        AgentTaskPacket(
            target_role='insight_agent',
            title='生成项目洞察',
            objective='综合结构和技术栈，输出项目类型、优势、风险与观察。',
            depends_on=['codebase_agent', 'profile_agent'],
            required_inputs=['技术栈画像', '入口与结构关系', 'README 线索'],
            expected_outputs=['project_type', 'project_type_evidence', 'strengths / risks / observations'],
            handoff_notes=[
                f'已有观察数：{len(analysis_result.observations)}',
                '优先给出可解释、可复核的结论。',
            ],
        ),
        AgentTaskPacket(
            target_role='verifier_agent',
            title='校验分析完整性',
            objective='检查关键字段、证据和项目类型推断是否完整。',
            depends_on=['insight_agent'],
            required_inputs=['完整 analysis_result', '前序 Agent 结构化输出'],
            expected_outputs=['verdict', 'issue_count', '待关注项'],
            handoff_notes=['若关键字段缺失，需要明确阻止后续入库。'],
        ),
    ]
    if include_memory_agent:
        packets.append(
            AgentTaskPacket(
                target_role='memory_agent',
                title='写入知识库',
                objective='把分析结果写入本地知识文档和向量索引。',
                depends_on=['verifier_agent'],
                required_inputs=['analysis_result', 'verification verdict'],
                expected_outputs=['知识文件路径', '向量索引状态'],
                handoff_notes=['仅在需要持久化知识时执行。'],
            )
        )
    packet_map = {item.target_role: item for item in packets}
    if agent_plan is None:
        return packet_map
    plan_depends_on = {item.role: item.depends_on for item in agent_plan}
    enabled_roles = {item.role for item in agent_plan}
    return {
        role: packet.model_copy(update={'depends_on': plan_depends_on.get(role, packet.depends_on)})
        for role, packet in packet_map.items()
        if role in enabled_roles
    }


def build_agent_trace_from_stage_trace(
    stage_trace: list[StageTraceEntry],
    *,
    include_memory_agent: bool = False,
    analysis_result: AnalysisRunResult | None = None,
    task_packets_by_role: dict[str, AgentTaskPacket] | None = None,
    agent_plan: list[AgentStepSpec] | None = None,
) -> list[AgentRunRecord]:
    """把 stage trace 聚合成更高层的 Agent 执行记录。"""
    plan = agent_plan or build_default_analysis_agent_plan(include_memory_agent=include_memory_agent)
    return _build_analysis_agent_records(
        plan,
        stage_trace,
        analysis_result=analysis_result,
        task_packets_by_role=task_packets_by_role,
    )


def run_multi_agent_analysis(
    url: str,
    *,
    persist_knowledge: bool = False,
) -> CoordinatedAnalysisResult:
    """在不改动现有 pipeline 的前提下，提供一层多 Agent 视角的分析编排。"""
    analysis_result = run_analysis(url)
    agent_plan = build_dynamic_analysis_agent_plan(
        analysis_result,
        include_memory_agent=persist_knowledge,
    )
    task_packets_by_role = build_analysis_task_packets(
        analysis_result,
        include_memory_agent=persist_knowledge,
        agent_plan=agent_plan,
    )
    planner_record = _build_planner_agent_record(agent_plan, task_packets_by_role)
    agent_trace = build_agent_trace_from_stage_trace(
        analysis_result.stage_trace,
        include_memory_agent=False,
        analysis_result=analysis_result,
        task_packets_by_role=task_packets_by_role,
        agent_plan=agent_plan,
    )
    agent_trace.insert(0, planner_record)

    verification_record = _build_verifier_agent_record(analysis_result)
    agent_trace.append(verification_record)

    shared_context = _build_analysis_shared_context(analysis_result)
    shared_context.update(
        {
            'planner_task_count': len(task_packets_by_role),
            'planner_active_roles': [item.role for item in agent_plan],
            'planner_skipped_roles': _collect_skipped_roles(agent_plan),
            'planner_parallel_roles': _collect_parallel_roles(agent_plan),
            'planner_parallel_groups': [' + '.join(item) for item in _collect_parallel_groups(agent_plan)],
            'agent_execution_mode': 'local_parallel',
            'planner_handoff_order': _collect_handoff_order(task_packets_by_role),
            'analysis_verification_verdict': verification_record.structured_output.metadata.get('verdict')
            if verification_record.structured_output is not None
            else 'unknown',
            'analysis_verification_issue_count': len(
                verification_record.structured_output.uncertainties
                if verification_record.structured_output is not None
                else []
            ),
        }
    )

    index_result: IndexResult | None = None
    if persist_knowledge:
        memory_outcome = execute_with_retry(index_analysis_result, analysis_result, retries=2)
        if memory_outcome.error is not None:
            raise memory_outcome.error
        index_result = memory_outcome.value
        agent_trace.append(
            _build_memory_agent_record(
                index_result,
                memory_outcome.attempt_count,
                memory_outcome.used_retry,
                memory_outcome.duration_ms,
            )
        )
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
    *,
    analysis_result: AnalysisRunResult | None = None,
    task_packet: AgentTaskPacket | None = None,
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
    status = 'failed' if failed_entries else 'success'
    error_message = failed_entries[0].error_message if failed_entries else None
    detail = _summarize_stage_details(related_entries)

    record = AgentRunRecord(
        role=step.role,
        display_name=step.display_name,
        status=status,
        stage_names=step.stage_names,
        completed_stage_names=completed_stage_names,
        detail=detail,
        error_message=error_message,
    )
    if analysis_result is None:
        return record
    structured_output = _build_analysis_agent_structured_output(step.role, analysis_result)
    if task_packet is not None:
        structured_output = _merge_task_packet_into_output(structured_output, task_packet)
    return record.model_copy(update={'structured_output': structured_output})


def _build_analysis_agent_records(
    agent_plan: list[AgentStepSpec],
    stage_trace: list[StageTraceEntry],
    *,
    analysis_result: AnalysisRunResult | None = None,
    task_packets_by_role: dict[str, AgentTaskPacket] | None = None,
) -> list[AgentRunRecord]:
    """按执行波次构建分析 Agent 记录；同波次角色可并行生成结构化输出。"""
    waves = _build_analysis_execution_waves(agent_plan)
    record_by_role: dict[str, AgentRunRecord] = {}
    for wave in waves:
        worker_steps = [
            step for step in wave
            if step.role not in {'planner_agent', 'verifier_agent', 'memory_agent'}
        ]
        if not worker_steps:
            continue
        if len(worker_steps) == 1:
            step = worker_steps[0]
            record_by_role[step.role] = _build_analysis_record_for_step(
                step,
                stage_trace,
                analysis_result=analysis_result,
                task_packets_by_role=task_packets_by_role,
            )
            continue
        max_workers = min(len(worker_steps), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                step.role: executor.submit(
                    _build_analysis_record_for_step,
                    step,
                    stage_trace,
                    analysis_result=analysis_result,
                    task_packets_by_role=task_packets_by_role,
                )
                for step in worker_steps
            }
            for role, future in future_map.items():
                record_by_role[role] = future.result()

    ordered_roles = [item.role for item in agent_plan if item.role in record_by_role]
    return [record_by_role[role] for role in ordered_roles]


def _build_analysis_record_for_step(
    step: AgentStepSpec,
    stage_trace: list[StageTraceEntry],
    *,
    analysis_result: AnalysisRunResult | None = None,
    task_packets_by_role: dict[str, AgentTaskPacket] | None = None,
) -> AgentRunRecord:
    """为单个分析角色生成执行记录。"""
    related_entries = [
        entry for entry in stage_trace if STAGE_TO_AGENT_ROLE.get(entry.stage_name) == step.role
    ]
    return _build_agent_run_record(
        step,
        related_entries,
        analysis_result=analysis_result,
        task_packet=task_packets_by_role.get(step.role) if task_packets_by_role is not None else None,
    )


def _build_planner_agent_record(
    agent_plan: list[AgentStepSpec],
    task_packets_by_role: dict[str, AgentTaskPacket],
) -> AgentRunRecord:
    """构建 planner_agent 的执行记录，用于显式展示任务分发结果。"""
    packet_list = [task_packets_by_role[step.role] for step in agent_plan if step.role in task_packets_by_role]
    parallel_roles = _collect_parallel_roles(agent_plan)
    parallel_groups = _collect_parallel_groups(agent_plan)
    active_roles = [step.role for step in agent_plan]
    skipped_roles = _collect_skipped_roles(agent_plan)
    return AgentRunRecord(
        role='planner_agent',
        display_name='Planner Agent',
        status='success',
        stage_names=['plan_analysis_agents'],
        completed_stage_names=['plan_analysis_agents'],
        detail=f'已生成 {len(packet_list)} 个分析任务卡片，并明确 {len(parallel_roles)} 个可并行角色。',
        structured_output=AgentStructuredOutput(
            conclusions=[
                f'已为分析链路规划 {len(packet_list)} 个下游任务。',
                f'执行顺序为：{", ".join(packet.target_role for packet in packet_list)}。',
            ],
            evidence=[
                AgentEvidenceItem(
                    kind='task_packet',
                    label=f'{packet.target_role}: {packet.title}',
                    snippet=packet.objective,
                )
                for packet in packet_list[:6]
            ],
            uncertainties=[],
            next_actions=[
                f'先交给 {packet_list[0].target_role} 开始执行。'
                if packet_list
                else '等待下游 Agent 执行。'
            ],
            metadata={
                'task_count': len(packet_list),
                'active_roles': active_roles,
                'skipped_roles': skipped_roles,
                'parallel_roles': parallel_roles,
                'parallel_groups': [' + '.join(item) for item in parallel_groups],
                'task_order': [packet.target_role for packet in packet_list[:8]],
            },
        ),
    )


def _merge_task_packet_into_output(
    output: AgentStructuredOutput,
    task_packet: AgentTaskPacket,
) -> AgentStructuredOutput:
    """把 planner_agent 下发的任务卡片补充进下游 Agent 的结构化输出。"""
    metadata = dict(output.metadata)
    metadata.update(
        {
            'task_title': task_packet.title,
            'depends_on': task_packet.depends_on,
            'required_inputs': task_packet.required_inputs[:4],
            'expected_outputs': task_packet.expected_outputs[:4],
        }
    )
    next_actions = list(output.next_actions)
    for item in task_packet.handoff_notes[:2]:
        note = f'交接提示：{item}'
        if note not in next_actions:
            next_actions.append(note)
    return output.model_copy(update={'metadata': metadata, 'next_actions': next_actions})


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
        structured_output=AgentStructuredOutput(
            conclusions=[
                f'知识文档已写入 {index_result.local_path}。',
                (
                    f'向量索引已写入 {index_result.vector_backend}。'
                    if index_result.vector_indexed
                    else '当前仅保存本地知识文档，未写入向量索引。'
                ),
            ],
            evidence=[
                AgentEvidenceItem(kind='path', label=index_result.local_path, source_path=index_result.local_path),
            ],
            uncertainties=[] if index_result.vector_indexed else ['当前未写入向量索引，后续检索仅依赖本地文档。'],
            next_actions=['分析结果已可供 search / answer / RAG 链路复用。'],
            metadata={
                'vector_indexed': index_result.vector_indexed,
                'vector_backend': index_result.vector_backend,
                'attempt_count': attempt_count,
            },
        ),
    )


def _build_verifier_agent_record(analysis_result: AnalysisRunResult) -> AgentRunRecord:
    """对分析结果做一轮轻量完整性校验。"""
    issues: list[str] = []
    notes: list[str] = []
    if not analysis_result.repo_info.repo_model.full_name:
        issues.append('缺少 repo_id。')
    if not analysis_result.clone_path:
        issues.append('缺少本地克隆路径。')
    if not analysis_result.scan_result.all_files:
        issues.append('扫描结果为空。')
    if not analysis_result.project_profile.primary_language:
        issues.append('主语言尚未识别。')
    if not analysis_result.project_type:
        issues.append('项目类型尚未生成。')
    if not analysis_result.project_type_evidence:
        issues.append('项目类型依据缺失。')

    if analysis_result.project_profile.entrypoints:
        notes.append(f"已识别 {len(analysis_result.project_profile.entrypoints)} 个入口线索。")
    if analysis_result.project_profile.frameworks:
        notes.append(f"已识别框架：{', '.join(analysis_result.project_profile.frameworks[:3])}。")
    if analysis_result.tech_stack:
        notes.append(f'已识别 {len(analysis_result.tech_stack)} 条技术栈信号。')

    completeness_score = 6 - len(issues)
    verdict = 'passed' if not issues else ('warning' if completeness_score >= 4 else 'failed')
    detail = f'分析完整性检查结果为 {verdict}'
    if issues:
        detail += f"；待关注：{'；'.join(issues[:2])}"

    return AgentRunRecord(
        role='verifier_agent',
        display_name='Verifier Agent',
        status='success',
        stage_names=['verify_analysis_completeness'],
        completed_stage_names=['verify_analysis_completeness'],
        detail=detail,
        structured_output=AgentStructuredOutput(
            conclusions=[f'分析完整性检查结果为 {verdict}。'],
            evidence=[
                AgentEvidenceItem(kind='repo', label=analysis_result.repo_info.repo_model.full_name),
                AgentEvidenceItem(kind='path', label=analysis_result.clone_path, source_path=analysis_result.clone_path),
            ],
            uncertainties=issues,
            next_actions=(
                ['当前分析结果可以继续进入 memory_agent。']
                if verdict != 'failed'
                else ['建议先补全关键字段，再决定是否入库。']
            ),
            metadata={
                'verdict': verdict,
                'issue_count': len(issues),
                'has_project_type': bool(analysis_result.project_type),
                'has_primary_language': bool(analysis_result.project_profile.primary_language),
            },
        ),
    )


def _build_analysis_agent_structured_output(
    role: str,
    analysis_result: AnalysisRunResult,
) -> AgentStructuredOutput:
    """为分析侧 Agent 构建统一结构化输出。"""
    repo_model = analysis_result.repo_info.repo_model
    profile = analysis_result.project_profile
    if role == 'repo_agent':
        conclusions = [f'已确认目标仓库为 {repo_model.full_name}。']
        if repo_model.description:
            conclusions.append(f'仓库描述为：{repo_model.description}')
        return AgentStructuredOutput(
            conclusions=conclusions,
            evidence=[
                AgentEvidenceItem(kind='repo', label=repo_model.full_name),
                AgentEvidenceItem(kind='branch', label=repo_model.default_branch or 'unknown'),
            ],
            uncertainties=[],
            next_actions=['交给 readme_agent 拉取 README。'],
            metadata={
                'default_branch': repo_model.default_branch or '',
                'stars': repo_model.stargazers_count,
                'topics': repo_model.topics,
            },
        )
    if role == 'readme_agent':
        readme_text = (analysis_result.repo_info.readme or '').strip()
        readme_preview = readme_text[:160] if readme_text else None
        return AgentStructuredOutput(
            conclusions=[
                'README 已获取，可作为项目概览与项目类型判断的补充依据。'
                if readme_text
                else 'README 缺失或未成功获取。'
            ],
            evidence=(
                [
                    AgentEvidenceItem(
                        kind='readme',
                        label='README',
                        source_path='README.md',
                        snippet=readme_preview,
                    )
                ]
                if readme_text
                else []
            ),
            uncertainties=[] if readme_text else ['README 缺失或读取失败。'],
            next_actions=['交给 structure_agent 下载并扫描仓库结构。'],
            metadata={
                'readme_available': bool(readme_text),
                'readme_length': len(readme_text),
            },
        )
    if role == 'structure_agent':
        key_files = [item.path for item in analysis_result.key_file_contents[:5]]
        top_level_dirs = _collect_top_level_directories(analysis_result)
        subproject_labels = _collect_subproject_labels(analysis_result)
        evidence_items = [
            AgentEvidenceItem(kind='path', label=analysis_result.clone_path, source_path=analysis_result.clone_path),
            *[
                AgentEvidenceItem(kind='key_file', label=path, source_path=path)
                for path in key_files
            ],
        ]
        if analysis_result.scan_result.tree_preview:
            tree_preview = ', '.join(analysis_result.scan_result.tree_preview[:4])
            evidence_items.append(
                AgentEvidenceItem(
                    kind='tree_preview',
                    label=f'目录预览：{tree_preview}',
                    snippet=tree_preview,
                )
            )
        evidence_items.extend(
            AgentEvidenceItem(kind='subproject', label=item)
            for item in subproject_labels[:2]
        )
        conclusions = [
            f'仓库已克隆到 {analysis_result.clone_path}。',
            (
                f'当前共扫描到 {len(analysis_result.scan_result.all_files)} 个文件，'
                f'识别出 {len(analysis_result.scan_result.key_files)} 个关键文件，'
                f'忽略了 {len(analysis_result.scan_result.ignored_entries)} 个路径。'
            ),
        ]
        if subproject_labels:
            conclusions.append(f'检测到 {len(profile.subprojects)} 个子项目/工作区：{", ".join(subproject_labels[:2])}。')
        elif top_level_dirs:
            conclusions.append(f'顶层目录重点包括：{", ".join(top_level_dirs[:4])}。')

        uncertainties: list[str] = []
        if not analysis_result.scan_result.all_files:
            uncertainties.append('扫描结果为空，后续画像可能不完整。')
        if analysis_result.scan_result.all_files and not analysis_result.scan_result.key_files:
            uncertainties.append('暂未识别出关键文件，代码画像的证据密度可能偏低。')
        return AgentStructuredOutput(
            conclusions=conclusions,
            evidence=evidence_items,
            uncertainties=uncertainties,
            next_actions=['交给 codebase_agent 提炼入口、模块关系与代码画像。'],
            metadata={
                'file_count': len(analysis_result.scan_result.all_files),
                'key_file_count': len(analysis_result.scan_result.key_files),
                'ignored_count': len(analysis_result.scan_result.ignored_entries),
                'top_level_dirs': top_level_dirs[:5],
                'subproject_count': len(profile.subprojects),
                'key_files': key_files,
            },
        )
    if role == 'codebase_agent':
        evidence_items = [
            *[
                AgentEvidenceItem(kind='entrypoint', label=item, source_path=item)
                for item in profile.entrypoints[:3]
            ],
            *[
                AgentEvidenceItem(
                    kind='api_route',
                    label=_format_api_route_label(item),
                    source_path=item.source_path,
                    location=_format_line_location(item.line_number),
                )
                for item in profile.api_route_summaries[:2]
            ],
            *[
                AgentEvidenceItem(
                    kind='function_summary',
                    label=item.qualified_name,
                    source_path=item.source_path,
                    location=_format_line_span(item.line_start, item.line_end),
                )
                for item in profile.function_summaries[:2]
            ],
            *[
                AgentEvidenceItem(
                    kind='class_summary',
                    label=item.qualified_name,
                    source_path=item.source_path,
                    location=_format_line_span(item.line_start, item.line_end),
                )
                for item in profile.class_summaries[:1]
            ],
            *[
                AgentEvidenceItem(
                    kind='module_relation',
                    label=f'{item.source_path} -> {item.target}',
                    source_path=item.source_path,
                    location=_format_line_location(item.line_number),
                )
                for item in profile.module_relations[:2]
            ],
            *[
                AgentEvidenceItem(
                    kind='code_relation',
                    label=f'{item.source_ref} -[{item.relation_type}]-> {item.target_ref}',
                    source_path=item.source_path,
                    location=_format_line_location(item.line_number),
                )
                for item in profile.code_relation_edges[:2]
            ],
        ]
        conclusions = [
            f"主语言为 {profile.primary_language or '未知'}。",
            (
                f"已识别 {len(profile.entrypoints)} 个入口、{len(profile.code_entities)} 个统一代码实体、"
                f"{len(profile.code_relation_edges)} 条代码关系。"
            ),
        ]
        if profile.function_summaries or profile.class_summaries or profile.api_route_summaries:
            conclusions.append(
                f"已提炼 {len(profile.function_summaries)} 个函数摘要、"
                f"{len(profile.class_summaries)} 个类摘要、"
                f"{len(profile.api_route_summaries)} 个接口摘要。"
            )

        uncertainties: list[str] = []
        if not profile.entrypoints and not profile.code_entities:
            uncertainties.append('暂未识别出明显入口或结构化代码实体。')
        if not profile.module_relations and not profile.code_relation_edges:
            uncertainties.append('尚未形成稳定的模块依赖或调用关系链。')
        return AgentStructuredOutput(
            conclusions=conclusions,
            evidence=evidence_items,
            uncertainties=uncertainties,
            next_actions=['交给 profile_agent 归纳技术栈与工程特征。'],
            metadata={
                'primary_language': profile.primary_language or '',
                'entrypoint_count': len(profile.entrypoints),
                'entity_count': len(profile.code_entities),
                'relation_count': len(profile.code_relation_edges),
                'function_summary_count': len(profile.function_summaries),
                'class_summary_count': len(profile.class_summaries),
                'api_route_count': len(profile.api_route_summaries),
                'entrypoints': profile.entrypoints[:5],
            },
        )
    if role == 'profile_agent':
        confirmed_signals = profile.confirmed_signals or [
            item for item in analysis_result.tech_stack if item.evidence_level in {'strong', 'medium'}
        ]
        weak_signals = profile.weak_signals or [
            item for item in analysis_result.tech_stack if item.evidence_level == 'weak'
        ]
        evidence_items = [
            *[
                AgentEvidenceItem(kind='framework', label=item)
                for item in profile.frameworks[:3]
            ],
            *[
                AgentEvidenceItem(kind='runtime', label=item)
                for item in profile.runtimes[:2]
            ],
            *[
                AgentEvidenceItem(kind='build_tool', label=item)
                for item in profile.build_tools[:2]
            ],
            *[
                AgentEvidenceItem(kind='package_manager', label=item)
                for item in profile.package_managers[:2]
            ],
            *[
                AgentEvidenceItem(kind='test_tool', label=item)
                for item in profile.test_tools[:2]
            ],
            *[
                AgentEvidenceItem(kind='deploy_tool', label=item)
                for item in profile.deploy_tools[:2]
            ],
            *[
                AgentEvidenceItem(
                    kind='tech_stack_signal',
                    label=_format_tech_signal_label(item),
                    source_path=item.source_path,
                    snippet=item.evidence,
                )
                for item in confirmed_signals[:3]
            ],
            *[
                AgentEvidenceItem(
                    kind='weak_signal',
                    label=_format_tech_signal_label(item),
                    source_path=item.source_path,
                    snippet=item.evidence,
                )
                for item in weak_signals[:2]
            ],
        ]
        conclusions = [
            f"已识别运行时/框架/构建工具信号，共 {len(analysis_result.tech_stack)} 条技术栈结果。",
            (
                f"确认信号 {len(confirmed_signals)} 条，弱信号 {len(weak_signals)} 条；"
                f"框架候选：{', '.join(profile.frameworks[:3]) or '未识别'}。"
            ),
        ]
        if profile.package_managers or profile.test_tools or profile.deploy_tools:
            conclusions.append(
                f"工程化配套包括包管理 {', '.join(profile.package_managers[:2]) or '无'}、"
                f"测试工具 {', '.join(profile.test_tools[:2]) or '无'}、"
                f"部署工具 {', '.join(profile.deploy_tools[:2]) or '无'}。"
            )
        return AgentStructuredOutput(
            conclusions=conclusions,
            evidence=evidence_items,
            uncertainties=(
                []
                if confirmed_signals
                else ['当前技术栈信号较弱，后续洞察可能偏保守。']
            ) + (
                ['当前仅有弱证据候选信号，建议后续结合更多配置文件复核。']
                if weak_signals and not confirmed_signals
                else []
            ),
            next_actions=['交给 insight_agent 生成项目类型、优势与风险。'],
            metadata={
                'frameworks': profile.frameworks[:5],
                'runtimes': profile.runtimes[:5],
                'build_tools': profile.build_tools[:5],
                'package_managers': profile.package_managers[:5],
                'test_tools': profile.test_tools[:5],
                'deploy_tools': profile.deploy_tools[:5],
                'tech_stack_count': len(analysis_result.tech_stack),
                'confirmed_signal_count': len(confirmed_signals),
                'weak_signal_count': len(weak_signals),
            },
        )
    if role == 'insight_agent':
        evidence_items = []
        if analysis_result.project_type_evidence:
            evidence_items.append(
                AgentEvidenceItem(
                    kind='project_type_evidence',
                    label='项目类型判断依据',
                    snippet=analysis_result.project_type_evidence[:200],
                )
            )
        evidence_items.extend(
            AgentEvidenceItem(kind='observation', label=item)
            for item in analysis_result.observations[:2]
        )
        evidence_items.extend(
            AgentEvidenceItem(kind='strength', label=item)
            for item in analysis_result.strengths[:2]
        )
        evidence_items.extend(
            AgentEvidenceItem(kind='risk', label=item)
            for item in analysis_result.risks[:2]
        )
        conclusions = [
            f"项目类型推断为 {analysis_result.project_type or '未知'}。",
            *analysis_result.observations[:2],
        ]
        if analysis_result.strengths:
            conclusions.append(f"当前优势重点：{analysis_result.strengths[0]}")
        elif analysis_result.risks:
            conclusions.append(f"当前最需关注：{analysis_result.risks[0]}")
        return AgentStructuredOutput(
            conclusions=conclusions,
            evidence=evidence_items,
            uncertainties=(
                []
                if analysis_result.project_type_evidence
                else ['项目类型缺少明确依据。']
            ) + (
                ['当前优势与风险均为空，洞察内容可能还不够充分。']
                if not analysis_result.strengths and not analysis_result.risks
                else []
            ),
            next_actions=['交给 verifier_agent 检查分析结果是否完整。'],
            metadata={
                'project_type': analysis_result.project_type or '',
                'has_project_type_evidence': bool(analysis_result.project_type_evidence),
                'observation_count': len(analysis_result.observations),
                'strength_count': len(analysis_result.strengths),
                'risk_count': len(analysis_result.risks),
            },
        )
    return AgentStructuredOutput()


def _summarize_stage_details(entries: list[StageTraceEntry]) -> str:
    """把多个 stage 的 detail 合并成更紧凑的 Agent 层说明。"""
    details: list[str] = []
    for entry in entries:
        if entry.detail:
            details.append(entry.detail)
    if details:
        return '；'.join(details)
    return '已完成对应阶段执行。'


def _build_analysis_shared_context(result: AnalysisRunResult) -> dict[str, str | int | float | bool | list[str]]:
    """汇总分析链路中各 Agent 会共享的一组上下文。"""
    profile = result.project_profile
    return {
        'repo_id': result.repo_info.repo_model.full_name,
        'project_type': result.project_type or '',
        'primary_language': profile.primary_language or '',
        'readme_available': bool(result.repo_info.readme and result.repo_info.readme.strip()),
        'entrypoints': profile.entrypoints,
        'frameworks': profile.frameworks,
        'function_summary_count': len(profile.function_summaries),
        'class_summary_count': len(profile.class_summaries),
        'api_route_summary_count': len(profile.api_route_summaries),
        'module_relation_count': len(profile.module_relations),
        'code_entity_count': len(profile.code_entities),
        'tech_stack_count': len(result.tech_stack),
    }


def _collect_parallel_roles(agent_plan: list[AgentStepSpec]) -> list[str]:
    """提取计划中标记为可并行执行的角色列表。"""
    return [item.role for item in agent_plan if item.can_run_in_parallel]


def _build_analysis_execution_waves(agent_plan: list[AgentStepSpec]) -> list[list[AgentStepSpec]]:
    """根据依赖关系把分析计划切分为可并行执行的波次。"""
    step_by_role = {item.role: item for item in agent_plan}
    remaining_roles = [item.role for item in agent_plan]
    completed_roles: set[str] = set()
    waves: list[list[AgentStepSpec]] = []

    while remaining_roles:
        ready_roles = [
            role
            for role in remaining_roles
            if all(dep in completed_roles for dep in step_by_role[role].depends_on)
        ]
        if not ready_roles:
            raise RuntimeError('分析 Agent 计划存在循环依赖，无法生成执行波次。')
        wave = [step_by_role[role] for role in ready_roles]
        waves.append(wave)
        completed_roles.update(ready_roles)
        remaining_roles = [role for role in remaining_roles if role not in completed_roles]

    return waves


def _collect_parallel_groups(agent_plan: list[AgentStepSpec]) -> list[list[str]]:
    """提取计划中的并行执行组，便于 planner 与 UI 展示。"""
    return [
        [step.role for step in wave if step.can_run_in_parallel]
        for wave in _build_analysis_execution_waves(agent_plan)
        if len([step for step in wave if step.can_run_in_parallel]) > 1
    ]


def _collect_skipped_roles(agent_plan: list[AgentStepSpec]) -> list[str]:
    """基于默认角色顺序，计算当前动态计划跳过了哪些角色。"""
    active_roles = {item.role for item in agent_plan}
    return [
        role
        for role in ANALYSIS_AGENT_ROLE_ORDER
        if role not in active_roles
    ]


def _resolve_enabled_analysis_roles(
    analysis_result: AnalysisRunResult,
    *,
    include_memory_agent: bool,
) -> set[str]:
    """根据当前仓库的分析信号，决定哪些分析 Agent 需要参与执行。"""
    profile = analysis_result.project_profile
    has_code_signals = any(
        [
            bool(profile.primary_language),
            bool(profile.entrypoints),
            bool(profile.code_entities),
            bool(profile.code_relation_edges),
            bool(profile.module_relations),
            bool(profile.function_summaries),
            bool(profile.class_summaries),
            bool(profile.api_route_summaries),
            bool(profile.subprojects),
        ]
    )
    has_profile_signals = any(
        [
            bool(analysis_result.tech_stack),
            bool(profile.frameworks),
            bool(profile.runtimes),
            bool(profile.build_tools),
            bool(profile.package_managers),
            bool(profile.test_tools),
            bool(profile.deploy_tools),
            bool(profile.confirmed_signals),
            bool(profile.weak_signals),
            bool(profile.primary_language),
        ]
    )
    enabled_roles = {
        'planner_agent',
        'repo_agent',
        'readme_agent',
        'structure_agent',
        'insight_agent',
        'verifier_agent',
    }
    if has_code_signals:
        enabled_roles.add('codebase_agent')
    if has_profile_signals:
        enabled_roles.add('profile_agent')
    elif 'codebase_agent' in enabled_roles:
        # 即使技术栈信号较弱，只要已进入代码结构分析，也保留画像阶段做兜底归纳。
        enabled_roles.add('profile_agent')
    if include_memory_agent:
        enabled_roles.add('memory_agent')
    return enabled_roles


def _find_previous_enabled_role(role: str, enabled_roles: set[str]) -> str | None:
    """在默认角色顺序中找到当前角色之前最近的启用角色。"""
    role_index = ANALYSIS_AGENT_ROLE_ORDER.index(role)
    for candidate in reversed(ANALYSIS_AGENT_ROLE_ORDER[:role_index]):
        if candidate in enabled_roles:
            return candidate
    return None


def _collect_handoff_order(task_packets_by_role: dict[str, AgentTaskPacket]) -> list[str]:
    """把任务卡片压缩为适合 shared_context 展示的交接链。"""
    ordered_roles = [
        'repo_agent',
        'readme_agent',
        'structure_agent',
        'codebase_agent',
        'profile_agent',
        'insight_agent',
        'verifier_agent',
        'memory_agent',
    ]
    items: list[str] = []
    for role in ordered_roles:
        packet = task_packets_by_role.get(role)
        if packet is None:
            continue
        deps = ','.join(packet.depends_on) if packet.depends_on else 'root'
        items.append(f'{role}<-{deps}')
    return items


def _collect_top_level_directories(analysis_result: AnalysisRunResult) -> list[str]:
    """从扫描结果中提取顶层目录，便于结构 Agent 共享整体布局线索。"""
    directories: list[str] = []
    seen: set[str] = set()
    for item in analysis_result.scan_result.all_files:
        parent_dir = item.parent_dir.strip('/')
        if not parent_dir:
            continue
        top_level_dir = parent_dir.split('/', 1)[0]
        if top_level_dir and top_level_dir not in seen:
            seen.add(top_level_dir)
            directories.append(top_level_dir)
    return directories


def _collect_subproject_labels(analysis_result: AnalysisRunResult) -> list[str]:
    """把子项目摘要压缩成适合结构化展示的一行标签。"""
    labels: list[str] = []
    for item in analysis_result.project_profile.subprojects:
        project_kind = item.project_kind or 'subproject'
        language_scope = item.language_scope or 'unknown'
        labels.append(f'{item.root_path} ({project_kind}, {language_scope})')
    return labels


def _format_line_location(line_number: int | None) -> str | None:
    """把单行号格式化为统一位置文本。"""
    if line_number is None or line_number <= 0:
        return None
    return f'L{line_number}'


def _format_line_span(line_start: int | None, line_end: int | None) -> str | None:
    """把起止行号压缩成适合证据展示的位置。"""
    if line_start is None or line_start <= 0:
        return None
    if line_end is None or line_end <= 0 or line_end == line_start:
        return f'L{line_start}'
    return f'L{line_start}-L{line_end}'


def _format_api_route_label(route_summary) -> str:
    """把接口摘要压缩为更易读的单行标签。"""
    methods = ','.join(route_summary.http_methods) if route_summary.http_methods else 'ROUTE'
    return f'{methods} {route_summary.route_path} -> {route_summary.handler_name}'


def _format_tech_signal_label(signal) -> str:
    """把技术栈信号压缩成适合 CLI / UI 展示的一行证据。"""
    parts = [signal.name]
    if signal.category:
        parts.append(signal.category)
    if signal.evidence_level:
        parts.append(signal.evidence_level)
    return ' | '.join(parts)
