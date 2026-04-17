from collections.abc import Callable

from repoinsight.analyze.stages import ANALYSIS_STAGES, build_state_from_url
from repoinsight.models.analysis_model import AnalysisRunResult, AnalysisState, StageTraceEntry



def run_analysis(url: str) -> AnalysisRunResult:
    """串联 analyze 主流程，但内部改为多 stage 编排。"""
    state = build_state_from_url(url)
    for stage in ANALYSIS_STAGES:
        state = _run_stage(state, stage)
    return _build_analysis_result(state)



def _run_stage(
    state: AnalysisState,
    stage: Callable[[AnalysisState], AnalysisState],
) -> AnalysisState:
    """统一执行单个 stage，并记录执行轨迹。"""
    stage_name = stage.__name__
    state.current_stage = stage_name

    try:
        next_state = stage(state)
    except Exception as exc:
        state.stage_trace.append(
            StageTraceEntry(
                stage_name=stage_name,
                status='failed',
                error_message=str(exc),
            )
        )
        raise

    next_state.completed_stages.append(stage_name)
    next_state.stage_trace.append(
        StageTraceEntry(
            stage_name=stage_name,
            status='success',
            detail=_build_stage_detail(stage_name, next_state),
        )
    )
    next_state.current_stage = None
    return next_state



def _build_stage_detail(stage_name: str, state: AnalysisState) -> str:
    """为 stage trace 生成简短说明，便于调试与后续接入缓存。"""
    if stage_name == 'validate_analysis_url_stage':
        return 'URL 校验通过'
    if stage_name == 'fetch_repo_metadata_stage':
        return '已获取仓库元数据'
    if stage_name == 'fetch_repo_readme_stage':
        if state.repo_info is not None and state.repo_info.readme and state.repo_info.readme.strip():
            return '已获取 README'
        return 'README 缺失或未成功获取'
    if stage_name == 'clone_repository_stage':
        if state.clone_path:
            return f'已克隆到 {state.clone_path}'
        return '已完成仓库克隆'
    if stage_name == 'scan_repository_stage':
        file_count = len(state.scan_result.all_files) if state.scan_result else 0
        key_count = len(state.scan_result.key_files) if state.scan_result else 0
        return f'扫描到 {file_count} 个文件，识别出 {key_count} 个关键文件'
    if stage_name == 'read_key_files_stage':
        return f'读取了 {len(state.key_file_contents)} 个关键文件'
    if stage_name == 'build_project_profile_stage':
        language = state.project_profile.primary_language or '未知主语言'
        return f'已生成项目画像，主语言为 {language}'
    if stage_name == 'build_tech_stack_stage':
        return f'识别出 {len(state.tech_stack)} 条技术栈信号'
    if stage_name == 'build_repo_insights_stage':
        if state.project_type:
            return f'已生成项目洞察，项目类型为 {state.project_type}'
        return '已生成项目洞察'
    return '阶段执行完成'



def _build_analysis_result(state: AnalysisState) -> AnalysisRunResult:
    """把中间状态收敛为最终对外暴露的分析结果。"""
    if state.repo_info is None:
        raise RuntimeError('分析状态缺少 repo_info')
    if state.clone_path is None:
        raise RuntimeError('分析状态缺少 clone_path')
    if state.scan_result is None:
        raise RuntimeError('分析状态缺少 scan_result')

    return AnalysisRunResult(
        repo_info=state.repo_info,
        clone_path=state.clone_path,
        scan_result=state.scan_result,
        key_file_contents=state.key_file_contents,
        tech_stack=state.tech_stack,
        project_profile=state.project_profile,
        project_type=state.project_type,
        project_type_evidence=state.project_type_evidence,
        strengths=state.strengths,
        risks=state.risks,
        observations=state.observations,
        stage_trace=state.stage_trace,
    )
