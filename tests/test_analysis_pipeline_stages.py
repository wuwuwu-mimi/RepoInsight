from repoinsight.analyze.pipeline import run_analysis
from repoinsight.models.analysis_model import AnalysisState, KeyFileContent, ProjectProfile
from repoinsight.models.file_model import FileEntry, ScanResult, ScanStats
from repoinsight.models.repo_model import RepoInfo, RepoModel
import repoinsight.analyze.pipeline as pipeline_module



def _build_state(url: str) -> AnalysisState:
    return AnalysisState(url=url)



def _build_repo_info() -> RepoInfo:
    return RepoInfo(
        repo_model=RepoModel(
            owner='demo',
            name='sample',
            full_name='demo/sample',
            html_url='https://github.com/demo/sample',
            default_branch='main',
        ),
        readme='demo repo',
    )



def _build_scan_result() -> ScanResult:
    entry = FileEntry(
        path='app.py',
        name='app.py',
        extension='.py',
        size_bytes=128,
        parent_dir='',
        is_key_file=True,
    )
    return ScanResult(
        root_path='E:/PythonProject/RepoInsight/clone/demo/sample',
        all_files=[entry],
        key_files=[entry],
        tree_preview=['app.py'],
        stats=ScanStats(total_seen=1, kept_count=1, key_file_count=1),
    )



def test_run_analysis_executes_stages_in_order_and_builds_result() -> None:
    original_build_state_from_url = pipeline_module.build_state_from_url
    original_analysis_stages = pipeline_module.ANALYSIS_STAGES
    calls: list[str] = []

    def stage_validate(state: AnalysisState) -> AnalysisState:
        calls.append('validate')
        assert state.url == 'https://github.com/demo/sample'
        return state

    def stage_metadata(state: AnalysisState) -> AnalysisState:
        calls.append('metadata')
        state.repo_info = _build_repo_info()
        return state

    def stage_clone(state: AnalysisState) -> AnalysisState:
        calls.append('clone')
        state.clone_path = 'E:/PythonProject/RepoInsight/clone/demo/sample'
        return state

    def stage_scan(state: AnalysisState) -> AnalysisState:
        calls.append('scan')
        state.scan_result = _build_scan_result()
        return state

    def stage_read(state: AnalysisState) -> AnalysisState:
        calls.append('read')
        state.key_file_contents = [KeyFileContent(path='app.py', size_bytes=128, content='print(1)')]
        return state

    def stage_profile(state: AnalysisState) -> AnalysisState:
        calls.append('profile')
        state.project_profile = ProjectProfile(primary_language='Python', languages=['Python'])
        return state

    def stage_stack(state: AnalysisState) -> AnalysisState:
        calls.append('stack')
        state.tech_stack = []
        return state

    def stage_insight(state: AnalysisState) -> AnalysisState:
        calls.append('insight')
        state.project_type = 'CLI 工具'
        state.strengths = ['结构清晰']
        state.risks = ['暂无']
        state.observations = ['可继续扩展']
        return state

    try:
        pipeline_module.build_state_from_url = _build_state
        pipeline_module.ANALYSIS_STAGES = (
            stage_validate,
            stage_metadata,
            stage_clone,
            stage_scan,
            stage_read,
            stage_profile,
            stage_stack,
            stage_insight,
        )

        result = run_analysis('https://github.com/demo/sample')
    finally:
        pipeline_module.build_state_from_url = original_build_state_from_url
        pipeline_module.ANALYSIS_STAGES = original_analysis_stages

    assert calls == ['validate', 'metadata', 'clone', 'scan', 'read', 'profile', 'stack', 'insight']
    assert result.repo_info.repo_model.full_name == 'demo/sample'
    assert result.clone_path.endswith('demo/sample')
    assert result.project_profile.primary_language == 'Python'
    assert result.project_type == 'CLI 工具'
    assert result.strengths == ['结构清晰']
    assert [entry.stage_name for entry in result.stage_trace] == [
        'stage_validate',
        'stage_metadata',
        'stage_clone',
        'stage_scan',
        'stage_read',
        'stage_profile',
        'stage_stack',
        'stage_insight',
    ]
    assert all(entry.status == 'success' for entry in result.stage_trace)
    assert result.stage_trace[0].detail == '阶段执行完成'
    assert result.stage_trace[-1].detail == '阶段执行完成'



def test_run_analysis_raises_when_required_state_is_missing() -> None:
    original_build_state_from_url = pipeline_module.build_state_from_url
    original_analysis_stages = pipeline_module.ANALYSIS_STAGES
    try:
        pipeline_module.build_state_from_url = _build_state
        pipeline_module.ANALYSIS_STAGES = (lambda state: state,)

        try:
            run_analysis('https://github.com/demo/sample')
        except RuntimeError as exc:
            assert 'repo_info' in str(exc)
        else:
            raise AssertionError('expected RuntimeError for missing repo_info')
    finally:
        pipeline_module.build_state_from_url = original_build_state_from_url
        pipeline_module.ANALYSIS_STAGES = original_analysis_stages



def test_run_stage_records_failed_trace_and_keeps_current_stage() -> None:
    state = AnalysisState(url='https://github.com/demo/sample')

    def failing_stage(current_state: AnalysisState) -> AnalysisState:
        raise RuntimeError('boom')

    try:
        pipeline_module._run_stage(state, failing_stage)
    except RuntimeError as exc:
        assert str(exc) == 'boom'
    else:
        raise AssertionError('expected RuntimeError from failing stage')

    assert state.current_stage == 'failing_stage'
    assert state.completed_stages == []
    assert len(state.stage_trace) == 1
    assert state.stage_trace[0].stage_name == 'failing_stage'
    assert state.stage_trace[0].status == 'failed'
    assert state.stage_trace[0].error_message == 'boom'
