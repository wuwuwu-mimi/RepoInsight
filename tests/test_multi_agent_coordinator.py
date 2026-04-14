from repoinsight.agents.analysis_coordinator import (
    build_agent_trace_from_stage_trace,
    build_default_analysis_agent_plan,
    run_multi_agent_analysis,
)
from repoinsight.models.analysis_model import StageTraceEntry
from repoinsight.models.rag_model import IndexResult
import repoinsight.agents.analysis_coordinator as coordinator_module
from tests.test_summary_builders import _build_result



def test_build_default_analysis_agent_plan_can_toggle_memory_agent() -> None:
    full_plan = build_default_analysis_agent_plan(include_memory_agent=True)
    no_memory_plan = build_default_analysis_agent_plan(include_memory_agent=False)

    assert full_plan[-1].role == 'memory_agent'
    assert [item.role for item in no_memory_plan] == [
        'metadata_agent',
        'structure_agent',
        'profile_agent',
        'insight_agent',
    ]



def test_build_agent_trace_from_stage_trace_groups_into_agent_records() -> None:
    stage_trace = [
        StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
        StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
        StageTraceEntry(stage_name='clone_repository_stage', status='success', detail='已克隆到 clone/demo'),
        StageTraceEntry(stage_name='scan_repository_stage', status='success', detail='扫描到 10 个文件，识别出 3 个关键文件'),
        StageTraceEntry(stage_name='read_key_files_stage', status='success', detail='读取了 3 个关键文件'),
        StageTraceEntry(stage_name='build_project_profile_stage', status='success', detail='已生成项目画像，主语言为 Python'),
        StageTraceEntry(stage_name='build_tech_stack_stage', status='success', detail='识别出 4 条技术栈信号'),
        StageTraceEntry(stage_name='build_repo_insights_stage', status='success', detail='已生成项目洞察，项目类型为 CLI 工具'),
    ]

    records = build_agent_trace_from_stage_trace(stage_trace)

    assert [item.role for item in records] == [
        'metadata_agent',
        'structure_agent',
        'profile_agent',
        'insight_agent',
    ]
    assert records[0].status == 'success'
    assert records[0].completed_stage_names == [
        'validate_analysis_url_stage',
        'fetch_repo_metadata_stage',
    ]
    assert 'URL 校验通过' in records[0].detail
    assert records[1].completed_stage_names == [
        'clone_repository_stage',
        'scan_repository_stage',
        'read_key_files_stage',
    ]
    assert '读取了 3 个关键文件' in records[1].detail
    assert records[3].detail == '已生成项目洞察，项目类型为 CLI 工具'



def test_build_agent_trace_marks_failed_agent_when_stage_failed() -> None:
    stage_trace = [
        StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
        StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
        StageTraceEntry(stage_name='clone_repository_stage', status='failed', error_message='网络异常'),
    ]

    records = build_agent_trace_from_stage_trace(stage_trace)

    assert records[1].role == 'structure_agent'
    assert records[1].status == 'failed'
    assert records[1].error_message == '网络异常'
    assert records[1].completed_stage_names == []



def test_run_multi_agent_analysis_can_attach_memory_agent() -> None:
    original_run_analysis = coordinator_module.run_analysis
    original_index_analysis_result = coordinator_module.index_analysis_result
    try:
        result = _build_result([], [])
        result.stage_trace = [
            StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
            StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
            StageTraceEntry(stage_name='clone_repository_stage', status='success', detail='已克隆到 clone/demo'),
            StageTraceEntry(stage_name='scan_repository_stage', status='success', detail='扫描到 0 个文件，识别出 0 个关键文件'),
            StageTraceEntry(stage_name='read_key_files_stage', status='success', detail='读取了 0 个关键文件'),
            StageTraceEntry(stage_name='build_project_profile_stage', status='success', detail='已生成项目画像，主语言为 Python'),
            StageTraceEntry(stage_name='build_tech_stack_stage', status='success', detail='识别出 0 条技术栈信号'),
            StageTraceEntry(stage_name='build_repo_insights_stage', status='success', detail='已生成项目洞察，项目类型为 CLI 工具'),
        ]

        coordinator_module.run_analysis = lambda url: result
        coordinator_module.index_analysis_result = lambda analysis_result: IndexResult(
            local_path='data/knowledge/demo__sample.json',
            vector_indexed=True,
            vector_backend='chroma',
            message='已同步写入 Chroma 向量索引。',
        )

        coordinated = run_multi_agent_analysis(
            'https://github.com/demo/sample',
            persist_knowledge=True,
        )

        assert coordinated.analysis_result.repo_info.repo_model.full_name == 'demo/sample'
        assert coordinated.index_result is not None
        assert coordinated.agent_plan[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-1].status == 'success'
        assert 'Chroma' in coordinated.agent_trace[-1].detail
    finally:
        coordinator_module.run_analysis = original_run_analysis
        coordinator_module.index_analysis_result = original_index_analysis_result


def test_run_multi_agent_analysis_retries_memory_agent_and_exposes_shared_context() -> None:
    original_run_analysis = coordinator_module.run_analysis
    original_index_analysis_result = coordinator_module.index_analysis_result
    try:
        result = _build_result([], [])
        result.project_type = 'CLI 工具'
        result.stage_trace = [
            StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
            StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
            StageTraceEntry(stage_name='clone_repository_stage', status='success', detail='已克隆到 clone/demo'),
            StageTraceEntry(stage_name='scan_repository_stage', status='success', detail='扫描到 0 个文件，识别出 0 个关键文件'),
            StageTraceEntry(stage_name='read_key_files_stage', status='success', detail='读取了 0 个关键文件'),
            StageTraceEntry(stage_name='build_project_profile_stage', status='success', detail='已生成项目画像，主语言为 Python'),
            StageTraceEntry(stage_name='build_tech_stack_stage', status='success', detail='识别出 0 条技术栈信号'),
            StageTraceEntry(stage_name='build_repo_insights_stage', status='success', detail='已生成项目洞察，项目类型为 CLI 工具'),
        ]

        call_count = {'value': 0}

        def flaky_index_analysis_result(analysis_result):
            call_count['value'] += 1
            if call_count['value'] == 1:
                raise RuntimeError('temporary memory error')
            return IndexResult(
                local_path='data/knowledge/demo__sample.json',
                vector_indexed=False,
                vector_backend='none',
                message='仅保存本地知识文档。',
            )

        coordinator_module.run_analysis = lambda url: result
        coordinator_module.index_analysis_result = flaky_index_analysis_result

        coordinated = run_multi_agent_analysis(
            'https://github.com/demo/sample',
            persist_knowledge=True,
        )

        assert call_count['value'] == 2
        assert coordinated.agent_trace[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-1].used_retry is True
        assert coordinated.agent_trace[-1].attempt_count == 2
        assert coordinated.shared_context['repo_id'] == 'demo/sample'
        assert coordinated.shared_context['project_type'] == 'CLI 工具'
    finally:
        coordinator_module.run_analysis = original_run_analysis
        coordinator_module.index_analysis_result = original_index_analysis_result
