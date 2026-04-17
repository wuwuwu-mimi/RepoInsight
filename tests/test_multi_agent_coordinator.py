from repoinsight.agents.analysis_coordinator import (
    build_analysis_task_packets,
    build_agent_trace_from_stage_trace,
    build_default_analysis_agent_plan,
    build_dynamic_analysis_agent_plan,
    run_multi_agent_analysis,
)
from repoinsight.models.analysis_model import (
    ApiRouteSummary,
    ClassSummary,
    CodeEntity,
    CodeRelationEdge,
    FunctionSummary,
    ModuleRelation,
    StageTraceEntry,
    SubprojectSummary,
    TechStackItem,
)
from repoinsight.models.file_model import IgnoredEntry
from repoinsight.models.rag_model import IndexResult
import repoinsight.agents.analysis_coordinator as coordinator_module
from tests.test_summary_builders import _build_result



def test_build_default_analysis_agent_plan_can_toggle_memory_agent() -> None:
    full_plan = build_default_analysis_agent_plan(include_memory_agent=True)
    no_memory_plan = build_default_analysis_agent_plan(include_memory_agent=False)

    assert full_plan[-1].role == 'memory_agent'
    assert [item.role for item in no_memory_plan] == [
        'planner_agent',
        'repo_agent',
        'readme_agent',
        'structure_agent',
        'codebase_agent',
        'profile_agent',
        'insight_agent',
        'verifier_agent',
    ]



def test_build_agent_trace_from_stage_trace_groups_into_agent_records() -> None:
    stage_trace = [
        StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
        StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
        StageTraceEntry(stage_name='fetch_repo_readme_stage', status='success', detail='已获取 README'),
        StageTraceEntry(stage_name='clone_repository_stage', status='success', detail='已克隆到 clone/demo'),
        StageTraceEntry(stage_name='scan_repository_stage', status='success', detail='扫描到 10 个文件，识别出 3 个关键文件'),
        StageTraceEntry(stage_name='read_key_files_stage', status='success', detail='读取了 3 个关键文件'),
        StageTraceEntry(stage_name='build_project_profile_stage', status='success', detail='已生成项目画像，主语言为 Python'),
        StageTraceEntry(stage_name='build_tech_stack_stage', status='success', detail='识别出 4 条技术栈信号'),
        StageTraceEntry(stage_name='build_repo_insights_stage', status='success', detail='已生成项目洞察，项目类型为 CLI 工具'),
    ]

    records = build_agent_trace_from_stage_trace(stage_trace)

    assert [item.role for item in records] == [
        'repo_agent',
        'readme_agent',
        'structure_agent',
        'codebase_agent',
        'profile_agent',
        'insight_agent',
    ]
    assert records[0].status == 'success'
    assert records[0].completed_stage_names == [
        'validate_analysis_url_stage',
        'fetch_repo_metadata_stage',
    ]
    assert 'URL 校验通过' in records[0].detail
    assert records[1].completed_stage_names == ['fetch_repo_readme_stage']
    assert 'README' in records[1].detail
    assert records[2].completed_stage_names == [
        'clone_repository_stage',
        'scan_repository_stage',
        'read_key_files_stage',
    ]
    assert '读取了 3 个关键文件' in records[2].detail
    assert records[3].detail == '已生成项目画像，主语言为 Python'
    assert records[4].detail == '识别出 4 条技术栈信号'
    assert records[5].detail == '已生成项目洞察，项目类型为 CLI 工具'



def test_build_analysis_task_packets_includes_planned_handoffs() -> None:
    result = _build_result([], ['app.py'])

    plan = build_dynamic_analysis_agent_plan(result, include_memory_agent=True)
    packets = build_analysis_task_packets(result, include_memory_agent=True, agent_plan=plan)

    assert packets['repo_agent'].depends_on == ['planner_agent']
    assert 'README 原文' in packets['readme_agent'].required_inputs
    assert 'entrypoints' in packets['codebase_agent'].expected_outputs[0]
    assert packets['profile_agent'].depends_on == ['structure_agent']
    assert packets['insight_agent'].depends_on == ['codebase_agent', 'profile_agent']
    assert packets['memory_agent'].depends_on == ['verifier_agent']





def test_build_dynamic_analysis_agent_plan_exposes_parallel_codebase_and_profile_wave() -> None:
    result = _build_result([], ['app.py', 'package.json'])

    plan = build_dynamic_analysis_agent_plan(result, include_memory_agent=False)
    step_by_role = {item.role: item for item in plan}

    assert step_by_role['codebase_agent'].depends_on == ['structure_agent']
    assert step_by_role['profile_agent'].depends_on == ['structure_agent']
    assert step_by_role['insight_agent'].depends_on == ['codebase_agent', 'profile_agent']

def test_build_dynamic_analysis_agent_plan_can_skip_code_and_profile_agents() -> None:
    result = _build_result([], ['README.md', 'docs/guide.md'])
    result.scan_result.all_files = []
    result.scan_result.key_files = []
    result.project_profile.primary_language = None
    result.project_profile.languages = []
    result.project_profile.runtimes = []
    result.project_profile.frameworks = []
    result.project_profile.build_tools = []
    result.project_profile.package_managers = []
    result.project_profile.test_tools = []
    result.project_profile.deploy_tools = []
    result.project_profile.entrypoints = []
    result.project_profile.subprojects = []
    result.project_profile.module_relations = []
    result.project_profile.code_entities = []
    result.project_profile.code_relation_edges = []
    result.project_profile.function_summaries = []
    result.project_profile.class_summaries = []
    result.project_profile.api_route_summaries = []
    result.project_profile.confirmed_signals = []
    result.project_profile.weak_signals = []
    result.tech_stack = []

    plan = build_dynamic_analysis_agent_plan(result, include_memory_agent=False)
    packets = build_analysis_task_packets(result, include_memory_agent=False, agent_plan=plan)

    assert [item.role for item in plan] == [
        'planner_agent',
        'repo_agent',
        'readme_agent',
        'structure_agent',
        'insight_agent',
        'verifier_agent',
    ]
    assert 'codebase_agent' not in packets
    assert 'profile_agent' not in packets

def test_build_agent_trace_marks_failed_agent_when_stage_failed() -> None:

    stage_trace = [
        StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
        StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
        StageTraceEntry(stage_name='fetch_repo_readme_stage', status='success', detail='已获取 README'),
        StageTraceEntry(stage_name='clone_repository_stage', status='failed', error_message='网络异常'),
    ]

    records = build_agent_trace_from_stage_trace(stage_trace)

    assert records[2].role == 'structure_agent'
    assert records[2].status == 'failed'
    assert records[2].error_message == '网络异常'
    assert records[2].completed_stage_names == []




def test_structure_agent_structured_output_exposes_tree_and_subprojects() -> None:
    result = _build_result([], ['apps/api/main.py', 'apps/web/src/main.tsx', 'docs/intro.md'])
    result.scan_result.ignored_entries = [
        IgnoredEntry(path='node_modules', entry_type='directory', reason='ignored_dir')
    ]
    result.project_profile.subprojects = [
        SubprojectSummary(
            root_path='apps/api',
            language_scope='python',
            project_kind='service',
            config_paths=['apps/api/pyproject.toml'],
            entrypoint_paths=['apps/api/main.py'],
            markers=['service'],
        ),
        SubprojectSummary(
            root_path='apps/web',
            language_scope='nodejs',
            project_kind='package',
            config_paths=['apps/web/package.json'],
            entrypoint_paths=['apps/web/src/main.tsx'],
            markers=['workspace'],
        ),
    ]

    output = coordinator_module._build_analysis_agent_structured_output('structure_agent', result)

    assert output.metadata['ignored_count'] == 1
    assert output.metadata['subproject_count'] == 2
    assert output.metadata['top_level_dirs'] == ['apps', 'docs']
    assert any(item.kind == 'tree_preview' and '目录预览' in item.label for item in output.evidence)
    assert any(item.kind == 'subproject' and 'apps/api' in item.label for item in output.evidence)
    assert any('子项目/工作区' in item for item in output.conclusions)


def test_codebase_agent_structured_output_exposes_routes_and_symbols() -> None:
    result = _build_result([], ['app.py', 'src/main.tsx'])
    result.project_profile.function_summaries = [
        FunctionSummary(
            name='create_app',
            qualified_name='create_app',
            source_path='app.py',
            language_scope='python',
            line_start=10,
            line_end=32,
            signature='def create_app() -> FastAPI',
            owner_class=None,
            is_async=False,
            decorators=[],
            parameters=[],
            called_symbols=['FastAPI', 'include_router'],
            return_signals=['FastAPI'],
            summary='创建并配置 FastAPI 应用。',
        )
    ]
    result.project_profile.class_summaries = [
        ClassSummary(
            name='Settings',
            qualified_name='Settings',
            source_path='app.py',
            language_scope='python',
            line_start=1,
            line_end=8,
            bases=['BaseSettings'],
            decorators=[],
            methods=['load'],
            summary='封装应用配置读取逻辑。',
        )
    ]
    result.project_profile.api_route_summaries = [
        ApiRouteSummary(
            route_path='/health',
            http_methods=['GET'],
            source_path='app.py',
            language_scope='python',
            framework='fastapi',
            handler_name='health_check',
            handler_qualified_name='health_check',
            owner_class=None,
            line_number=48,
            decorators=['app.get'],
            called_symbols=['jsonify'],
            summary='提供健康检查接口。',
        )
    ]
    result.project_profile.module_relations = [
        ModuleRelation(
            source_path='app.py',
            target='routers.health',
            relation_type='import',
            line_number=3,
        )
    ]
    result.project_profile.code_entities = [
        CodeEntity(
            entity_kind='function',
            name='create_app',
            qualified_name='create_app',
            source_path='app.py',
            language_scope='python',
            location='app.py:L10-L32',
            tags=['entrypoint'],
        )
    ]
    result.project_profile.code_relation_edges = [
        CodeRelationEdge(
            source_ref='create_app',
            target_ref='health_check',
            relation_type='calls',
            source_path='app.py',
            line_number=24,
        )
    ]

    output = coordinator_module._build_analysis_agent_structured_output('codebase_agent', result)

    assert output.metadata['function_summary_count'] == 1
    assert output.metadata['class_summary_count'] == 1
    assert output.metadata['api_route_count'] == 1
    assert output.metadata['entrypoints'] == ['app.py', 'src/main.tsx']
    assert any(item.kind == 'api_route' and 'GET /health -> health_check' in item.label for item in output.evidence)
    assert any(item.kind == 'function_summary' and item.location == 'L10-L32' for item in output.evidence)
    assert any(item.kind == 'code_relation' and 'calls' in item.label for item in output.evidence)
    assert any('函数摘要' in item for item in output.conclusions)


def test_profile_agent_structured_output_exposes_signal_layers() -> None:
    result = _build_result([], ['app.py', 'package.json', 'docker-compose.yml'])
    result.tech_stack = [
        TechStackItem(
            name='FastAPI',
            category='framework',
            evidence='import fastapi',
            evidence_level='strong',
            evidence_source='import',
            source_path='app.py',
        ),
        TechStackItem(
            name='Docker',
            category='deploy_tool',
            evidence='docker-compose.yml',
            evidence_level='medium',
            evidence_source='config',
            source_path='docker-compose.yml',
        ),
        TechStackItem(
            name='Celery',
            category='runtime',
            evidence='requirements-dev.txt mentions celery',
            evidence_level='weak',
            evidence_source='dependency',
            source_path='requirements-dev.txt',
        ),
    ]
    result.project_profile.confirmed_signals = result.tech_stack[:2]
    result.project_profile.weak_signals = result.tech_stack[2:]

    output = coordinator_module._build_analysis_agent_structured_output('profile_agent', result)

    assert output.metadata['tech_stack_count'] == 3
    assert output.metadata['confirmed_signal_count'] == 2
    assert output.metadata['weak_signal_count'] == 1
    assert output.metadata['package_managers'] == ['npm', 'pip']
    assert output.metadata['deploy_tools'] == ['Docker']
    assert any(item.kind == 'tech_stack_signal' and 'FastAPI | framework | strong' in item.label for item in output.evidence)
    assert any(item.kind == 'weak_signal' and 'Celery | runtime | weak' in item.label for item in output.evidence)
    assert any('工程化配套' in item for item in output.conclusions)


def test_insight_agent_structured_output_exposes_evidence_and_counts() -> None:
    result = _build_result([], ['README.md', 'app.py'])
    result.project_type = 'Web 服务'
    result.project_type_evidence = 'README 提到 FastAPI 服务，并且存在 app.py 入口。'
    result.observations = [
        '该仓库初步被归类为：Web 服务。',
        '当前规则识别出的技术栈包括：Python, FastAPI。',
    ]
    result.strengths = ['README 已成功获取，说明项目基础文档相对完整。']
    result.risks = ['暂未检测到许可证信息，直接复用前建议确认开源使用边界。']

    output = coordinator_module._build_analysis_agent_structured_output('insight_agent', result)

    assert output.metadata['project_type'] == 'Web 服务'
    assert output.metadata['has_project_type_evidence'] is True
    assert output.metadata['observation_count'] == 2
    assert output.metadata['strength_count'] == 1
    assert output.metadata['risk_count'] == 1
    assert any(item.kind == 'project_type_evidence' and item.snippet == result.project_type_evidence for item in output.evidence)
    assert any(item.kind == 'observation' and 'Web 服务' in item.label for item in output.evidence)
    assert any(item.kind == 'strength' for item in output.evidence)
    assert any(item.kind == 'risk' for item in output.evidence)
    assert any('当前优势重点' in item for item in output.conclusions)


def test_run_multi_agent_analysis_can_skip_codebase_and_profile_agents() -> None:
    original_run_analysis = coordinator_module.run_analysis
    try:
        result = _build_result([], ['README.md', 'docs/guide.md'])
        result.stage_trace = [
            StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
            StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
            StageTraceEntry(stage_name='fetch_repo_readme_stage', status='success', detail='已获取 README'),
            StageTraceEntry(stage_name='clone_repository_stage', status='success', detail='已克隆到 clone/demo'),
            StageTraceEntry(stage_name='scan_repository_stage', status='success', detail='扫描到 2 个文件，识别出 1 个关键文件'),
            StageTraceEntry(stage_name='read_key_files_stage', status='success', detail='读取了 1 个关键文件'),
            StageTraceEntry(stage_name='build_project_profile_stage', status='success', detail='已生成项目画像，主语言为未知'),
            StageTraceEntry(stage_name='build_tech_stack_stage', status='success', detail='识别出 0 条技术栈信号'),
            StageTraceEntry(stage_name='build_repo_insights_stage', status='success', detail='已生成项目洞察，项目类型为文档项目'),
        ]
        result.project_type = '文档项目'
        result.project_profile.primary_language = None
        result.project_profile.languages = []
        result.project_profile.runtimes = []
        result.project_profile.frameworks = []
        result.project_profile.build_tools = []
        result.project_profile.package_managers = []
        result.project_profile.test_tools = []
        result.project_profile.deploy_tools = []
        result.project_profile.entrypoints = []
        result.project_profile.subprojects = []
        result.project_profile.module_relations = []
        result.project_profile.code_entities = []
        result.project_profile.code_relation_edges = []
        result.project_profile.function_summaries = []
        result.project_profile.class_summaries = []
        result.project_profile.api_route_summaries = []
        result.project_profile.confirmed_signals = []
        result.project_profile.weak_signals = []
        result.tech_stack = []

        coordinator_module.run_analysis = lambda url: result

        coordinated = run_multi_agent_analysis(
            'https://github.com/demo/sample',
            persist_knowledge=False,
        )

        assert [item.role for item in coordinated.agent_plan] == [
            'planner_agent',
            'repo_agent',
            'readme_agent',
            'structure_agent',
            'insight_agent',
            'verifier_agent',
        ]
        assert [item.role for item in coordinated.agent_trace] == [
            'planner_agent',
            'repo_agent',
            'readme_agent',
            'structure_agent',
            'insight_agent',
            'verifier_agent',
        ]
        assert coordinated.shared_context['planner_skipped_roles'] == ['codebase_agent', 'profile_agent', 'memory_agent']
        assert coordinated.shared_context['planner_parallel_groups'] == []
    finally:
        coordinator_module.run_analysis = original_run_analysis

def test_run_multi_agent_analysis_can_attach_memory_agent() -> None:
    original_run_analysis = coordinator_module.run_analysis
    original_index_analysis_result = coordinator_module.index_analysis_result
    try:
        result = _build_result([], [])
        result.stage_trace = [
            StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
            StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
            StageTraceEntry(stage_name='fetch_repo_readme_stage', status='success', detail='已获取 README'),
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
        assert coordinated.agent_plan[0].role == 'planner_agent'
        assert coordinated.agent_trace[0].role == 'planner_agent'
        assert coordinated.agent_plan[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-2].role == 'verifier_agent'
        assert coordinated.agent_trace[-2].structured_output is not None
        assert coordinated.agent_trace[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-1].status == 'success'
        assert 'Chroma' in coordinated.agent_trace[-1].detail
        assert coordinated.shared_context['planner_task_count'] >= 7
        assert 'codebase_agent' in coordinated.shared_context['planner_parallel_roles']
        assert 'profile_agent' in coordinated.shared_context['planner_parallel_roles']
        assert 'codebase_agent + profile_agent' in coordinated.shared_context['planner_parallel_groups']
        assert coordinated.shared_context['agent_execution_mode'] == 'local_parallel'
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
            StageTraceEntry(stage_name='fetch_repo_readme_stage', status='success', detail='已获取 README'),
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
        assert coordinated.agent_trace[0].role == 'planner_agent'
        assert coordinated.agent_trace[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-1].used_retry is True
        assert coordinated.agent_trace[-1].attempt_count == 2
        assert coordinated.agent_trace[-2].role == 'verifier_agent'
        assert coordinated.shared_context['repo_id'] == 'demo/sample'
        assert coordinated.shared_context['project_type'] == 'CLI 工具'
        assert coordinated.shared_context['analysis_verification_verdict'] in {'passed', 'warning', 'failed'}
        assert any(item.startswith('repo_agent<-planner_agent') for item in coordinated.shared_context['planner_handoff_order'])
    finally:
        coordinator_module.run_analysis = original_run_analysis
        coordinator_module.index_analysis_result = original_index_analysis_result
