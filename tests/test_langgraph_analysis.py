import io
import sys
import types
from contextlib import redirect_stdout

import repoinsight.agents.analysis_coordinator as analysis_coordinator_module
import repoinsight.agents.langgraph_analysis as langgraph_analysis_module
import repoinsight.cli.main as cli_main
from repoinsight.agents.langgraph_analysis import run_langgraph_analysis
from repoinsight.agents.models import CoordinatedAnalysisResult
from repoinsight.models.analysis_model import StageTraceEntry
from repoinsight.models.rag_model import IndexResult
from tests.test_summary_builders import _build_result


class _FakeCompiledGraph:
    def __init__(self, graph) -> None:
        self._graph = graph

    def invoke(self, state):
        current = self._graph.entry_point
        current_state = dict(state)
        while current is not None:
            updates = self._graph.nodes[current](current_state)
            if updates:
                current_state.update(updates)
            if current in self._graph.conditional_edges:
                router, mapping = self._graph.conditional_edges[current]
                next_node = mapping[router(current_state)]
            else:
                next_node = self._graph.edges.get(current)
            if next_node == self._graph.end_token:
                break
            current = next_node
        return current_state


class _FakeStateGraph:
    def __init__(self, _state_type) -> None:
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.entry_point = None
        self.end_token = '__end__'

    def add_node(self, name, func) -> None:
        self.nodes[name] = func

    def set_entry_point(self, name) -> None:
        self.entry_point = name

    def add_edge(self, source, target) -> None:
        self.edges[source] = target

    def add_conditional_edges(self, source, router, mapping) -> None:
        self.conditional_edges[source] = (router, mapping)

    def compile(self):
        return _FakeCompiledGraph(self)


def _install_fake_langgraph() -> None:
    fake_langgraph = types.ModuleType('langgraph')
    fake_graph_module = types.ModuleType('langgraph.graph')
    fake_graph_module.END = '__end__'
    fake_graph_module.StateGraph = _FakeStateGraph
    sys.modules['langgraph'] = fake_langgraph
    sys.modules['langgraph.graph'] = fake_graph_module


def _uninstall_fake_langgraph() -> None:
    sys.modules.pop('langgraph', None)
    sys.modules.pop('langgraph.graph', None)


def _build_analysis_result():
    result = _build_result([], [])
    result.project_type = 'CLI 工具'
    result.stage_trace = [
        StageTraceEntry(stage_name='validate_analysis_url_stage', status='success', detail='URL 校验通过'),
        StageTraceEntry(stage_name='fetch_repo_metadata_stage', status='success', detail='已获取仓库元数据'),
        StageTraceEntry(stage_name='fetch_repo_readme_stage', status='success', detail='已获取 README'),
        StageTraceEntry(stage_name='clone_repository_stage', status='success', detail='已克隆到 clones/demo__sample'),
        StageTraceEntry(stage_name='scan_repository_stage', status='success', detail='扫描到 0 个文件，识别出 0 个关键文件'),
        StageTraceEntry(stage_name='read_key_files_stage', status='success', detail='读取了 0 个关键文件'),
        StageTraceEntry(stage_name='build_project_profile_stage', status='success', detail='已生成项目画像，主语言为 Python'),
        StageTraceEntry(stage_name='build_tech_stack_stage', status='success', detail='识别出 0 条技术栈信号'),
        StageTraceEntry(stage_name='build_repo_insights_stage', status='success', detail='已生成项目洞察，项目类型为 CLI 工具'),
    ]
    return result


def test_run_langgraph_analysis_with_fake_langgraph() -> None:
    original_run_analysis = langgraph_analysis_module.run_analysis
    _install_fake_langgraph()
    try:
        langgraph_analysis_module.run_analysis = lambda url: _build_analysis_result()

        coordinated = run_langgraph_analysis(
            'https://github.com/demo/sample',
            persist_knowledge=False,
        )

        assert coordinated.analysis_result.repo_info.repo_model.full_name == 'demo/sample'
        assert coordinated.shared_context['orchestrator'] == 'langgraph'
        assert coordinated.shared_context['project_type'] == 'CLI 工具'
        assert [item.role for item in coordinated.agent_trace] == [
            'planner_agent',
            'repo_agent',
            'readme_agent',
            'structure_agent',
            'codebase_agent',
            'profile_agent',
            'insight_agent',
            'verifier_agent',
        ]
        assert coordinated.index_result is None
        assert coordinated.agent_trace[-1].structured_output is not None
    finally:
        langgraph_analysis_module.run_analysis = original_run_analysis
        _uninstall_fake_langgraph()



def test_run_langgraph_analysis_can_skip_codebase_and_profile_agents() -> None:
    original_run_analysis = langgraph_analysis_module.run_analysis
    _install_fake_langgraph()
    try:
        result = _build_analysis_result()
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
        langgraph_analysis_module.run_analysis = lambda url: result

        coordinated = run_langgraph_analysis(
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
        assert coordinated.shared_context['agent_execution_mode'] == 'langgraph_dynamic'
    finally:
        langgraph_analysis_module.run_analysis = original_run_analysis
        _uninstall_fake_langgraph()

def test_run_langgraph_analysis_can_attach_memory_agent() -> None:
    original_run_analysis = langgraph_analysis_module.run_analysis
    original_index_analysis_result = langgraph_analysis_module.index_analysis_result
    _install_fake_langgraph()
    try:
        langgraph_analysis_module.run_analysis = lambda url: _build_analysis_result()
        langgraph_analysis_module.index_analysis_result = lambda analysis_result: IndexResult(
            local_path='data/knowledge/demo__sample.json',
            vector_indexed=True,
            vector_backend='chroma',
            message='已同步写入 Chroma 向量索引。',
        )

        coordinated = run_langgraph_analysis(
            'https://github.com/demo/sample',
            persist_knowledge=True,
        )

        assert coordinated.index_result is not None
        assert coordinated.agent_plan[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-1].role == 'memory_agent'
        assert coordinated.agent_trace[-2].role == 'verifier_agent'
        assert coordinated.shared_context['memory_vector_backend'] == 'chroma'
        assert coordinated.shared_context['agent_execution_mode'] == 'langgraph_dynamic'
        assert 'codebase_agent + profile_agent' in coordinated.shared_context['planner_parallel_groups']
    finally:
        langgraph_analysis_module.run_analysis = original_run_analysis
        langgraph_analysis_module.index_analysis_result = original_index_analysis_result
        _uninstall_fake_langgraph()


def test_analyze_command_reports_missing_langgraph_dependency() -> None:
    original_apply_embedding_mode = cli_main._apply_embedding_mode
    original_run_langgraph_analysis = cli_main.run_langgraph_analysis
    try:
        cli_main._apply_embedding_mode = lambda mode: True
        cli_main.run_langgraph_analysis = lambda *args, **kwargs: (_ for _ in ()).throw(
            cli_main.LangGraphUnavailableError('当前未安装 LangGraph，请先执行 `pip install langgraph`。')
        )
        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.analyze(
                'https://github.com/demo/sample',
                save_report=False,
                orchestrator='langgraph',
            )

        rendered = output.getvalue()
        assert 'LangGraph 编排不可用' in rendered
        assert 'pip install langgraph' in rendered
    finally:
        cli_main._apply_embedding_mode = original_apply_embedding_mode
        cli_main.run_langgraph_analysis = original_run_langgraph_analysis


def test_analyze_command_uses_langgraph_orchestrator() -> None:
    original_apply_embedding_mode = cli_main._apply_embedding_mode
    original_run_langgraph_analysis = cli_main.run_langgraph_analysis
    original_run_multi_agent_analysis = cli_main.run_multi_agent_analysis
    try:
        cli_main._apply_embedding_mode = lambda mode: True
        called = {'langgraph': 0, 'local': 0}

        def fake_run_langgraph_analysis(url: str, *, persist_knowledge: bool) -> CoordinatedAnalysisResult:
            called['langgraph'] += 1
            assert persist_knowledge is False
            result = _build_analysis_result()
            task_packets = analysis_coordinator_module.build_analysis_task_packets(
                result,
                include_memory_agent=False,
            )
            return CoordinatedAnalysisResult(
                analysis_result=result,
                agent_plan=analysis_coordinator_module.build_default_analysis_agent_plan(),
                agent_trace=[
                    analysis_coordinator_module._build_planner_agent_record(
                        analysis_coordinator_module.build_default_analysis_agent_plan(),
                        task_packets,
                    )
                ]
                + analysis_coordinator_module.build_agent_trace_from_stage_trace(
                    result.stage_trace,
                    analysis_result=result,
                    task_packets_by_role=task_packets,
                )
                + [analysis_coordinator_module._build_verifier_agent_record(result)],
                index_result=None,
                shared_context={'orchestrator': 'langgraph', 'planner_task_count': len(task_packets)},
            )

        def fake_run_multi_agent_analysis(url: str, *, persist_knowledge: bool) -> CoordinatedAnalysisResult:
            called['local'] += 1
            raise AssertionError('不应走本地编排器')

        cli_main.run_langgraph_analysis = fake_run_langgraph_analysis
        cli_main.run_multi_agent_analysis = fake_run_multi_agent_analysis

        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.analyze(
                'https://github.com/demo/sample',
                save_report=False,
                orchestrator='langgraph',
            )

        rendered = output.getvalue()
        assert called['langgraph'] == 1
        assert called['local'] == 0
        assert '编排器' in rendered
        assert 'langgraph' in rendered
        assert '任务规划' in rendered
        assert '仓库摘要' in rendered
        assert 'README 摘要' in rendered
        assert '代码库摘要' in rendered
        assert '技术栈摘要' in rendered
        assert '洞察摘要' in rendered
        assert '分析验证' in rendered
        assert '验证结论' in rendered
        assert '关键证据' in rendered
    finally:
        cli_main._apply_embedding_mode = original_apply_embedding_mode
        cli_main.run_langgraph_analysis = original_run_langgraph_analysis
        cli_main.run_multi_agent_analysis = original_run_multi_agent_analysis
