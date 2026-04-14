import io
import shutil
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import repoinsight.agents.langgraph_answer as langgraph_answer_module
import repoinsight.cli.main as cli_main
from repoinsight.agents.langgraph_answer import run_langgraph_answer
from repoinsight.models.analysis_model import KeyFileContent
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents
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



def _install_fake_langgraph():
    fake_langgraph = types.ModuleType('langgraph')
    fake_graph_module = types.ModuleType('langgraph.graph')
    fake_graph_module.END = '__end__'
    fake_graph_module.StateGraph = _FakeStateGraph
    sys.modules['langgraph'] = fake_langgraph
    sys.modules['langgraph.graph'] = fake_graph_module



def _uninstall_fake_langgraph() -> None:
    sys.modules.pop('langgraph', None)
    sys.modules.pop('langgraph.graph', None)



def test_run_langgraph_answer_with_fake_langgraph() -> None:
    temp_dir = Path('data/test_langgraph_answer')
    shutil.rmtree(temp_dir, ignore_errors=True)
    _install_fake_langgraph()
    try:
        package_json = KeyFileContent(
            path='package.json',
            size_bytes=320,
            content=(
                '{'
                '"name":"hello-agent",'
                '"packageManager":"pnpm@9.0.0",'
                '"scripts":{"dev":"vite","start":"node server.js"},'
                '"dependencies":{"react":"^18.0.0"}'
                '}'
            ),
        )
        main_file = KeyFileContent(
            path='main.py',
            size_bytes=240,
            content=(
                'def build_agent():\n'
                '    return None\n'
            ),
        )
        result = _build_result([package_json, main_file], ['package.json', 'main.py'])
        result.repo_info.readme = '# helloAgent\n\n一个从零手写 Agent 的练习项目。'
        result.project_type = 'Agent 学习项目'
        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        coordinated = run_langgraph_answer(
            repo_id='demo/sample',
            question='这个项目是做什么的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert coordinated.route_decision.focus == 'overview'
        assert coordinated.shared_context['orchestrator'] == 'langgraph'
        assert [item.role for item in coordinated.agent_trace] == [
            'router_agent',
            'retrieval_agent',
            'synthesis_agent',
            'verifier_agent',
            'recovery_agent',
            'revision_agent',
        ]
        assert coordinated.verification_result is not None
        assert coordinated.answer_result.answer_mode == 'extractive'
    finally:
        _uninstall_fake_langgraph()
        shutil.rmtree(temp_dir, ignore_errors=True)



def test_answer_command_reports_missing_langgraph_dependency() -> None:
    original_get_llm_settings = cli_main.get_llm_settings
    original_run_langgraph_answer = cli_main.run_langgraph_answer
    try:
        cli_main.get_llm_settings = lambda: None
        cli_main.run_langgraph_answer = lambda **kwargs: (_ for _ in ()).throw(
            cli_main.LangGraphUnavailableError('当前未安装 LangGraph，请先执行 `pip install langgraph`。')
        )
        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.answer(
                'demo/sample',
                '这个项目是做什么的？',
                use_llm=False,
                stream=False,
                orchestrator='langgraph',
            )

        rendered = output.getvalue()
        assert 'LangGraph 编排不可用' in rendered
        assert 'pip install langgraph' in rendered
    finally:
        cli_main.get_llm_settings = original_get_llm_settings
        cli_main.run_langgraph_answer = original_run_langgraph_answer
