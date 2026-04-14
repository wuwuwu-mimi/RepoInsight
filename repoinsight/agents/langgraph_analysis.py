from __future__ import annotations

from typing import Any, TypedDict

from repoinsight.agents.analysis_coordinator import (
    _build_agent_run_record,
    _build_analysis_shared_context,
    _build_memory_agent_record,
    build_default_analysis_agent_plan,
)
from repoinsight.agents.execution import execute_with_retry
from repoinsight.agents.langgraph_answer import (
    LangGraphUnavailableError,
    _load_langgraph_components,
)
from repoinsight.agents.models import AgentRunRecord, AgentStepSpec, CoordinatedAnalysisResult
from repoinsight.analyze.pipeline import run_analysis
from repoinsight.models.analysis_model import StageTraceEntry
from repoinsight.models.rag_model import IndexResult
from repoinsight.storage.index_service import index_analysis_result


class _AnalysisGraphState(TypedDict, total=False):
    """LangGraph 分析图在节点间流转的状态。"""

    url: str
    persist_knowledge: bool
    analysis_result: Any
    agent_plan: list[AgentStepSpec]
    agent_trace: list[AgentRunRecord]
    index_result: IndexResult | None
    shared_context: dict[str, str | int | float | bool | list[str]]


def run_langgraph_analysis(
    url: str,
    *,
    persist_knowledge: bool = False,
) -> CoordinatedAnalysisResult:
    """使用 LangGraph 编排分析多 Agent 流程。"""
    compiled_graph = build_analysis_state_graph()
    final_state = compiled_graph.invoke(
        {
            'url': url,
            'persist_knowledge': persist_knowledge,
            'agent_plan': build_default_analysis_agent_plan(include_memory_agent=persist_knowledge),
            'agent_trace': [],
            'shared_context': {'orchestrator': 'langgraph'},
            'index_result': None,
        }
    )
    return CoordinatedAnalysisResult(
        analysis_result=final_state['analysis_result'],
        agent_plan=final_state['agent_plan'],
        agent_trace=final_state['agent_trace'],
        index_result=final_state.get('index_result'),
        shared_context=final_state['shared_context'],
    )


def build_analysis_state_graph():
    """构建 LangGraph 版分析状态图。"""
    StateGraph, END = _load_langgraph_components()
    graph = StateGraph(_AnalysisGraphState)
    graph.add_node('pipeline_agent', _pipeline_node)
    graph.add_node('metadata_agent', _metadata_node)
    graph.add_node('structure_agent', _structure_node)
    graph.add_node('profile_agent', _profile_node)
    graph.add_node('insight_agent', _insight_node)
    graph.add_node('memory_agent', _memory_node)
    graph.set_entry_point('pipeline_agent')
    graph.add_edge('pipeline_agent', 'metadata_agent')
    graph.add_edge('metadata_agent', 'structure_agent')
    graph.add_edge('structure_agent', 'profile_agent')
    graph.add_edge('profile_agent', 'insight_agent')
    graph.add_conditional_edges(
        'insight_agent',
        _after_insight_route,
        {
            'memory_agent': 'memory_agent',
            'end': END,
        },
    )
    graph.add_edge('memory_agent', END)
    return graph.compile()


def _pipeline_node(state: _AnalysisGraphState) -> dict[str, Any]:
    """执行现有分析 pipeline，并为后续 Agent 节点准备共享上下文。"""
    analysis_result = run_analysis(state['url'])
    shared_context = dict(state['shared_context'])
    shared_context.update(_build_analysis_shared_context(analysis_result))
    shared_context['stage_trace_count'] = len(analysis_result.stage_trace)
    return {
        'analysis_result': analysis_result,
        'shared_context': shared_context,
    }


def _metadata_node(state: _AnalysisGraphState) -> dict[str, Any]:
    """汇总 metadata_agent 的执行记录。"""
    return _build_role_update(state, 'metadata_agent')


def _structure_node(state: _AnalysisGraphState) -> dict[str, Any]:
    """汇总 structure_agent 的执行记录。"""
    return _build_role_update(state, 'structure_agent')


def _profile_node(state: _AnalysisGraphState) -> dict[str, Any]:
    """汇总 profile_agent 的执行记录。"""
    return _build_role_update(state, 'profile_agent')


def _insight_node(state: _AnalysisGraphState) -> dict[str, Any]:
    """汇总 insight_agent 的执行记录。"""
    return _build_role_update(state, 'insight_agent')


def _memory_node(state: _AnalysisGraphState) -> dict[str, Any]:
    """执行 memory_agent，把分析结果写入知识文档与向量索引。"""
    memory_outcome = execute_with_retry(
        index_analysis_result,
        state['analysis_result'],
        retries=2,
    )
    if memory_outcome.error is not None:
        raise memory_outcome.error

    index_result = memory_outcome.value
    agent_trace = list(state['agent_trace'])
    agent_trace.append(
        _build_memory_agent_record(
            index_result,
            memory_outcome.attempt_count,
            memory_outcome.used_retry,
            memory_outcome.duration_ms,
        )
    )
    shared_context = dict(state['shared_context'])
    shared_context['memory_vector_indexed'] = index_result.vector_indexed
    shared_context['memory_vector_backend'] = index_result.vector_backend
    return {
        'index_result': index_result,
        'agent_trace': agent_trace,
        'shared_context': shared_context,
    }


def _build_role_update(state: _AnalysisGraphState, role: str) -> dict[str, Any]:
    """根据 stage trace 追加某个分析 Agent 的执行记录。"""
    step = _find_plan_step(state['agent_plan'], role)
    related_entries = _collect_stage_entries(state['analysis_result'].stage_trace, step.stage_names)
    agent_trace = list(state['agent_trace'])
    agent_trace.append(_build_agent_run_record(step, related_entries))
    return {'agent_trace': agent_trace}


def _find_plan_step(plan: list[AgentStepSpec], role: str) -> AgentStepSpec:
    """从 Agent 计划中找到指定角色。"""
    for step in plan:
        if step.role == role:
            return step
    raise RuntimeError(f'未找到角色 {role} 对应的 Agent 计划。')


def _collect_stage_entries(
    stage_trace: list[StageTraceEntry],
    stage_names: list[str],
) -> list[StageTraceEntry]:
    """按阶段名筛选当前 Agent 对应的 stage trace。"""
    return [entry for entry in stage_trace if entry.stage_name in stage_names]


def _after_insight_route(state: _AnalysisGraphState) -> str:
    """决定 insight_agent 之后是否需要进入 memory_agent。"""
    if state['persist_knowledge']:
        return 'memory_agent'
    return 'end'
