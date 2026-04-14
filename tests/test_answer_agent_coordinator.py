import shutil
from pathlib import Path

from repoinsight.agents.answer_coordinator import (
    build_default_answer_agent_plan,
    run_multi_agent_answer,
)
import repoinsight.agents.answer_coordinator as coordinator_module
from repoinsight.agents.execution import ExecutionOutcome
from repoinsight.agents.models import CodeInvestigationResult
from repoinsight.models.answer_model import RepoAnswerResult
from repoinsight.models.analysis_model import KeyFileContent
from repoinsight.models.rag_model import KnowledgeDocument, SearchHit, SearchResult
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents
from tests.test_summary_builders import _build_result



def test_build_default_answer_agent_plan_returns_seven_roles() -> None:
    plan = build_default_answer_agent_plan()

    assert [item.role for item in plan] == [
        'router_agent',
        'retrieval_agent',
        'code_agent',
        'synthesis_agent',
        'verifier_agent',
        'recovery_agent',
        'revision_agent',
    ]
    assert plan[1].depends_on == ['router_agent']
    assert plan[2].depends_on == ['retrieval_agent']
    assert plan[3].depends_on == ['retrieval_agent', 'code_agent']
    assert plan[4].depends_on == ['synthesis_agent']
    assert plan[5].depends_on == ['verifier_agent']
    assert plan[6].depends_on == ['recovery_agent']



def test_run_multi_agent_answer_builds_router_retrieval_synthesis_trace() -> None:
    temp_dir = Path('data/test_answer_agent_coordinator')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
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
                'class AlwaysFailTool:\n'
                '    def __init__(self):\n'
                '        super().__init__()\n'
            ),
        )
        result = _build_result([package_json, main_file], ['package.json', 'main.py'])
        result.repo_info.readme = (
            '# helloAgent\n\n'
            '一个从零手写 Agent 的练习项目。\n\n'
            '这个仓库的目标是把 Prompt、推理、工具调用和最终回答这条链路一步一步写清楚。'
        )
        result.repo_info.repo_model.description = '一个从零实现 Agent 核心链路的学习型仓库'
        result.project_type = 'Agent 学习项目'
        result.project_type_evidence = 'README 与关键文件都围绕 Agent 主链路展开'

        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='这个项目是做什么的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert coordinated.route_decision.focus == 'overview'
        assert coordinated.retrieval_hit_count > 0
        assert coordinated.retrieved_doc_types[0] == 'readme_summary'
        assert coordinated.code_investigation is None
        assert [item.role for item in coordinated.agent_trace] == [
            'router_agent',
            'retrieval_agent',
            'synthesis_agent',
            'verifier_agent',
            'recovery_agent',
            'revision_agent',
        ]
        assert coordinated.agent_trace[0].status == 'success'
        assert 'top_k=6' in coordinated.agent_trace[0].detail
        assert coordinated.agent_trace[1].status == 'success'
        assert 'readme_summary' in coordinated.agent_trace[1].detail
        assert coordinated.agent_trace[2].status == 'success'
        assert coordinated.agent_trace[3].status == 'success'
        assert coordinated.agent_trace[4].status in {'success', 'skipped'}
        assert coordinated.agent_trace[5].status in {'success', 'skipped'}
        assert coordinated.verification_result is not None
        assert coordinated.shared_context['verification_issue_tags'] == coordinated.verification_result.issue_tags
        assert 'revision_applied' in coordinated.shared_context
        assert coordinated.answer_result.answer_mode == 'extractive'
        assert '一个从零手写 Agent 的练习项目' in coordinated.answer_result.answer
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def test_run_multi_agent_answer_uses_router_decision_when_no_hits() -> None:
    original_search_knowledge_base = coordinator_module.search_knowledge_base
    try:
        captured: dict[str, object] = {'top_ks': []}

        def fake_search_knowledge_base(
            query: str,
            top_k: int = 5,
            target_dir: str = 'data/knowledge',
            repo_id: str | None = None,
        ):
            captured['query'] = query
            captured['top_k'] = top_k
            captured['top_ks'].append(top_k)
            captured['repo_id'] = repo_id
            return SearchResult(
                query=query,
                hits=[],
                backend='local',
                repo_count=0,
                document_count=0,
            )

        coordinator_module.search_knowledge_base = fake_search_knowledge_base

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='这个项目怎么启动',
            use_llm=False,
        )

        assert coordinated.route_decision.focus == 'startup'
        assert coordinated.route_decision.retrieval_top_k == 6
        assert captured['top_ks'][0] == 6
        assert coordinated.retrieval_hit_count == 0
        assert coordinated.answer_result.fallback_used is True
        assert coordinated.agent_trace[1].detail.endswith('当前没有召回可用证据。')
        assert coordinated.agent_trace[2].status == 'success'
        assert coordinated.agent_trace[3].status == 'success'
        assert coordinated.agent_trace[4].status in {'success', 'skipped'}
        assert coordinated.agent_trace[5].status in {'success', 'skipped'}
    finally:
        coordinator_module.search_knowledge_base = original_search_knowledge_base


def test_run_multi_agent_answer_retries_retrieval_and_records_shared_context() -> None:
    original_search_knowledge_base = coordinator_module.search_knowledge_base
    try:
        package_json = KeyFileContent(
            path='package.json',
            size_bytes=320,
            content=(
                '{'
                '"name":"demo-web",'
                '"packageManager":"pnpm@9.0.0",'
                '"scripts":{"dev":"vite","start":"node server.js"},'
                '"dependencies":{"react":"^18.0.0"}'
                '}'
            ),
        )
        result = _build_result([package_json], ['package.json'])
        documents = build_knowledge_documents(result)
        temp_dir = Path('data/test_answer_agent_retry')
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        call_count = {'value': 0}

        def flaky_search_knowledge_base(**kwargs):
            call_count['value'] += 1
            if call_count['value'] == 1:
                raise RuntimeError('temporary retrieval error')
            return original_search_knowledge_base(**kwargs)

        coordinator_module.search_knowledge_base = flaky_search_knowledge_base

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='这个项目是做什么的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert call_count['value'] == 2
        assert coordinated.agent_trace[1].used_retry is True
        assert coordinated.agent_trace[1].attempt_count == 2
        assert coordinated.shared_context['question_focus'] == 'overview'
        assert coordinated.shared_context['retrieval_hit_count'] > 0
        assert coordinated.verification_result is not None
        assert isinstance(coordinated.verification_result.issue_tags, list)
    finally:
        coordinator_module.search_knowledge_base = original_search_knowledge_base
        shutil.rmtree(Path('data/test_answer_agent_retry'), ignore_errors=True)


def test_run_multi_agent_answer_skips_low_confidence_code_context() -> None:
    original_run_code_agent_pipeline = coordinator_module._run_code_agent_pipeline
    original_build_answer_result_from_context = coordinator_module._build_answer_result_from_context
    try:
        package_json = KeyFileContent(
            path='package.json',
            size_bytes=320,
            content=(
                '{'
                '"name":"hello-agent",'
                '"packageManager":"pnpm@9.0.0",'
                '"scripts":{"dev":"vite","start":"node server.js"}'
                '}'
            ),
        )
        main_file = KeyFileContent(
            path='main.py',
            size_bytes=120,
            content='def handle_login():\n    return True\n',
        )
        result = _build_result([package_json, main_file], ['package.json', 'main.py'])
        documents = build_knowledge_documents(result)
        temp_dir = Path('data/test_answer_agent_low_confidence')
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        captured: dict[str, object] = {}

        def fake_run_code_agent_pipeline(**kwargs):
            return coordinator_module.CodeAgentExecutionResult(
                outcome=ExecutionOutcome(
                    value=CodeInvestigationResult(
                        focus='implementation',
                        summary='只命中到一个模糊符号。',
                        matched_symbols=['maybe_handler'],
                        trace_steps=[],
                        evidence_locations=[],
                        source_paths=[],
                        called_symbols=[],
                        relevance_score=0.2,
                        confidence_level='low',
                        quality_notes=['问题关键词与命中线索重合很弱'],
                        cache_hit=False,
                    ),
                    attempt_count=1,
                    duration_ms=3,
                    error=None,
                ),
                recovery_attempted=False,
                recovery_improved=False,
                recovery_hit_count=0,
            )

        def fake_build_answer_result_from_context(**kwargs):
            captured['extra_context_lines'] = kwargs.get('extra_context_lines')
            return RepoAnswerResult(
                repo_id=kwargs['repo_id'],
                question=kwargs['question'],
                answer='ok',
                answer_mode='extractive',
                backend=kwargs['backend'],
                fallback_used=False,
                llm_enabled=False,
                llm_attempted=False,
                llm_error=None,
                evidence=[],
            )

        coordinator_module._run_code_agent_pipeline = fake_run_code_agent_pipeline
        coordinator_module._build_answer_result_from_context = fake_build_answer_result_from_context

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='handle_login 是怎么实现的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert coordinated.code_investigation is not None
        assert coordinated.code_investigation.confidence_level == 'low'
        assert captured['extra_context_lines'] is None
        assert coordinated.shared_context['code_context_used'] is False
        assert '未注入回答' in coordinated.agent_trace[3].detail
        assert coordinated.agent_trace[-3].role == 'verifier_agent'
        assert coordinated.agent_trace[-2].role == 'recovery_agent'
        assert coordinated.agent_trace[-1].role == 'revision_agent'
        assert coordinated.verification_result is not None
        assert 'code_confidence_low' in coordinated.verification_result.issue_tags
    finally:
        coordinator_module._run_code_agent_pipeline = original_run_code_agent_pipeline
        coordinator_module._build_answer_result_from_context = original_build_answer_result_from_context
        shutil.rmtree(Path('data/test_answer_agent_low_confidence'), ignore_errors=True)


def test_run_multi_agent_answer_can_recover_low_confidence_code_context() -> None:
    original_run_code_agent_with_retry = coordinator_module._run_code_agent_with_retry
    original_execute_with_retry = coordinator_module.execute_with_retry
    try:
        package_json = KeyFileContent(
            path='package.json',
            size_bytes=320,
            content=(
                '{'
                '"name":"hello-agent",'
                '"packageManager":"pnpm@9.0.0",'
                '"scripts":{"dev":"vite","start":"node server.js"}'
                '}'
            ),
        )
        main_file = KeyFileContent(
            path='main.py',
            size_bytes=120,
            content='def handle_login():\n    return True\n',
        )
        result = _build_result([package_json, main_file], ['package.json', 'main.py'])
        documents = build_knowledge_documents(result)
        temp_dir = Path('data/test_answer_agent_recovery')
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        def fake_run_code_agent_with_retry(**kwargs):
            return ExecutionOutcome(
                value=CodeInvestigationResult(
                    focus='implementation',
                    summary='首次只命中到一个模糊符号。',
                    matched_symbols=['maybe_handler'],
                    trace_steps=[],
                    evidence_locations=[],
                    source_paths=[],
                    called_symbols=[],
                    relevance_score=0.2,
                    confidence_level='low',
                    quality_notes=['问题关键词与命中线索重合很弱'],
                    cache_hit=False,
                ),
                attempt_count=1,
                duration_ms=3,
                error=None,
            )

        def fake_execute_with_retry(func, *args, **kwargs):
            if func is coordinator_module.investigate_code_hits:
                return ExecutionOutcome(
                    value=CodeInvestigationResult(
                        focus='implementation',
                        summary='扩检后已定位到 handle_login 的实现链路。',
                        matched_symbols=['handle_login'],
                        trace_steps=[],
                        evidence_locations=['main.py:L1'],
                        source_paths=['main.py'],
                        called_symbols=['create_session_token'],
                        relevance_score=0.78,
                        confidence_level='high',
                        quality_notes=['问题关键词与命中线索存在重合'],
                        cache_hit=False,
                    ),
                    attempt_count=1,
                    duration_ms=4,
                    error=None,
                )
            return original_execute_with_retry(func, *args, **kwargs)

        coordinator_module._run_code_agent_with_retry = fake_run_code_agent_with_retry
        coordinator_module.execute_with_retry = fake_execute_with_retry

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='handle_login 是怎么实现的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert coordinated.code_investigation is not None
        assert coordinated.code_investigation.recovery_attempted is True
        assert coordinated.code_investigation.recovery_improved is True
        assert coordinated.code_investigation.confidence_level == 'high'
        assert coordinated.shared_context['code_recovery_attempted'] is True
        assert coordinated.shared_context['code_recovery_improved'] is True
        assert coordinated.shared_context['code_context_used'] is True
        assert coordinated.shared_context['verification_verdict'] in {'passed', 'warning', 'failed'}
        assert coordinated.shared_context['verification_issue_tags'] == coordinated.verification_result.issue_tags
        assert '自动扩检恢复并提升结果' in coordinated.agent_trace[2].detail
        assert coordinated.agent_trace[-2].role == 'recovery_agent'
        assert coordinated.agent_trace[-1].role == 'revision_agent'
        assert coordinated.verification_result is not None
    finally:
        coordinator_module._run_code_agent_with_retry = original_run_code_agent_with_retry
        coordinator_module.execute_with_retry = original_execute_with_retry
        shutil.rmtree(Path('data/test_answer_agent_recovery'), ignore_errors=True)


def test_run_multi_agent_answer_tags_sparse_retrieval_failures() -> None:
    original_search_knowledge_base = coordinator_module.search_knowledge_base
    try:
        def fake_search_knowledge_base(query: str, top_k: int = 5, target_dir: str = 'data/knowledge', repo_id: str | None = None):
            return SearchResult(
                query=query,
                hits=[],
                backend='local',
                repo_count=0,
                document_count=0,
            )

        coordinator_module.search_knowledge_base = fake_search_knowledge_base

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='这个项目怎么启动？',
            use_llm=False,
        )

        assert coordinated.verification_result is not None
        assert 'retrieval_sparse' in coordinated.verification_result.issue_tags
        assert 'supporting_lines_missing' in coordinated.verification_result.issue_tags
        assert coordinated.shared_context['verification_issue_tags'] == coordinated.verification_result.issue_tags
        assert coordinated.shared_context['issue_tag_recovery_attempted'] is True
        assert coordinated.agent_trace[-2].role == 'recovery_agent'
    finally:
        coordinator_module.search_knowledge_base = original_search_knowledge_base


def test_run_multi_agent_answer_can_recover_sparse_retrieval_context() -> None:
    original_search_knowledge_base = coordinator_module.search_knowledge_base
    try:
        call_count = {'value': 0}

        def fake_search_knowledge_base(
            query: str,
            top_k: int = 5,
            target_dir: str = 'data/knowledge',
            repo_id: str | None = None,
        ):
            call_count['value'] += 1
            if call_count['value'] == 1:
                return SearchResult(
                    query=query,
                    hits=[],
                    backend='local',
                    repo_count=0,
                    document_count=0,
                )

            document = KnowledgeDocument(
                repo_id=repo_id or 'demo/sample',
                doc_id='readme-1',
                doc_type='readme_summary',
                title='README 概要',
                content='这个项目用于演示一个轻量多 Agent CLI，并通过 README 说明用途。',
                source_path='README.md',
                metadata={'source': 'test'},
            )
            return SearchResult(
                query=query,
                hits=[
                    SearchHit(
                        document=document,
                        score=0.92,
                        snippet='README 说明这个项目用于演示一个轻量多 Agent CLI。',
                    )
                ],
                backend='local',
                repo_count=1,
                document_count=1,
            )

        coordinator_module.search_knowledge_base = fake_search_knowledge_base

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='这个项目是做什么的？',
            use_llm=False,
        )

        assert call_count['value'] >= 2
        assert coordinated.retrieval_hit_count == 1
        assert coordinated.shared_context['issue_tag_recovery_attempted'] is True
        assert coordinated.shared_context['issue_tag_recovery_improved'] is True
        assert coordinated.shared_context['verification_verdict'] in {'passed', 'warning'}
        assert coordinated.answer_result.fallback_used is False
        assert 'README' in coordinated.answer_result.answer
        assert coordinated.agent_trace[-2].role == 'recovery_agent'
        assert '验证结论由' in coordinated.agent_trace[-2].detail
    finally:
        coordinator_module.search_knowledge_base = original_search_knowledge_base
