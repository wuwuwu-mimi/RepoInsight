import shutil
from pathlib import Path

import repoinsight.agents.code_agent as code_agent_module
from repoinsight.agents.answer_coordinator import run_multi_agent_answer
from repoinsight.agents.code_agent import (
    build_code_investigation_context_lines,
    clear_code_agent_cache,
    investigate_code_hits,
)
from repoinsight.answer.service import _prioritize_hits_for_focus
from repoinsight.models.analysis_model import AnalysisRunResult, ApiRouteSummary, ClassSummary, FunctionSummary, ProjectProfile
from repoinsight.models.rag_model import SearchHit
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents
from tests.test_code_index_mvp import _build_repo_info, _build_scan_result



def _build_single_file_analysis_result() -> AnalysisRunResult:
    return AnalysisRunResult(
        repo_info=_build_repo_info(),
        clone_path='E:/PythonProject/RepoInsight/clones/demo__sample',
        scan_result=_build_scan_result(['app.py']),
        key_file_contents=[],
        project_profile=ProjectProfile(
            primary_language='Python',
            languages=['Python'],
            api_route_summaries=[
                ApiRouteSummary(
                    route_path='/login',
                    http_methods=['POST'],
                    source_path='app.py',
                    language_scope='python',
                    framework='fastapi',
                    handler_name='handle_login',
                    handler_qualified_name='AuthService.handle_login',
                    owner_class='AuthService',
                    line_number=18,
                    decorators=['router.post("/login")'],
                    called_symbols=['verify_password', 'create_session_token'],
                    summary='接口 POST /login 由 AuthService.handle_login 处理，并会调用 verify_password、create_session_token。',
                )
            ],
            function_summaries=[
                FunctionSummary(
                    name='handle_login',
                    qualified_name='AuthService.handle_login',
                    source_path='app.py',
                    language_scope='python',
                    line_start=18,
                    line_end=33,
                    signature='def AuthService.handle_login(self, username, password)',
                    owner_class='AuthService',
                    decorators=[],
                    parameters=['self', 'username', 'password'],
                    called_symbols=['verify_password', 'create_session_token'],
                    return_signals=['token'],
                    summary='方法 AuthService.handle_login 负责校验登录凭证，并创建会话 token。',
                ),
                FunctionSummary(
                    name='verify_password',
                    qualified_name='verify_password',
                    source_path='app.py',
                    language_scope='python',
                    line_start=35,
                    line_end=37,
                    signature='def verify_password(username, password)',
                    owner_class=None,
                    decorators=[],
                    parameters=['username', 'password'],
                    called_symbols=[],
                    return_signals=['bool'],
                    summary='函数 verify_password 负责校验账号密码。',
                ),
                FunctionSummary(
                    name='create_session_token',
                    qualified_name='create_session_token',
                    source_path='app.py',
                    language_scope='python',
                    line_start=40,
                    line_end=42,
                    signature='def create_session_token(username)',
                    owner_class=None,
                    decorators=[],
                    parameters=['username'],
                    called_symbols=[],
                    return_signals=['token'],
                    summary='函数 create_session_token 负责生成登录态 token。',
                ),
            ],
            class_summaries=[
                ClassSummary(
                    name='AuthService',
                    qualified_name='AuthService',
                    source_path='app.py',
                    language_scope='python',
                    line_start=2,
                    line_end=40,
                    bases=[],
                    decorators=[],
                    methods=['handle_login'],
                    summary='类 AuthService 封装认证与登录流程。',
                )
            ],
        ),
    )



def _build_cross_file_analysis_result() -> AnalysisRunResult:
    return AnalysisRunResult(
        repo_info=_build_repo_info(),
        clone_path='E:/PythonProject/RepoInsight/clones/demo__sample',
        scan_result=_build_scan_result(['api/routes.py', 'services/auth_service.py', 'repositories/session_repo.py']),
        key_file_contents=[],
        project_profile=ProjectProfile(
            primary_language='Python',
            languages=['Python'],
            api_route_summaries=[
                ApiRouteSummary(
                    route_path='/session',
                    http_methods=['POST'],
                    source_path='api/routes.py',
                    language_scope='python',
                    framework='fastapi',
                    handler_name='create_session',
                    handler_qualified_name='create_session',
                    owner_class=None,
                    line_number=5,
                    decorators=['router.post("/session")'],
                    called_symbols=['auth_service.login_user'],
                    summary='接口 POST /session 会调用 auth_service.login_user 完成登录流程。',
                )
            ],
            function_summaries=[
                FunctionSummary(
                    name='create_session',
                    qualified_name='create_session',
                    source_path='api/routes.py',
                    language_scope='python',
                    line_start=5,
                    line_end=8,
                    signature='def create_session(payload)',
                    owner_class=None,
                    decorators=[],
                    parameters=['payload'],
                    called_symbols=['auth_service.login_user'],
                    return_signals=['session'],
                    summary='函数 create_session 负责接收请求并转交给 auth_service.login_user。',
                ),
                FunctionSummary(
                    name='login_user',
                    qualified_name='auth_service.login_user',
                    source_path='services/auth_service.py',
                    language_scope='python',
                    line_start=3,
                    line_end=7,
                    signature='def login_user(payload)',
                    owner_class=None,
                    decorators=[],
                    parameters=['payload'],
                    called_symbols=['session_repo.persist_session'],
                    return_signals=['session'],
                    summary='函数 auth_service.login_user 负责校验用户并调用 session_repo.persist_session。',
                ),
                FunctionSummary(
                    name='persist_session',
                    qualified_name='session_repo.persist_session',
                    source_path='repositories/session_repo.py',
                    language_scope='python',
                    line_start=2,
                    line_end=4,
                    signature='def persist_session(user_id)',
                    owner_class=None,
                    decorators=[],
                    parameters=['user_id'],
                    called_symbols=[],
                    return_signals=['session'],
                    summary='函数 session_repo.persist_session 负责把会话信息写入存储。',
                ),
            ],
            class_summaries=[],
        ),
    )



def _build_single_file_clone_repo(temp_root: Path) -> Path:
    clone_root = temp_root / 'clone' / 'demo' / 'sample'
    clone_root.mkdir(parents=True, exist_ok=True)
    (clone_root / '.git').mkdir(exist_ok=True)
    (clone_root / 'app.py').write_text(
        (
            'class AuthService:\n'
            '    def __init__(self):\n'
            '        pass\n'
            '\n'
            '    def other(self):\n'
            '        return None\n'
            '\n'
            '    def another(self):\n'
            '        return None\n'
            '\n'
            '    def more(self):\n'
            '        return None\n'
            '\n'
            '    def helper(self):\n'
            '        return None\n'
            '\n'
            '    def handle_login(self, username, password):\n'
            '        if not verify_password(username, password):\n'
            "            raise ValueError('invalid')\n"
            '        token = create_session_token(username)\n'
            '        return token\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n'
            'def verify_password(username, password):\n'
            "    return username == 'demo' and password == 'secret'\n"
            '\n'
            '\n'
            'def create_session_token(username):\n'
            "    return f'token-{username}'\n"
        ),
        encoding='utf-8',
    )
    return clone_root



def _build_cross_file_clone_repo(temp_root: Path) -> Path:
    clone_root = temp_root / 'clone' / 'demo' / 'sample'
    (clone_root / '.git').mkdir(parents=True, exist_ok=True)
    (clone_root / 'api').mkdir(parents=True, exist_ok=True)
    (clone_root / 'services').mkdir(parents=True, exist_ok=True)
    (clone_root / 'repositories').mkdir(parents=True, exist_ok=True)
    (clone_root / 'api' / 'routes.py').write_text(
        (
            'from services import auth_service\n\n'
            'def helper():\n'
            '    return None\n\n'
            'def create_session(payload):\n'
            '    return auth_service.login_user(payload)\n'
        ),
        encoding='utf-8',
    )
    (clone_root / 'services' / 'auth_service.py').write_text(
        (
            'from repositories import session_repo\n\n'
            'def login_user(payload):\n'
            "    user_id = payload['user_id']\n"
            '    return session_repo.persist_session(user_id)\n'
        ),
        encoding='utf-8',
    )
    (clone_root / 'repositories' / 'session_repo.py').write_text(
        (
            'def persist_session(user_id):\n'
            "    return {'user_id': user_id, 'session': 'ok'}\n"
        ),
        encoding='utf-8',
    )
    return clone_root



def test_investigate_code_hits_extracts_route_symbol_and_calls() -> None:
    temp_dir = Path('data/test_code_agent_extract')
    clone_root = Path('data/test_code_agent_extract_repo')
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(clone_root, ignore_errors=True)
    original_get_clone_path = code_agent_module.get_clone_path
    try:
        clear_code_agent_cache()
        real_clone_path = _build_single_file_clone_repo(clone_root)
        code_agent_module.get_clone_path = lambda repo_id: real_clone_path
        documents = build_knowledge_documents(_build_single_file_analysis_result())
        save_repo_documents('demo/sample', documents, target_dir=str(temp_dir))

        search_result = search_knowledge_base(
            query='POST /login 是怎么实现的',
            top_k=4,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        prioritized_hits = _prioritize_hits_for_focus(search_result.hits, 'api')

        investigation = investigate_code_hits(
            'POST /login 是怎么实现的',
            prioritized_hits,
            'api',
            repo_id='demo/sample',
            target_dir=str(temp_dir),
        )

        assert investigation is not None
        assert investigation.focus == 'api'
        assert '/login' in investigation.matched_routes
        assert 'AuthService.handle_login' in investigation.matched_symbols
        assert 'app.py' in investigation.source_paths
        assert any(location.startswith('app.py:L18') for location in investigation.evidence_locations)
        assert 'verify_password' in investigation.called_symbols
        assert any(step.label == 'verify_password' for step in investigation.trace_steps)
        assert any(step.snippet and 'create_session_token' in step.snippet for step in investigation.trace_steps)
        assert any('POST /login' in note for note in investigation.implementation_notes)
        assert investigation.relevance_score >= 0.4
        assert investigation.confidence_level in {'medium', 'high'}
        assert investigation.quality_notes

        context_lines = build_code_investigation_context_lines(investigation)
        assert context_lines[0].startswith('[code_agent] 已定位到接口 /login 的实现入口')
        assert any('app.py:L18' in line for line in context_lines)
    finally:
        clear_code_agent_cache()
        code_agent_module.get_clone_path = original_get_clone_path
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_root, ignore_errors=True)



def test_investigate_code_hits_can_follow_cross_file_call_chain() -> None:
    temp_dir = Path('data/test_code_agent_cross_file')
    clone_root = Path('data/test_code_agent_cross_file_repo')
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(clone_root, ignore_errors=True)
    original_get_clone_path = code_agent_module.get_clone_path
    try:
        clear_code_agent_cache()
        real_clone_path = _build_cross_file_clone_repo(clone_root)
        code_agent_module.get_clone_path = lambda repo_id: real_clone_path
        documents = build_knowledge_documents(_build_cross_file_analysis_result())
        save_repo_documents('demo/sample', documents, target_dir=str(temp_dir))

        search_result = search_knowledge_base(
            query='POST /session 是怎么实现的',
            top_k=1,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        prioritized_hits = _prioritize_hits_for_focus(search_result.hits, 'api')

        investigation = investigate_code_hits(
            'POST /session 是怎么实现的',
            prioritized_hits,
            'api',
            repo_id='demo/sample',
            target_dir=str(temp_dir),
            max_follow_depth=3,
        )

        assert investigation is not None
        assert any(path == 'api/routes.py' for path in investigation.source_paths)
        assert any(path == 'services/auth_service.py' for path in investigation.source_paths)
        assert any(path == 'repositories/session_repo.py' for path in investigation.source_paths)
        assert 'auth_service.login_user' in investigation.matched_symbols
        assert 'session_repo.persist_session' in investigation.matched_symbols
        assert any(step.depth == 2 and step.label == 'session_repo.persist_session' for step in investigation.trace_steps)
        assert any(step.parent_label == 'auth_service.login_user' for step in investigation.trace_steps)
        assert any(step.location == 'services/auth_service.py:L3-L7' for step in investigation.trace_steps)
        assert '当前已展开到第 2 层调用链' in investigation.summary
        assert any(chain == 'POST /session -> auth_service.login_user -> session_repo.persist_session' for chain in investigation.relation_chains)
        assert investigation.relevance_score >= 0.4
        assert investigation.confidence_level in {'medium', 'high'}

        context_lines = build_code_investigation_context_lines(investigation)
        assert any('代表性关系链：POST /session -> auth_service.login_user -> session_repo.persist_session' in line for line in context_lines)
        assert any('auth_service.login_user <- POST /session' in line for line in context_lines)
        assert any('session_repo.persist_session <- auth_service.login_user' in line for line in context_lines)
    finally:
        clear_code_agent_cache()
        code_agent_module.get_clone_path = original_get_clone_path
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_root, ignore_errors=True)


def test_investigate_code_hits_can_follow_class_methods_via_relation_metadata() -> None:
    temp_dir = Path('data/test_code_agent_class_follow')
    clone_root = Path('data/test_code_agent_class_follow_repo')
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(clone_root, ignore_errors=True)
    original_get_clone_path = code_agent_module.get_clone_path
    try:
        clear_code_agent_cache()
        real_clone_path = _build_single_file_clone_repo(clone_root)
        code_agent_module.get_clone_path = lambda repo_id: real_clone_path
        documents = build_knowledge_documents(_build_single_file_analysis_result())
        save_repo_documents('demo/sample', documents, target_dir=str(temp_dir))
        class_document = next(item for item in documents if item.doc_type == 'class_summary')

        investigation = investigate_code_hits(
            'AuthService 的登录流程在哪里实现？',
            [SearchHit(document=class_document, score=1.0, snippet='AuthService 负责登录流程')],
            'implementation',
            repo_id='demo/sample',
            target_dir=str(temp_dir),
            max_follow_depth=2,
        )

        assert investigation is not None
        assert investigation.trace_steps[0].label == 'AuthService'
        assert any(step.label == 'AuthService.handle_login' for step in investigation.trace_steps)
        assert any(
            step.parent_label == 'AuthService' and step.label == 'AuthService.handle_login'
            for step in investigation.trace_steps
        )
        assert any(chain.startswith('AuthService -> AuthService.handle_login') for chain in investigation.relation_chains)
        assert 'verify_password' in investigation.called_symbols
    finally:
        clear_code_agent_cache()
        code_agent_module.get_clone_path = original_get_clone_path
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_root, ignore_errors=True)



def test_run_multi_agent_answer_runs_code_agent_for_implementation_questions() -> None:
    temp_dir = Path('data/test_code_agent_coordinator')
    clone_root = Path('data/test_code_agent_coordinator_repo')
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(clone_root, ignore_errors=True)
    original_get_clone_path = code_agent_module.get_clone_path
    try:
        clear_code_agent_cache()
        real_clone_path = _build_single_file_clone_repo(clone_root)
        code_agent_module.get_clone_path = lambda repo_id: real_clone_path
        documents = build_knowledge_documents(_build_single_file_analysis_result())
        save_repo_documents('demo/sample', documents, target_dir=str(temp_dir))

        coordinated = run_multi_agent_answer(
            repo_id='demo/sample',
            question='handle_login 是怎么实现的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert coordinated.route_decision.focus == 'implementation'
        assert coordinated.code_investigation is not None
        assert coordinated.code_investigation.matched_symbols[0] == 'AuthService.handle_login'
        assert coordinated.code_investigation.trace_steps
        assert coordinated.code_investigation.trace_steps[0].location is not None
        assert coordinated.shared_context['code_trace_count'] >= 1
        assert coordinated.shared_context['code_context_used'] is True
        assert [item.role for item in coordinated.agent_trace] == [
            'router_agent',
            'retrieval_agent',
            'code_agent',
            'synthesis_agent',
            'verifier_agent',
            'recovery_agent',
            'revision_agent',
        ]
        assert coordinated.agent_trace[2].detail is not None
        assert 'AuthService.handle_login' in coordinated.agent_trace[2].detail
        assert coordinated.verification_result is not None
        assert '实现链路：' in coordinated.answer_result.answer
        assert '补充线索：' in coordinated.answer_result.answer
        assert 'AuthService.handle_login' in coordinated.answer_result.answer
        assert 'create_session_token' in coordinated.answer_result.answer
    finally:
        clear_code_agent_cache()
        code_agent_module.get_clone_path = original_get_clone_path
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_root, ignore_errors=True)


def test_investigate_code_hits_uses_cache_on_second_call() -> None:
    temp_dir = Path('data/test_code_agent_cache')
    clone_root = Path('data/test_code_agent_cache_repo')
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(clone_root, ignore_errors=True)
    original_get_clone_path = code_agent_module.get_clone_path
    try:
        clear_code_agent_cache()
        real_clone_path = _build_single_file_clone_repo(clone_root)
        code_agent_module.get_clone_path = lambda repo_id: real_clone_path
        documents = build_knowledge_documents(_build_single_file_analysis_result())
        save_repo_documents('demo/sample', documents, target_dir=str(temp_dir))

        search_result = search_knowledge_base(
            query='handle_login 是怎么实现的？',
            top_k=4,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        prioritized_hits = _prioritize_hits_for_focus(search_result.hits, 'implementation')

        first = investigate_code_hits(
            'handle_login 是怎么实现的？',
            prioritized_hits,
            'implementation',
            repo_id='demo/sample',
            target_dir=str(temp_dir),
        )
        second = investigate_code_hits(
            'handle_login 是怎么实现的？',
            prioritized_hits,
            'implementation',
            repo_id='demo/sample',
            target_dir=str(temp_dir),
        )

        assert first is not None
        assert second is not None
        assert first.cache_hit is False
        assert second.cache_hit is True
        assert second.matched_symbols == first.matched_symbols
        assert second.relevance_score == first.relevance_score
    finally:
        clear_code_agent_cache()
        code_agent_module.get_clone_path = original_get_clone_path
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_root, ignore_errors=True)

def test_investigate_code_hits_can_follow_handler_from_api_metadata_only() -> None:
    temp_dir = Path('data/test_code_agent_api_follow')
    clone_root = Path('data/test_code_agent_api_follow_repo')
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(clone_root, ignore_errors=True)
    original_get_clone_path = code_agent_module.get_clone_path
    try:
        clear_code_agent_cache()
        real_clone_path = _build_single_file_clone_repo(clone_root)
        code_agent_module.get_clone_path = lambda repo_id: real_clone_path
        documents = build_knowledge_documents(_build_single_file_analysis_result())
        save_repo_documents('demo/sample', documents, target_dir=str(temp_dir))

        search_result = search_knowledge_base(
            query='POST /login 是怎么实现的',
            top_k=1,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        api_only_hits = [hit for hit in search_result.hits if hit.document.doc_type == 'api_route_summary']

        investigation = investigate_code_hits(
            'POST /login 是怎么实现的',
            api_only_hits,
            'api',
            repo_id='demo/sample',
            target_dir=str(temp_dir),
            max_follow_depth=2,
        )

        assert investigation is not None
        assert 'AuthService.handle_login' in investigation.matched_symbols
        assert any(step.label == 'AuthService.handle_login' for step in investigation.trace_steps)
        assert any(step.parent_label == 'POST /login' for step in investigation.trace_steps)
    finally:
        clear_code_agent_cache()
        code_agent_module.get_clone_path = original_get_clone_path
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_root, ignore_errors=True)
