import shutil
from pathlib import Path

import repoinsight.analyze.code_index.python_indexer as python_indexer_module
from repoinsight.answer.service import answer_repo_question
from repoinsight.analyze.project_profile_inference import infer_project_profile
from repoinsight.models.analysis_model import (
    AnalysisRunResult,
    ApiRouteSummary,
    ClassSummary,
    CodeEntity,
    CodeRelationEdge,
    FunctionSummary,
    KeyFileContent,
    ProjectProfile,
)
from repoinsight.models.file_model import FileEntry, ScanResult, ScanStats
from repoinsight.models.repo_model import RepoInfo, RepoModel
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents


def _build_file_entry(path: str) -> FileEntry:
    name = path.split('/')[-1]
    parent_dir = '/'.join(path.split('/')[:-1])
    extension = f'.{name.split(".")[-1]}' if '.' in name else None
    return FileEntry(
        path=path,
        name=name,
        extension=extension,
        size_bytes=256,
        parent_dir=parent_dir,
        is_key_file=True,
    )


def _build_repo_info() -> RepoInfo:
    return RepoInfo(
        repo_model=RepoModel(
            owner='demo',
            name='sample',
            full_name='demo/sample',
            html_url='https://github.com/demo/sample',
            default_branch='main',
            primary_language='Python',
            languages={'Python': 1200, 'TypeScript': 900},
        ),
        readme='demo sample repo',
    )


def _build_scan_result(paths: list[str]) -> ScanResult:
    entries = [_build_file_entry(path) for path in paths]
    return ScanResult(
        root_path='E:/PythonProject/RepoInsight/clones/demo__sample',
        all_files=entries,
        key_files=entries,
        tree_preview=paths,
        stats=ScanStats(total_seen=len(paths), kept_count=len(paths), key_file_count=len(paths)),
    )


def test_infer_project_profile_extracts_api_routes_for_python_and_javascript() -> None:
    repo_info = _build_repo_info()
    scan_result = _build_scan_result(['app.py', 'src/routes.js'])
    key_file_contents = [
        KeyFileContent(
            path='app.py',
            size_bytes=640,
            content=(
                'from fastapi import APIRouter\n\n'
                'router = APIRouter()\n\n'
                '@router.post("/login")\n'
                'async def login(username: str, password: str):\n'
                '    token = issue_token(username)\n'
                '    return token\n'
            ),
        ),
        KeyFileContent(
            path='src/routes.js',
            size_bytes=720,
            content=(
                'function healthHandler(req, res) {\n'
                '  return service.health()\n'
                '}\n\n'
                "router.get('/health', healthHandler)\n"
            ),
        ),
    ]

    profile = infer_project_profile(repo_info, scan_result, key_file_contents)

    routes = {(tuple(item.http_methods), item.route_path, item.handler_qualified_name) for item in profile.api_route_summaries}
    assert (('POST',), '/login', 'login') in routes
    assert (('GET',), '/health', 'healthHandler') in routes

    python_route = next(item for item in profile.api_route_summaries if item.route_path == '/login')
    assert python_route.framework == 'fastapi'
    assert 'issue_token' in python_route.called_symbols

    js_route = next(item for item in profile.api_route_summaries if item.route_path == '/health')
    assert js_route.framework == 'express'
    assert 'service.health' in js_route.called_symbols


def test_build_knowledge_documents_includes_api_route_summary_docs() -> None:
    repo_info = _build_repo_info()
    scan_result = _build_scan_result(['app.py'])
    profile = ProjectProfile(
        primary_language='Python',
        languages=['Python'],
        entrypoints=['app.py'],
        code_entities=[
            CodeEntity(
                entity_kind='api_route',
                name='POST /login',
                qualified_name='AuthService.handle_login',
                source_path='app.py',
                language_scope='python',
                location='app.py:L12',
                tags=['route', 'fastapi'],
            )
        ],
        code_relation_edges=[
            CodeRelationEdge(
                source_ref='POST /login',
                target_ref='AuthService.handle_login',
                relation_type='handle_route',
                source_path='app.py',
                line_number=12,
            )
        ],
        function_summaries=[
            FunctionSummary(
                name='handle_login',
                qualified_name='AuthService.handle_login',
                source_path='app.py',
                language_scope='python',
                line_start=10,
                line_end=20,
                signature='def AuthService.handle_login(self, username, password)',
                owner_class='AuthService',
                decorators=[],
                parameters=['self', 'username', 'password'],
                called_symbols=['verify_password', 'create_session_token'],
                return_signals=['token'],
                summary='方法 AuthService.handle_login 负责校验登录并生成会话 token。',
            )
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
                line_number=12,
                decorators=['router.post("/login")'],
                called_symbols=['verify_password', 'create_session_token'],
                summary='接口 POST /login 由 AuthService.handle_login 处理，并会调用 verify_password、create_session_token。',
            )
        ],
    )
    result = AnalysisRunResult(
        repo_info=repo_info,
        clone_path='E:/PythonProject/RepoInsight/clones/demo__sample',
        scan_result=scan_result,
        key_file_contents=[],
        project_profile=profile,
    )

    documents = build_knowledge_documents(result)

    api_document = next(item for item in documents if item.doc_type == 'api_route_summary')
    assert '路由路径：/login' in api_document.content
    assert 'HTTP 方法：POST' in api_document.content
    assert api_document.metadata['route_path'] == '/login'
    assert api_document.metadata['handler_qualified_name'] == 'AuthService.handle_login'
    assert 'POST /login' in api_document.metadata['code_entity_names']
    assert 'AuthService.handle_login' in api_document.metadata['code_entity_refs']
    assert api_document.metadata['code_relation_sources'][:2] == ['app.py', 'POST /login']
    assert api_document.metadata['code_relation_targets'][:2] == ['POST /login', 'AuthService.handle_login']
    assert api_document.metadata['code_relation_types'][:2] == ['contain_route', 'handle_route']
    assert 'handle_route' in api_document.metadata['code_relation_types']


def test_search_and_answer_prefer_api_route_summary_for_api_questions() -> None:
    temp_dir = Path('data/test_api_route_mvp')
    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        result = AnalysisRunResult(
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
                    )
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
        knowledge_documents = build_knowledge_documents(result)
        save_repo_documents('demo/sample', knowledge_documents, target_dir=str(temp_dir))

        search_result = search_knowledge_base(
            query='POST /login 是怎么实现的',
            top_k=3,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        assert search_result.hits
        assert search_result.hits[0].document.doc_type == 'api_route_summary'

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='POST /login 是怎么实现的',
            top_k=3,
            target_dir=str(temp_dir),
            use_llm=False,
        )
        assert answer_result.answer_mode == 'extractive'
        assert '/login' in answer_result.answer
        assert 'handle_login' in answer_result.answer
        assert 'create_session_token' in answer_result.answer or 'verify_password' in answer_result.answer
        assert answer_result.evidence
        assert answer_result.evidence[0].doc_type == 'api_route_summary'
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_infer_project_profile_extracts_multiline_typescript_routes_with_tree_sitter() -> None:
    repo_info = _build_repo_info()
    scan_result = _build_scan_result(['src/routes.ts'])
    key_file_contents = [
        KeyFileContent(
            path='src/routes.ts',
            size_bytes=960,
            content=(
                "import express from 'express'\n\n"
                'interface LoginRequest {\n'
                '  username: string\n'
                '}\n\n'
                'export const createSession = async (payload: LoginRequest) => {\n'
                '  return authService.create(payload)\n'
                '}\n\n'
                'router.post(\n'
                "  '/session',\n"
                '  createSession,\n'
                ')\n'
            ),
        )
    ]

    profile = infer_project_profile(repo_info, scan_result, key_file_contents)

    assert any(item.name == 'LoginRequest' and item.symbol_type == 'interface' for item in profile.code_symbols)
    assert any(item.target == 'express' and item.relation_type == 'import' for item in profile.module_relations)

    route = next(item for item in profile.api_route_summaries if item.route_path == '/session')
    assert route.framework == 'express'
    assert route.handler_qualified_name == 'createSession'
    assert 'authService.create' in route.called_symbols

    function_summary = next(item for item in profile.function_summaries if item.qualified_name == 'createSession')
    assert function_summary.language_scope == 'typescript'
    assert 'payload' in function_summary.parameters


def test_infer_project_profile_prefers_python_tree_sitter_before_ast_fallback() -> None:
    repo_info = _build_repo_info()
    scan_result = _build_scan_result(['app.py'])
    key_file_contents = [
        KeyFileContent(
            path='app.py',
            size_bytes=880,
            content=(
                'from fastapi import APIRouter\n\n'
                'router = APIRouter()\n\n'
                '@router.post("/login")\n'
                'async def login(username: str, password: str) -> str:\n'
                '    token = issue_token(username)\n'
                '    return token\n\n'
                'class AuthService(BaseService):\n'
                '    @classmethod\n'
                '    async def handle(cls, payload, *args, **kwargs):\n'
                '        value = process(payload)\n'
                '        return value\n'
            ),
        )
    ]

    original_visit = python_indexer_module._PythonModuleIndexer.visit
    original_regex = python_indexer_module._extract_python_index_with_regex
    try:
        python_indexer_module._PythonModuleIndexer.visit = lambda self, tree: (_ for _ in ()).throw(
            AssertionError('should not fall back to ast module indexer')
        )
        python_indexer_module._extract_python_index_with_regex = lambda file_content, accumulator: (_ for _ in ()).throw(
            AssertionError('should not fall back to regex extractor')
        )

        profile = infer_project_profile(repo_info, scan_result, key_file_contents)
    finally:
        python_indexer_module._PythonModuleIndexer.visit = original_visit
        python_indexer_module._extract_python_index_with_regex = original_regex

    assert any(item.target == 'fastapi' for item in profile.module_relations)
    assert any(item.name == 'AuthService' for item in profile.class_summaries)

    route = next(item for item in profile.api_route_summaries if item.route_path == '/login')
    assert route.framework == 'fastapi'
    assert route.handler_qualified_name == 'login'
    assert 'issue_token' in route.called_symbols

    method_summary = next(item for item in profile.function_summaries if item.qualified_name == 'AuthService.handle')
    assert method_summary.is_async is True
    assert '*args' in method_summary.parameters
    assert '**kwargs' in method_summary.parameters
    assert 'process' in method_summary.called_symbols
