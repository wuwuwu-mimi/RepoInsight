import shutil
from pathlib import Path

from repoinsight.models.analysis_model import ApiRouteSummary, ClassSummary, FunctionSummary, KeyFileContent
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents
from tests.test_summary_builders import _build_result


def test_build_knowledge_documents_include_code_chunk_docs() -> None:
    app_file = KeyFileContent(
        path='app.py',
        size_bytes=512,
        content=(
            'class AuthService:\n'
            '    def handle_login(self, username, password):\n'
            '        token = create_session_token(username)\n'
            '        return token\n'
            '\n'
            '@router.post("/login")\n'
            'def login(username: str, password: str):\n'
            '    return auth_service.handle_login(username, password)\n'
        ),
    )
    result = _build_result([app_file], ['app.py'])
    result.project_profile.function_summaries = [
        FunctionSummary(
            name='handle_login',
            qualified_name='AuthService.handle_login',
            source_path='app.py',
            language_scope='python',
            line_start=2,
            line_end=4,
            signature='def handle_login(self, username, password)',
            owner_class='AuthService',
            decorators=[],
            parameters=['self', 'username', 'password'],
            called_symbols=['create_session_token'],
            return_signals=['token'],
            summary='负责校验登录并生成 token。',
        )
    ]
    result.project_profile.class_summaries = [
        ClassSummary(
            name='AuthService',
            qualified_name='AuthService',
            source_path='app.py',
            language_scope='python',
            line_start=1,
            line_end=4,
            bases=[],
            decorators=[],
            methods=['handle_login'],
            summary='封装登录认证流程。',
        )
    ]
    result.project_profile.api_route_summaries = [
        ApiRouteSummary(
            route_path='/login',
            http_methods=['POST'],
            source_path='app.py',
            language_scope='python',
            framework='fastapi',
            handler_name='login',
            handler_qualified_name='login',
            owner_class=None,
            line_number=6,
            decorators=['router.post("/login")'],
            called_symbols=['AuthService.handle_login'],
            summary='POST /login 会调用 AuthService.handle_login。',
        )
    ]

    documents = build_knowledge_documents(result)

    function_chunk = next(item for item in documents if item.doc_type == 'function_body_chunk')
    class_chunk = next(item for item in documents if item.doc_type == 'class_body_chunk')
    route_chunk = next(item for item in documents if item.doc_type == 'route_handler_chunk')

    assert function_chunk.metadata['qualified_name'] == 'AuthService.handle_login'
    assert function_chunk.metadata['line_start'] == 1
    assert function_chunk.metadata['line_end'] == 6
    assert '源码片段：' in function_chunk.content
    assert 'L2:     def handle_login(self, username, password):' in function_chunk.content

    assert class_chunk.metadata['qualified_name'] == 'AuthService'
    assert 'L1: class AuthService:' in class_chunk.content

    assert route_chunk.metadata['route_path'] == '/login'
    assert route_chunk.metadata['handler_qualified_name'] == 'login'
    assert 'L6: @router.post("/login")' in route_chunk.content


def test_build_knowledge_documents_include_config_chunk_doc() -> None:
    package_json = KeyFileContent(
        path='package.json',
        size_bytes=320,
        content=(
            '{\n'
            '  "name": "demo-web",\n'
            '  "packageManager": "pnpm@9.0.0",\n'
            '  "scripts": {\n'
            '    "dev": "vite",\n'
            '    "start": "node server.js --token ${OPENAI_API_KEY}"\n'
            '  }\n'
            '}\n'
        ),
    )
    result = _build_result([package_json], ['package.json'])

    documents = build_knowledge_documents(result)

    config_chunk = next(item for item in documents if item.doc_type == 'config_chunk')
    assert config_chunk.metadata['config_kind'] == 'package_manager'
    assert '配置片段：' in config_chunk.content
    assert '"packageManager": "pnpm@9.0.0"' in config_chunk.content
    assert 'OPENAI_API_KEY' in config_chunk.content


def test_search_prefers_function_body_chunk_for_precise_implementation_query() -> None:
    temp_dir = Path('data/test_function_body_chunk_search')
    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        app_file = KeyFileContent(
            path='app.py',
            size_bytes=256,
            content=(
                'class AuthService:\n'
                '    def handle_login(self, username, password):\n'
                '        token = create_session_token(username)\n'
                '        return token\n'
            ),
        )
        result = _build_result([app_file], ['app.py'])
        result.project_profile.function_summaries = [
            FunctionSummary(
                name='handle_login',
                qualified_name='AuthService.handle_login',
                source_path='app.py',
                language_scope='python',
                line_start=2,
                line_end=4,
                signature='def handle_login(self, username, password)',
                owner_class='AuthService',
                decorators=[],
                parameters=['self', 'username', 'password'],
                called_symbols=['create_session_token'],
                return_signals=['token'],
                summary='负责校验登录并生成 token。',
            )
        ]
        save_repo_documents('demo/sample', build_knowledge_documents(result), target_dir=str(temp_dir))

        search_result = search_knowledge_base(
            query='AuthService.handle_login 是怎么实现的？',
            top_k=3,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )

        assert search_result.hits
        assert search_result.hits[0].document.doc_type == 'function_body_chunk'
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_search_prefers_route_handler_chunk_for_api_implementation_query() -> None:
    temp_dir = Path('data/test_route_handler_chunk_search')
    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        app_file = KeyFileContent(
            path='app.py',
            size_bytes=256,
            content=(
                '@router.post("/login")\n'
                'def login(username: str, password: str):\n'
                '    return auth_service.handle_login(username, password)\n'
            ),
        )
        result = _build_result([app_file], ['app.py'])
        result.project_profile.api_route_summaries = [
            ApiRouteSummary(
                route_path='/login',
                http_methods=['POST'],
                source_path='app.py',
                language_scope='python',
                framework='fastapi',
                handler_name='login',
                handler_qualified_name='login',
                owner_class=None,
                line_number=1,
                decorators=['router.post("/login")'],
                called_symbols=['auth_service.handle_login'],
                summary='POST /login 会调用 auth_service.handle_login。',
            )
        ]
        save_repo_documents('demo/sample', build_knowledge_documents(result), target_dir=str(temp_dir))

        search_result = search_knowledge_base(
            query='POST /login 是怎么实现的？',
            top_k=3,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )

        assert search_result.hits
        assert search_result.hits[0].document.doc_type == 'route_handler_chunk'
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_knowledge_documents_include_typescript_and_go_chunk_docs() -> None:
    ts_file = KeyFileContent(
        path='src/routes.ts',
        size_bytes=360,
        content=(
            "import express from 'express'\n"
            '\n'
            'export const createSession = async (payload: SessionPayload) => {\n'
            '  return sessionService.create(payload)\n'
            '}\n'
            '\n'
            "router.post('/session', createSession)\n"
        ),
    )
    go_file = KeyFileContent(
        path='cmd/server/main.go',
        size_bytes=220,
        content=(
            'package main\n'
            '\n'
            'func BuildLoginToken(userID string) string {\n'
            '    return signToken(userID)\n'
            '}\n'
        ),
    )
    result = _build_result([ts_file, go_file], ['src/routes.ts', 'cmd/server/main.go'])
    result.project_profile.primary_language = 'TypeScript'
    result.project_profile.languages = ['TypeScript', 'Go']
    result.project_profile.function_summaries = [
        FunctionSummary(
            name='createSession',
            qualified_name='createSession',
            source_path='src/routes.ts',
            language_scope='typescript',
            line_start=3,
            line_end=5,
            signature='export const createSession = async (payload: SessionPayload) =>',
            owner_class=None,
            is_async=True,
            decorators=[],
            parameters=['payload'],
            called_symbols=['sessionService.create'],
            return_signals=['session'],
            summary='负责接收 session 请求并调用 sessionService.create。',
        ),
        FunctionSummary(
            name='BuildLoginToken',
            qualified_name='BuildLoginToken',
            source_path='cmd/server/main.go',
            language_scope='go',
            line_start=3,
            line_end=5,
            signature='func BuildLoginToken(userID string) string',
            owner_class=None,
            decorators=[],
            parameters=['userID'],
            called_symbols=['signToken'],
            return_signals=['string'],
            summary='负责生成登录 token，并委托 signToken 完成签名。',
        ),
    ]
    result.project_profile.api_route_summaries = [
        ApiRouteSummary(
            route_path='/session',
            http_methods=['POST'],
            source_path='src/routes.ts',
            language_scope='typescript',
            framework='express',
            handler_name='createSession',
            handler_qualified_name='createSession',
            owner_class=None,
            line_number=7,
            decorators=["router.post('/session')"],
            called_symbols=['sessionService.create'],
            summary='POST /session 会调用 createSession，并进一步进入 sessionService.create。',
        )
    ]

    documents = build_knowledge_documents(result)

    ts_function_chunk = next(
        item
        for item in documents
        if item.doc_type == 'function_body_chunk' and item.source_path == 'src/routes.ts'
    )
    go_function_chunk = next(
        item
        for item in documents
        if item.doc_type == 'function_body_chunk' and item.source_path == 'cmd/server/main.go'
    )
    ts_route_chunk = next(
        item
        for item in documents
        if item.doc_type == 'route_handler_chunk' and item.source_path == 'src/routes.ts'
    )

    assert ts_function_chunk.metadata['language_scope'] == 'typescript'
    assert 'L3: export const createSession = async (payload: SessionPayload) => {' in ts_function_chunk.content
    assert go_function_chunk.metadata['language_scope'] == 'go'
    assert 'L3: func BuildLoginToken(userID string) string {' in go_function_chunk.content
    assert ts_route_chunk.metadata['language_scope'] == 'typescript'
    assert "L7: router.post('/session', createSession)" in ts_route_chunk.content
