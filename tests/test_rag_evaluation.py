import shutil
from pathlib import Path

from repoinsight.models.analysis_model import ApiRouteSummary, FunctionSummary, KeyFileContent
from repoinsight.search.evaluation import (
    RagEvaluationCase,
    build_code_rag_eval_cases,
    build_full_rag_eval_cases,
    build_multilang_code_rag_eval_cases,
    evaluate_search_cases,
)
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents
from tests.test_summary_builders import _build_result


def test_evaluate_search_cases_hits_config_and_entrypoint_documents() -> None:
    temp_dir = Path('data/test_rag_eval')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        package_json = KeyFileContent(
            path='package.json',
            size_bytes=320,
            content=(
                '{'
                '"name":"demo-web",'
                '"packageManager":"pnpm@9.0.0",'
                '"scripts":{"dev":"vite","start":"node server.js --token ${OPENAI_API_KEY}"},'
                '"dependencies":{"react":"^18.0.0","redis":"^4.0.0"},'
                '"main":"server.js"'
                '}'
            ),
        )
        app_file = KeyFileContent(
            path='app.py',
            size_bytes=420,
            content=(
                'from fastapi import FastAPI\n'
                'import redis\n\n'
                'app = FastAPI()\n\n'
                'if __name__ == "__main__":\n'
                '    import uvicorn\n'
                '    uvicorn.run(app)\n'
            ),
        )
        pyproject = KeyFileContent(
            path='pyproject.toml',
            size_bytes=240,
            content=(
                '[project]\n'
                'name = "demo-api"\n'
                'requires-python = ">=3.11"\n'
            ),
        )
        result = _build_result(
            [package_json, pyproject, app_file],
            ['package.json', 'pyproject.toml', 'app.py'],
        )

        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        report = evaluate_search_cases(
            [
                RagEvaluationCase(
                    case_id='startup',
                    query='这个项目怎么启动',
                    expected_repo_id='demo/sample',
                    expected_doc_types=['entrypoint_summary'],
                    expected_terms_any=['uvicorn'],
                ),
                RagEvaluationCase(
                    case_id='env-vars',
                    query='这个项目依赖哪些环境变量',
                    expected_repo_id='demo/sample',
                    expected_doc_types=['config_summary'],
                    expected_terms_any=['OPENAI_API_KEY'],
                ),
            ],
            target_dir=str(temp_dir),
        )

        assert report.total_cases == 2
        assert report.passed_cases == 2
        assert report.failed_cases == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_full_rag_eval_cases_includes_code_level_questions() -> None:
    cases = build_full_rag_eval_cases('demo/sample')
    case_ids = {item.case_id for item in cases}

    assert 'repo-purpose' in case_ids
    assert 'implementation-handle-login' in case_ids
    assert 'api-post-login' in case_ids
    assert 'api-post-session' in case_ids


def test_evaluate_search_cases_hits_code_chunk_documents_for_code_questions() -> None:
    temp_dir = Path('data/test_rag_eval_code')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        route_file = KeyFileContent(
            path='api/routes.py',
            size_bytes=420,
            content=(
                'from services import auth_service\n'
                'from repositories import session_repo\n'
                '\n'
                '@router.post("/login")\n'
                'def login(username: str, password: str):\n'
                '    return AuthService().handle_login(username, password)\n'
                '\n'
                '@router.post("/session")\n'
                'def create_session(payload: dict):\n'
                '    return auth_service.login_user(payload)\n'
            ),
        )
        auth_service_file = KeyFileContent(
            path='services/auth_service.py',
            size_bytes=360,
            content=(
                'class AuthService:\n'
                '    def handle_login(self, username, password):\n'
                '        token = create_session_token(username)\n'
                '        return token\n'
                '\n'
                'def login_user(payload: dict):\n'
                '    return session_repo.persist_session(payload["user_id"])\n'
            ),
        )
        session_repo_file = KeyFileContent(
            path='repositories/session_repo.py',
            size_bytes=180,
            content=(
                'def persist_session(user_id):\n'
                '    return {"user_id": user_id, "persisted": True}\n'
            ),
        )
        result = _build_result(
            [route_file, auth_service_file, session_repo_file],
            ['api/routes.py', 'services/auth_service.py', 'repositories/session_repo.py'],
        )
        result.project_profile.function_summaries = [
            FunctionSummary(
                name='handle_login',
                qualified_name='AuthService.handle_login',
                source_path='services/auth_service.py',
                language_scope='python',
                line_start=2,
                line_end=4,
                signature='def handle_login(self, username, password)',
                owner_class='AuthService',
                decorators=[],
                parameters=['self', 'username', 'password'],
                called_symbols=['create_session_token'],
                return_signals=['token'],
                summary='负责校验登录凭证并生成会话 token。',
            ),
            FunctionSummary(
                name='login_user',
                qualified_name='auth_service.login_user',
                source_path='services/auth_service.py',
                language_scope='python',
                line_start=6,
                line_end=7,
                signature='def login_user(payload)',
                owner_class=None,
                decorators=[],
                parameters=['payload'],
                called_symbols=['session_repo.persist_session'],
                return_signals=['session'],
                summary='负责校验用户并调用 session_repo.persist_session。',
            ),
            FunctionSummary(
                name='persist_session',
                qualified_name='session_repo.persist_session',
                source_path='repositories/session_repo.py',
                language_scope='python',
                line_start=1,
                line_end=2,
                signature='def persist_session(user_id)',
                owner_class=None,
                decorators=[],
                parameters=['user_id'],
                called_symbols=[],
                return_signals=['dict'],
                summary='负责把会话信息写入存储。',
            ),
        ]
        result.project_profile.api_route_summaries = [
            ApiRouteSummary(
                route_path='/login',
                http_methods=['POST'],
                source_path='api/routes.py',
                language_scope='python',
                framework='fastapi',
                handler_name='login',
                handler_qualified_name='login',
                owner_class=None,
                line_number=4,
                decorators=['router.post("/login")'],
                called_symbols=['AuthService.handle_login'],
                summary='POST /login 会调用 AuthService.handle_login。',
            ),
            ApiRouteSummary(
                route_path='/session',
                http_methods=['POST'],
                source_path='api/routes.py',
                language_scope='python',
                framework='fastapi',
                handler_name='create_session',
                handler_qualified_name='create_session',
                owner_class=None,
                line_number=8,
                decorators=['router.post("/session")'],
                called_symbols=['auth_service.login_user', 'session_repo.persist_session'],
                summary='POST /session 会调用 auth_service.login_user，并最终进入 session_repo.persist_session。',
            ),
        ]

        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        report = evaluate_search_cases(
            build_code_rag_eval_cases('demo/sample'),
            target_dir=str(temp_dir),
        )

        assert report.total_cases == 3
        assert report.passed_cases == 3
        assert report.failed_cases == 0
        assert any(
            item.case.case_id == 'implementation-handle-login'
            and 'function_body_chunk' in item.hit_doc_types
            for item in report.results
        )
        assert any(
            item.case.case_id == 'api-post-login'
            and 'route_handler_chunk' in item.hit_doc_types
            for item in report.results
        )
        assert any(
            item.case.case_id == 'api-post-session'
            and any(doc_type in {'route_handler_chunk', 'function_body_chunk'} for doc_type in item.hit_doc_types)
            for item in report.results
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_multilang_code_rag_eval_cases_includes_typescript_go_and_rust_questions() -> None:
    cases = build_multilang_code_rag_eval_cases('demo/sample')
    case_ids = {item.case_id for item in cases}

    assert 'ts-implementation-create-session' in case_ids
    assert 'ts-api-post-session' in case_ids
    assert 'go-implementation-build-login-token' in case_ids
    assert 'rust-implementation-persist-session' in case_ids


def test_evaluate_search_cases_hits_multilang_code_chunk_documents() -> None:
    temp_dir = Path('data/test_rag_eval_multilang')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        ts_file = KeyFileContent(
            path='src/routes.ts',
            size_bytes=420,
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
        rust_file = KeyFileContent(
            path='src/lib.rs',
            size_bytes=220,
            content=(
                'pub fn persist_session(user_id: &str) -> String {\n'
                '    write_store(user_id)\n'
                '}\n'
            ),
        )
        result = _build_result(
            [ts_file, go_file, rust_file],
            ['src/routes.ts', 'cmd/server/main.go', 'src/lib.rs'],
        )
        result.project_profile.primary_language = 'TypeScript'
        result.project_profile.languages = ['TypeScript', 'Go', 'Rust']
        result.project_profile.runtimes = ['Node.js']
        result.project_profile.frameworks = ['Express']
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
            FunctionSummary(
                name='persist_session',
                qualified_name='persist_session',
                source_path='src/lib.rs',
                language_scope='rust',
                line_start=1,
                line_end=3,
                signature='pub fn persist_session(user_id: &str) -> String',
                owner_class=None,
                decorators=[],
                parameters=['user_id'],
                called_symbols=['write_store'],
                return_signals=['String'],
                summary='负责把 session 写入底层存储。',
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
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        report = evaluate_search_cases(
            build_multilang_code_rag_eval_cases('demo/sample'),
            target_dir=str(temp_dir),
        )

        assert report.total_cases == 4
        assert report.passed_cases == 4
        assert report.failed_cases == 0
        assert any(
            item.case.case_id == 'ts-implementation-create-session'
            and 'function_body_chunk' in item.hit_doc_types
            for item in report.results
        )
        assert any(
            item.case.case_id == 'ts-api-post-session'
            and 'route_handler_chunk' in item.hit_doc_types
            for item in report.results
        )
        assert any(
            item.case.case_id == 'go-implementation-build-login-token'
            and 'cmd/server/main.go' in item.hit_source_paths
            for item in report.results
        )
        assert any(
            item.case.case_id == 'rust-implementation-persist-session'
            and 'src/lib.rs' in item.hit_source_paths
            for item in report.results
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
