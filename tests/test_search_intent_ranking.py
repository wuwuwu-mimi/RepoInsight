import shutil
from pathlib import Path

from repoinsight.models.analysis_model import CodeSymbol, KeyFileContent, ModuleRelation, SubprojectSummary
from repoinsight.models.rag_model import KnowledgeDocument, SearchHit
import repoinsight.search.service as search_service_module
import repoinsight.storage.chroma_store as chroma_store_module
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents
from tests.test_summary_builders import _build_result


def test_search_knowledge_base_applies_intent_aware_ranking() -> None:
    temp_dir = Path('data/test_search_intent')
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
                '"dependencies":{"react":"^18.0.0"},'
                '"workspaces":["packages/*","services/*"],'
                '"main":"server.js"'
                '}'
            ),
        )
        app_file = KeyFileContent(
            path='services/api/app.py',
            size_bytes=420,
            content=(
                'from fastapi import FastAPI\n'
                'from api.router import router\n\n'
                'app = FastAPI()\n\n'
                'def create_app():\n'
                '    return app\n\n'
                'app.include_router(router)\n\n'
                'if __name__ == "__main__":\n'
                '    import uvicorn\n'
                '    uvicorn.run(app)\n'
            ),
        )
        pyproject = KeyFileContent(
            path='services/api/pyproject.toml',
            size_bytes=240,
            content=(
                '[project]\n'
                'name = "demo-api"\n'
                'requires-python = ">=3.11"\n'
            ),
        )
        result = _build_result(
            [package_json, pyproject, app_file],
            ['package.json', 'services/api/pyproject.toml', 'services/api/app.py'],
        )
        result.project_profile.subprojects = [
            SubprojectSummary(
                root_path='services/api',
                language_scope='python',
                project_kind='service',
                config_paths=['services/api/pyproject.toml'],
                entrypoint_paths=['services/api/app.py'],
                markers=['service'],
            )
        ]
        result.project_profile.entrypoints = ['services/api/app.py']
        result.project_profile.code_symbols = [
            CodeSymbol(
                name='create_app',
                symbol_type='function',
                source_path='services/api/app.py',
                line_number=6,
            )
        ]
        result.project_profile.module_relations = [
            ModuleRelation(
                source_path='services/api/app.py',
                target='api.router',
                relation_type='import',
                line_number=2,
            )
        ]

        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))

        startup_result = search_knowledge_base(
            query='这个项目怎么启动',
            top_k=3,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        env_result = search_knowledge_base(
            query='这个项目依赖哪些环境变量',
            top_k=3,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        architecture_result = search_knowledge_base(
            query='这个仓库有哪些子项目和模块关系',
            top_k=3,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )
        overview_result = search_knowledge_base(
            query='这个项目是做什么的？',
            top_k=3,
            target_dir=str(temp_dir),
            repo_id='demo/sample',
        )

        assert startup_result.hits[0].document.doc_type == 'entrypoint_summary'
        assert env_result.hits[0].document.doc_type == 'config_summary'
        assert architecture_result.hits[0].document.doc_type == 'subproject_summary'
        assert overview_result.hits[0].document.doc_type in {'readme_summary', 'repo_summary'}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_search_knowledge_base_skips_chroma_for_small_repo_scope() -> None:
    original_load_repo_documents = search_service_module.load_repo_documents
    original_search_documents_in_chroma = search_service_module.search_documents_in_chroma
    try:
        documents = [
            KnowledgeDocument(
                doc_id='demo/sample::entrypoint',
                repo_id='demo/sample',
                doc_type='entrypoint_summary',
                title='demo/sample::app.py 入口摘要',
                content='来源文件：app.py\n启动命令：uvicorn app:app --reload\n',
                source_path='app.py',
                metadata={'entrypoint_startup_commands': ['uvicorn app:app --reload']},
            )
        ]
        captured = {'called': False}

        search_service_module.load_repo_documents = lambda repo_id, target_dir='data/knowledge': documents

        def fake_search_documents_in_chroma(query: str, top_k: int = 5, repo_id: str | None = None, target_dir: str = 'data/chroma'):
            captured['called'] = True
            return [
                SearchHit(
                    document=documents[0],
                    score=0.91,
                    snippet='启动命令：uvicorn app:app --reload',
                )
            ]

        search_service_module.search_documents_in_chroma = fake_search_documents_in_chroma

        result = search_knowledge_base(
            query='这个项目怎么启动',
            top_k=3,
            repo_id='demo/sample',
        )

        assert result.backend == 'local'
        assert result.hits[0].document.doc_type == 'entrypoint_summary'
        assert captured['called'] is False
    finally:
        search_service_module.load_repo_documents = original_load_repo_documents
        search_service_module.search_documents_in_chroma = original_search_documents_in_chroma


def test_search_knowledge_base_skips_chroma_for_small_global_corpus() -> None:
    original_load_all_documents = search_service_module.load_all_documents
    original_search_documents_in_chroma = search_service_module.search_documents_in_chroma
    try:
        documents = [
            KnowledgeDocument(
                doc_id='demo/sample::repo_summary',
                repo_id='demo/sample',
                doc_type='repo_summary',
                title='demo/sample::项目概览',
                content='这个项目是做什么的：用于测试全局检索回退策略。',
                source_path=None,
                metadata={'project_type': 'CLI 工具', 'project_markers': ['做什么', '概览']},
            )
        ]
        captured = {'called': False}

        search_service_module.load_all_documents = lambda target_dir='data/knowledge': documents

        def fake_search_documents_in_chroma(
            query: str,
            top_k: int = 5,
            repo_id: str | None = None,
            target_dir: str = 'data/chroma',
        ):
            captured['called'] = True
            return [
                SearchHit(
                    document=documents[0],
                    score=0.91,
                    snippet='这个项目是做什么的：用于测试全局检索回退策略。',
                )
            ]

        search_service_module.search_documents_in_chroma = fake_search_documents_in_chroma

        result = search_knowledge_base(
            query='这个项目是做什么的？',
            top_k=3,
        )

        assert result.backend == 'local'
        assert result.hits[0].document.doc_type == 'repo_summary'
        assert captured['called'] is False
    finally:
        search_service_module.load_all_documents = original_load_all_documents
        search_service_module.search_documents_in_chroma = original_search_documents_in_chroma


def test_build_embedding_text_includes_title_metadata_and_content() -> None:
    document = KnowledgeDocument(
        doc_id='demo/sample::entrypoint',
        repo_id='demo/sample',
        doc_type='entrypoint_summary',
        title='demo/sample::app.py 入口摘要',
        content='启动命令：uvicorn app:app --reload',
        source_path='app.py',
        metadata={
            'primary_language': 'Python',
            'frameworks': ['FastAPI'],
            'entrypoint_startup_commands': ['uvicorn app:app --reload'],
        },
    )

    embedding_text = chroma_store_module._build_embedding_text(document)

    assert '标题: demo/sample::app.py 入口摘要' in embedding_text
    assert '类型: entrypoint_summary' in embedding_text
    assert '路径: app.py' in embedding_text
    assert 'primary_language: Python' in embedding_text
    assert 'frameworks: FastAPI' in embedding_text
    assert 'entrypoint_startup_commands: uvicorn app:app --reload' in embedding_text
    assert '正文:' in embedding_text

def test_search_knowledge_base_prefers_exact_symbol_match_for_implementation_queries() -> None:
    original_load_repo_documents = search_service_module.load_repo_documents
    try:
        documents = [
            KnowledgeDocument(
                doc_id='demo/sample::readme',
                repo_id='demo/sample',
                doc_type='readme_summary',
                title='demo/sample::README summary',
                content='This project includes a login feature and user authentication flow.',
                source_path='README.md',
                metadata={'project_type': 'Web API'},
            ),
            KnowledgeDocument(
                doc_id='demo/sample::function',
                repo_id='demo/sample',
                doc_type='function_summary',
                title='demo/sample::AuthService.handle_login',
                content=(
                    'symbol_name: handle_login\n'
                    'qualified_name: AuthService.handle_login\n'
                    'source_path: app/auth_service.py\n'
                    'summary: handle_login verifies password and creates a session token.'
                ),
                source_path='app/auth_service.py',
                metadata={
                    'symbol_name': 'handle_login',
                    'qualified_name': 'AuthService.handle_login',
                    'called_symbols': ['verify_password', 'create_session_token'],
                    'source_path': 'app/auth_service.py',
                },
            ),
        ]
        search_service_module.load_repo_documents = lambda repo_id, target_dir='data/knowledge': documents

        result = search_knowledge_base(
            query='AuthService.handle_login 是怎么实现的？',
            top_k=2,
            repo_id='demo/sample',
        )

        assert result.hits
        assert result.hits[0].document.doc_type == 'function_summary'
        assert result.hits[0].document.title.endswith('AuthService.handle_login')
    finally:
        search_service_module.load_repo_documents = original_load_repo_documents


def test_search_knowledge_base_can_match_unified_code_entity_metadata() -> None:
    original_load_repo_documents = search_service_module.load_repo_documents
    try:
        documents = [
            KnowledgeDocument(
                doc_id='demo/sample::key_file::app.py',
                repo_id='demo/sample',
                doc_type='key_file_summary',
                title='demo/sample::app.py',
                content='app.py contains the core auth flow.',
                source_path='app.py',
                metadata={
                    'code_entity_names': ['handle_login'],
                    'code_entity_kinds': ['function'],
                    'code_entity_refs': ['AuthService.handle_login'],
                    'code_relation_targets': ['create_session_token'],
                    'code_relation_sources': ['AuthService.handle_login'],
                    'code_relation_types': ['call'],
                },
            ),
            KnowledgeDocument(
                doc_id='demo/sample::readme',
                repo_id='demo/sample',
                doc_type='readme_summary',
                title='demo/sample::README',
                content='This project provides authentication.',
                source_path='README.md',
                metadata={},
            ),
        ]
        search_service_module.load_repo_documents = lambda repo_id, target_dir='data/knowledge': documents

        result = search_knowledge_base(
            query='AuthService.handle_login 调用了什么？',
            top_k=2,
            repo_id='demo/sample',
        )

        assert result.hits
        assert result.hits[0].document.doc_type == 'key_file_summary'
        assert result.hits[0].document.source_path == 'app.py'
    finally:
        search_service_module.load_repo_documents = original_load_repo_documents
