import shutil
from pathlib import Path

from repoinsight.models.analysis_model import CodeSymbol, KeyFileContent, ModuleRelation, SubprojectSummary
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

        assert startup_result.hits[0].document.doc_type == 'entrypoint_summary'
        assert env_result.hits[0].document.doc_type == 'config_summary'
        assert architecture_result.hits[0].document.doc_type == 'subproject_summary'
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
