import shutil
from pathlib import Path

from repoinsight.models.analysis_model import KeyFileContent
from repoinsight.search.evaluation import RagEvaluationCase, evaluate_search_cases
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
