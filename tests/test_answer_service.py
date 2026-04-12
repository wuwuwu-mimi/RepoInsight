import shutil
from pathlib import Path

import repoinsight.answer.service as answer_service_module
from repoinsight.answer.service import answer_repo_question
from repoinsight.models.analysis_model import KeyFileContent
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import save_repo_documents
from tests.test_summary_builders import _build_result


def test_answer_repo_question_returns_extractive_answer() -> None:
    temp_dir = Path('data/test_answer_service')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
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
        answer_service_module.get_llm_settings = lambda: None

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='这个项目怎么启动',
            target_dir=str(temp_dir),
        )

        assert answer_result.repo_id == 'demo/sample'
        assert answer_result.answer_mode == 'extractive'
        assert answer_result.llm_enabled is True
        assert answer_result.llm_attempted is False
        assert answer_result.answer.startswith('结论：')
        assert '\n依据：\n' in answer_result.answer
        assert '\n不确定点：\n' in answer_result.answer
        assert 'uvicorn app:app --reload' in answer_result.answer
        assert 'app.py' in answer_result.answer
        assert 'pyproject.toml' in answer_result.answer
        assert answer_result.evidence
        assert any(item.doc_type == 'entrypoint_summary' for item in answer_result.evidence)
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_answer_repo_question_uses_llm_when_available() -> None:
    temp_dir = Path('data/test_answer_service_llm')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
    original_generate_answer_with_llm = answer_service_module.generate_answer_with_llm
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

        answer_service_module.get_llm_settings = lambda: object()
        answer_service_module.generate_answer_with_llm = lambda **_: '这是 LLM 生成的回答。'

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='这个项目怎么启动',
            target_dir=str(temp_dir),
        )

        assert answer_result.answer_mode == 'llm'
        assert answer_result.answer == '这是 LLM 生成的回答。'
        assert answer_result.fallback_used is False
        assert answer_result.llm_enabled is True
        assert answer_result.llm_attempted is True
        assert answer_result.llm_error is None
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        answer_service_module.generate_answer_with_llm = original_generate_answer_with_llm
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_answer_repo_question_can_disable_llm_explicitly() -> None:
    temp_dir = Path('data/test_answer_service_no_llm')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
    original_generate_answer_with_llm = answer_service_module.generate_answer_with_llm
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

        answer_service_module.get_llm_settings = lambda: object()
        answer_service_module.generate_answer_with_llm = lambda **_: '这条回答不应被使用。'

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='这个项目怎么启动',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert answer_result.answer_mode == 'extractive'
        assert answer_result.answer.startswith('结论：')
        assert 'uvicorn app:app --reload' in answer_result.answer
        assert answer_result.llm_enabled is False
        assert answer_result.llm_attempted is False
        assert answer_result.llm_error is None
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        answer_service_module.generate_answer_with_llm = original_generate_answer_with_llm
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_answer_repo_question_passes_stream_callback_to_llm() -> None:
    temp_dir = Path('data/test_answer_service_stream')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
    original_generate_answer_with_llm = answer_service_module.generate_answer_with_llm
    streamed_chunks: list[str] = []
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

        answer_service_module.get_llm_settings = lambda: object()

        def fake_generate_answer_with_llm(**kwargs):
            assert kwargs['stream'] is True
            kwargs['on_chunk']('结论：\n')
            kwargs['on_chunk']('- 可以使用 uvicorn 启动。')
            return '结论：\n- 可以使用 uvicorn 启动。\n依据：\n- app.py 中定义了 FastAPI app。\n不确定点：\n- 无明显不确定点。'

        answer_service_module.generate_answer_with_llm = fake_generate_answer_with_llm

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='这个项目怎么启动',
            target_dir=str(temp_dir),
            llm_stream=True,
            on_llm_chunk=streamed_chunks.append,
        )

        assert ''.join(streamed_chunks).startswith('结论：')
        assert answer_result.answer_mode == 'llm'
        assert answer_result.llm_attempted is True
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        answer_service_module.generate_answer_with_llm = original_generate_answer_with_llm
        shutil.rmtree(temp_dir, ignore_errors=True)
