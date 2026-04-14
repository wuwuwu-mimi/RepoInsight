import shutil
from pathlib import Path

import repoinsight.answer.service as answer_service_module
from repoinsight.answer.service import answer_repo_question
from repoinsight.models.analysis_model import KeyFileContent
from repoinsight.models.rag_model import KnowledgeDocument, SearchHit
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


def test_answer_repo_question_prefers_readme_for_overview_questions() -> None:
    temp_dir = Path('data/test_answer_service_overview')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
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
        result = _build_result(
            [package_json, main_file],
            ['package.json', 'main.py'],
        )
        result.repo_info.readme = (
            '# helloAgent\n\n'
            '一个从零手写 Agent 的练习项目。\n\n'
            '这个仓库的目标不是做一个大而全的框架，'
            '而是把 Prompt、推理、工具调用和最终回答这条链路一步一步写清楚。'
        )
        result.repo_info.repo_model.description = '一个从零实现 Agent 核心链路的学习型仓库'
        result.project_type = 'Agent 学习项目'
        result.project_type_evidence = 'README 与关键文件都围绕 Agent 主链路展开'

        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))
        answer_service_module.get_llm_settings = lambda: None

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='这个项目是做什么的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert answer_result.answer_mode == 'extractive'
        assert '一个从零手写 Agent 的练习项目' in answer_result.answer
        assert 'Agent 学习项目' in answer_result.answer
        assert answer_result.evidence
        assert answer_result.evidence[0].doc_type == 'readme_summary'
        assert any(item.doc_type == 'repo_summary' for item in answer_result.evidence)
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_select_supporting_lines_can_use_snippet_for_precise_code_query() -> None:
    hit = SearchHit(
        document=KnowledgeDocument(
            doc_id='demo/sample::function',
            repo_id='demo/sample',
            doc_type='function_summary',
            title='demo/sample::AuthService.handle_login',
            content='This summary is compact.',
            source_path='app/auth_service.py',
            metadata={},
        ),
        score=9.2,
        snippet='qualified_name: AuthService.handle_login',
    )

    lines = answer_service_module._select_supporting_lines(
        'AuthService.handle_login 是怎么实现的？',
        [hit],
        'implementation',
        max_lines=2,
    )

    assert lines
    assert 'AuthService.handle_login' in lines[0]


def test_append_extra_context_notes_promotes_relation_chain_section() -> None:
    answer_text = '结论：\n- handle_login 负责登录。\n依据：\n- function_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] 已定位到与问题最相关的实现符号 AuthService.handle_login（置信度=high，相关性=0.92）',
        '[code_agent] 代表性关系链：POST /login -> AuthService.handle_login -> create_session_token',
        '[code_agent] [depth=1] AuthService.handle_login @ app.py:L18-L33 -> 函数 handle_login 承担当前实现逻辑',
    ]

    merged = answer_service_module._append_extra_context_notes(answer_text, extra_context_lines)

    assert '实现链路：' in merged
    assert 'POST /login -> AuthService.handle_login -> create_session_token' in merged
    assert '补充线索：' in merged
    assert '已定位到与问题最相关的实现符号 AuthService.handle_login' in merged
