import shutil
from pathlib import Path

from repoinsight.agents.models import CodeRelationChain, CodeRelationEdge
import repoinsight.answer.service as answer_service_module
from repoinsight.answer.service import answer_repo_question
from repoinsight.models.analysis_model import ApiRouteSummary, FunctionSummary, KeyFileContent
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


def test_select_supporting_lines_can_use_chunk_code_lines_for_implementation_query() -> None:
    hit = SearchHit(
        document=KnowledgeDocument(
            doc_id='demo/sample::function_body_chunk::app.py::AuthService.handle_login',
            repo_id='demo/sample',
            doc_type='function_body_chunk',
            title='demo/sample::AuthService.handle_login 函数源码片段',
            content=(
                '仓库 demo/sample 中的函数源码片段。\n'
                '来源文件：app.py\n'
                '限定名：AuthService.handle_login\n'
                '代码位置：L10-L14\n'
                '源码片段：\n'
                '```text\n'
                'L10: def handle_login(username, password):\n'
                'L11:     token = create_session_token(username)\n'
                'L12:     return token\n'
                '```\n'
            ),
            source_path='app.py',
            metadata={
                'qualified_name': 'AuthService.handle_login',
                'line_start': 10,
                'line_end': 14,
            },
        ),
        score=8.8,
        snippet='AuthService.handle_login 源码片段',
    )

    lines = answer_service_module._select_supporting_lines(
        'AuthService.handle_login 是怎么实现的？',
        [hit],
        'implementation',
        max_lines=3,
    )

    assert lines
    assert any('L10: def handle_login(username, password):' in line for line in lines)


def test_build_implementation_answer_uses_chunk_evidence() -> None:
    hit = SearchHit(
        document=KnowledgeDocument(
            doc_id='demo/sample::function_body_chunk::app.py::AuthService.handle_login',
            repo_id='demo/sample',
            doc_type='function_body_chunk',
            title='demo/sample::AuthService.handle_login 函数源码片段',
            content=(
                '仓库 demo/sample 中的函数源码片段。\n'
                '来源文件：app.py\n'
                '符号名称：handle_login\n'
                '限定名：AuthService.handle_login\n'
                '函数签名：def handle_login(username, password)\n'
                '代码位置：L10-L14\n'
                '调用：create_session_token\n'
                '摘要：负责校验登录并生成 token。\n'
                '源码片段：\n'
                '```text\n'
                'L10: def handle_login(username, password):\n'
                'L11:     token = create_session_token(username)\n'
                'L12:     return token\n'
                '```\n'
            ),
            source_path='app.py',
            metadata={
                'qualified_name': 'AuthService.handle_login',
                'symbol_name': 'handle_login',
            },
        ),
        score=9.0,
        snippet='AuthService.handle_login 源码片段',
    )

    answer = answer_service_module._build_implementation_answer(
        [hit],
        ['[function_body_chunk | app.py] L10: def handle_login(username, password):'],
    )

    assert '已命中更贴近实现的源码片段' in answer
    assert 'L10: def handle_login(username, password):' in answer
    assert '代码位置：L10-L14' in answer


def test_build_api_answer_uses_route_handler_chunk_evidence() -> None:
    hit = SearchHit(
        document=KnowledgeDocument(
            doc_id='demo/sample::route_handler_chunk::app.py::POST::/login',
            repo_id='demo/sample',
            doc_type='route_handler_chunk',
            title='demo/sample::POST /login 接口源码片段',
            content=(
                '仓库 demo/sample 中的接口处理源码片段。\n'
                '来源文件：app.py\n'
                '路由路径：/login\n'
                'HTTP 方法：POST\n'
                '处理函数：login\n'
                '处理限定名：login\n'
                '代码位置：L20-L24\n'
                '调用：auth_service.handle_login\n'
                '摘要：POST /login 会调用 auth_service.handle_login。\n'
                '源码片段：\n'
                '```text\n'
                'L20: @router.post(\"/login\")\n'
                'L21: def login(username, password):\n'
                'L22:     return auth_service.handle_login(username, password)\n'
                '```\n'
            ),
            source_path='app.py',
            metadata={
                'route_path': '/login',
                'handler_qualified_name': 'login',
            },
        ),
        score=9.1,
        snippet='POST /login 接口源码片段',
    )

    answer = answer_service_module._build_api_answer(
        [hit],
        ['[route_handler_chunk | app.py] L20: @router.post("/login")'],
    )

    assert '已命中更贴近实现的源码片段' in answer
    assert 'L20: @router.post("/login")' in answer
    assert '代码位置：L20-L24' in answer


def test_append_extra_context_notes_promotes_relation_chain_section() -> None:
    answer_text = '结论：\n- handle_login 负责登录。\n依据：\n- function_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] 已定位到与问题最相关的实现符号 AuthService.handle_login（置信度=high，相关性=0.92）',
        '[code_agent] 代表性关系链：POST /login -> AuthService.handle_login -> create_session_token',
        '[code_agent] [depth=1] AuthService.handle_login @ app.py:L18-L33 -> 函数 handle_login 承担当前实现逻辑',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='implementation',
    )

    assert '实现链路：' in merged
    assert 'POST /login -> AuthService.handle_login -> create_session_token' in merged
    assert '主实现流程从 POST /login 开始，随后依次调用 AuthService.handle_login、create_session_token' in merged
    assert '补充实现线索：' in merged
    assert '已定位到与问题最相关的实现符号 AuthService.handle_login' in merged


def test_append_extra_context_notes_uses_api_specific_relation_wording() -> None:
    answer_text = '结论：\n- 登录接口入口已经定位。\n依据：\n- api_route_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] 关系链类型：POST /login -[handle_route]-> AuthService.handle_login -[call]-> create_session_token',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='api',
    )

    assert '接口链路：' in merged
    assert '请求从 POST /login 路由到 AuthService.handle_login；从 AuthService.handle_login 继续调用 create_session_token' in merged


def test_append_extra_context_notes_formats_trace_steps_as_process_lines() -> None:
    answer_text = '结论：\n- handle_login 负责登录。\n依据：\n- function_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] [depth=0] AuthService.handle_login @ app.py:L18-L33 -> 函数 AuthService.handle_login 承担当前实现逻辑',
        '[code_agent] [depth=1] create_session_token <- AuthService.handle_login @ app.py:L40-L42 -> 作为 AuthService.handle_login 的下一跳，函数 create_session_token 负责生成登录态 token',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='implementation',
    )

    assert '实现过程：' in merged
    assert '先定位到 AuthService.handle_login（app.py:L18-L33），承担当前实现逻辑。' in merged
    assert '从 AuthService.handle_login 继续跟到 create_session_token（app.py:L40-L42），负责生成登录态 token。' in merged


def test_append_extra_context_notes_uses_relation_specific_wording() -> None:
    answer_text = '结论：\n- 登录流程已定位。\n依据：\n- function_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] [depth=1] [relation=delegate_service] auth_service.login_user <- POST /session @ services/auth_service.py:L3-L7 -> 作为 POST /session 的下一跳，函数 auth_service.login_user 负责校验用户',
        '[code_agent] [depth=2] [relation=delegate_repository] session_repo.persist_session <- auth_service.login_user @ repositories/session_repo.py:L2-L4 -> 作为 auth_service.login_user 的下一跳，函数 session_repo.persist_session 负责把会话信息写入存储',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='api',
    )

    assert '接口过程：' in merged
    assert '从 POST /session 继续把业务处理交给服务 auth_service.login_user（services/auth_service.py:L3-L7）' in merged
    assert '从 auth_service.login_user 继续把数据访问交给 session_repo.persist_session（repositories/session_repo.py:L2-L4）' in merged


def test_append_extra_context_notes_uses_architecture_specific_relation_wording() -> None:
    answer_text = '结论：\n- 当前问题更接近模块关系。\n依据：\n- subproject_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] 代表性关系链：api.gateway -> services.auth_service -> repositories.session_repo',
        '[code_agent] 关键模块：services.auth_service 负责认证编排',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='architecture',
    )

    assert '模块链路：' in merged
    assert '模块链路从 api.gateway 出发，中间经过 services.auth_service，最终落到 repositories.session_repo' in merged
    assert '补充模块线索：' in merged


def test_append_extra_context_notes_formats_architecture_trace_steps() -> None:
    answer_text = '结论：\n- 当前问题更接近模块关系。\n依据：\n- subproject_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] [depth=0] services.auth_service @ services/auth_service.py:L3-L7 -> 函数 services.auth_service 承担当前实现逻辑',
        '[code_agent] [depth=1] repositories.session_repo <- services.auth_service @ repositories/session_repo.py:L2-L4 -> 作为 services.auth_service 的下一跳，函数 repositories.session_repo 负责把会话信息写入存储',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='architecture',
    )

    assert '模块展开：' in merged
    assert '当前模块链路先落在 services.auth_service（services/auth_service.py:L3-L7）' in merged
    assert 'services.auth_service 继续依赖到 repositories.session_repo（repositories/session_repo.py:L2-L4）' in merged


def test_append_extra_context_notes_formats_import_module_relation_for_architecture() -> None:
    answer_text = '结论：\n- 当前问题更接近模块关系。\n依据：\n- key_file_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] [depth=1] [relation=import_module] services.auth_service <- api/routes.py @ services/auth_service.py:L1-L4 -> 作为 api/routes.py 的下一跳，关键文件 services/auth_service.py 提供当前模块链路的实现上下文',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='architecture',
    )

    assert '模块展开：' in merged
    assert 'api/routes.py 继续依赖模块 services.auth_service（services/auth_service.py:L1-L4）' in merged


def test_append_extra_context_notes_prefers_structured_relation_chain_details() -> None:
    answer_text = '结论：\n- 当前问题更接近模块关系。\n依据：\n- key_file_summary 命中。\n不确定点：\n- 暂无。'
    extra_context_lines = [
        '[code_agent] 代表性关系链：legacy.gateway -> legacy.service -> legacy.repo',
        '[code_agent] 关键模块：services.auth_service 负责认证编排',
    ]

    merged = answer_service_module._append_extra_context_notes(
        answer_text,
        extra_context_lines,
        focus='architecture',
        relation_chain_details=[
            CodeRelationChain(
                plain_text='POST /session -> auth_service.login_user -> session_repo.persist_session',
                typed_text=(
                    'POST /session -[delegate_service]-> auth_service.login_user '
                    '-[delegate_repository]-> session_repo.persist_session'
                ),
                labels=['POST /session', 'auth_service.login_user', 'session_repo.persist_session'],
                edges=[
                    CodeRelationEdge(
                        source_label='POST /session',
                        target_label='auth_service.login_user',
                        relation_type='delegate_service',
                    ),
                    CodeRelationEdge(
                        source_label='auth_service.login_user',
                        target_label='session_repo.persist_session',
                        relation_type='delegate_repository',
                    ),
                ],
            )
        ],
    )

    assert 'POST /session -[delegate_service]-> auth_service.login_user -[delegate_repository]-> session_repo.persist_session' in merged
    assert 'legacy.gateway -> legacy.service -> legacy.repo' not in merged


def test_build_answer_result_from_context_appends_architecture_extra_context() -> None:
    hit = SearchHit(
        document=KnowledgeDocument(
            doc_id='demo/sample::subproject',
            repo_id='demo/sample',
            doc_type='subproject_summary',
            title='demo/sample::backend',
            content='摘要：backend 子项目负责 HTTP 接口与服务编排。',
            source_path='backend/README.md',
            metadata={},
        ),
        score=8.8,
        snippet='backend 子项目负责 HTTP 接口与服务编排。',
    )

    result = answer_service_module._build_answer_result_from_context(
        repo_id='demo/sample',
        question='这个项目的模块关系是怎样的？',
        focus='architecture',
        backend='chroma',
        prioritized_hits=[hit],
        selected_lines=['[subproject_summary | backend/README.md] 摘要：backend 子项目负责 HTTP 接口与服务编排。'],
        extra_context_lines=[
            '[code_agent] 代表性关系链：api.gateway -> services.auth_service -> repositories.session_repo',
        ],
        use_llm=False,
        llm_stream=False,
        on_llm_chunk=None,
    )

    assert result.answer_mode == 'extractive'
    assert '模块链路：' in result.answer
    assert 'api.gateway -> services.auth_service -> repositories.session_repo' in result.answer


def test_build_architecture_answer_organizes_subprojects_and_module_relations() -> None:
    hits = [
        SearchHit(
            document=KnowledgeDocument(
                doc_id='demo/sample::subproject::backend',
                repo_id='demo/sample',
                doc_type='subproject_summary',
                title='demo/sample::backend 子项目摘要',
                content=(
                    '仓库 demo/sample 的子项目摘要。\n'
                    '子项目根目录：backend\n'
                    '语言范围：python\n'
                    '子项目类型：api-service\n'
                    '配置文件：backend/pyproject.toml\n'
                    '入口文件：backend/main.py\n'
                ),
                source_path='backend',
                metadata={},
            ),
            score=9.0,
            snippet='backend 子项目负责 API 服务。',
        ),
        SearchHit(
            document=KnowledgeDocument(
                doc_id='demo/sample::key_file::backend/main.py',
                repo_id='demo/sample',
                doc_type='key_file_summary',
                title='demo/sample::backend/main.py',
                content=(
                    '仓库 demo/sample 的关键文件 backend/main.py 摘要候选内容。\n'
                    '所属子项目：backend。\n'
                    '关键符号：function create_app @L7。\n'
                    '模块依赖：backend.main -> services.auth_service (import @L1)；'
                    'services.auth_service -> repositories.session_repo (call @L18)。\n'
                ),
                source_path='backend/main.py',
                metadata={},
            ),
            score=8.7,
            snippet='backend/main.py 依赖 services.auth_service。',
        ),
    ]

    answer = answer_service_module._build_architecture_answer(
        hits,
        selected_lines=['[key_file_summary | backend/main.py] 模块依赖：backend.main -> services.auth_service'],
    )

    assert '当前最相关的子项目或模块入口包括：backend。' in answer
    assert '当前最值得优先核对的模块依赖包括：backend.main -> services.auth_service' in answer
    assert '关键实现符号主要集中在：function create_app @L7。' in answer
    assert '入口文件有 backend/main.py；配置文件有 backend/pyproject.toml。' in answer


def test_answer_repo_question_builds_architecture_focused_answer() -> None:
    temp_dir = Path('data/test_answer_service_architecture')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
    try:
        documents = [
            KnowledgeDocument(
                doc_id='demo/sample::repo_summary',
                repo_id='demo/sample',
                doc_type='repo_summary',
                title='demo/sample 仓库摘要',
                content=(
                    '仓库 demo/sample 的项目概览。\n'
                    '子项目：backend(api-service)、frontend(web-app)。\n'
                    '模块依赖关系数量：4。\n'
                ),
                source_path='README',
                metadata={},
            ),
            KnowledgeDocument(
                doc_id='demo/sample::subproject_summary::backend',
                repo_id='demo/sample',
                doc_type='subproject_summary',
                title='demo/sample::backend 子项目摘要',
                content=(
                    '仓库 demo/sample 的子项目摘要。\n'
                    '子项目根目录：backend\n'
                    '语言范围：python\n'
                    '子项目类型：api-service\n'
                    '配置文件：backend/pyproject.toml\n'
                    '入口文件：backend/main.py\n'
                ),
                source_path='backend',
                metadata={},
            ),
            KnowledgeDocument(
                doc_id='demo/sample::key_file::backend/main.py',
                repo_id='demo/sample',
                doc_type='key_file_summary',
                title='demo/sample::backend/main.py',
                content=(
                    '仓库 demo/sample 的关键文件 backend/main.py 摘要候选内容。\n'
                    '所属子项目：backend。\n'
                    '关键符号：function create_app @L7；function auth_service.login_user @L18。\n'
                    '模块依赖：backend.main -> services.auth_service (import @L1)；'
                    'services.auth_service -> repositories.session_repo (call @L18)。\n'
                ),
                source_path='backend/main.py',
                metadata={},
            ),
        ]
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))
        answer_service_module.get_llm_settings = lambda: None

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='这个项目的认证模块依赖关系是怎样的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert answer_result.answer_mode == 'extractive'
        assert '当前最相关的子项目或模块入口包括：backend。' in answer_result.answer
        assert '当前最值得优先核对的模块依赖包括：backend.main -> services.auth_service' in answer_result.answer
        assert '关键实现符号主要集中在：function create_app @L7；function auth_service.login_user @L18。' in answer_result.answer
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_answer_repo_question_builds_implementation_answer_with_chunk_evidence() -> None:
    temp_dir = Path('data/test_answer_service_implementation')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
    try:
        auth_service_file = KeyFileContent(
            path='services/auth_service.py',
            size_bytes=320,
            content=(
                'class AuthService:\n'
                '    def handle_login(self, username, password):\n'
                '        token = create_session_token(username)\n'
                '        return token\n'
            ),
        )
        result = _build_result([auth_service_file], ['services/auth_service.py'])
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
            )
        ]

        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))
        answer_service_module.get_llm_settings = lambda: None

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='AuthService.handle_login 是怎么实现的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert answer_result.answer_mode == 'extractive'
        assert '已命中更贴近实现的源码片段' in answer_result.answer
        assert 'AuthService.handle_login' in answer_result.answer
        assert '代码位置：L1-L4' in answer_result.answer
        assert 'L2:     def handle_login(self, username, password):' in answer_result.answer
        assert 'create_session_token' in answer_result.answer
        assert any(item.doc_type == 'function_body_chunk' for item in answer_result.evidence)
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_answer_repo_question_builds_api_answer_with_route_chunk_evidence() -> None:
    temp_dir = Path('data/test_answer_service_api_impl')
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_get_llm_settings = answer_service_module.get_llm_settings
    try:
        route_file = KeyFileContent(
            path='api/routes.py',
            size_bytes=300,
            content=(
                '@router.post("/login")\n'
                'def login(username: str, password: str):\n'
                '    return auth_service.handle_login(username, password)\n'
            ),
        )
        result = _build_result([route_file], ['api/routes.py'])
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
                line_number=1,
                decorators=['router.post("/login")'],
                called_symbols=['auth_service.handle_login'],
                summary='POST /login 会调用 auth_service.handle_login。',
            )
        ]

        documents = build_knowledge_documents(result)
        save_repo_documents(repo_id='demo/sample', documents=documents, target_dir=str(temp_dir))
        answer_service_module.get_llm_settings = lambda: None

        answer_result = answer_repo_question(
            repo_id='demo/sample',
            question='POST /login 是怎么实现的？',
            target_dir=str(temp_dir),
            use_llm=False,
        )

        assert answer_result.answer_mode == 'extractive'
        assert '已命中更贴近实现的源码片段' in answer_result.answer
        assert 'POST /login' in answer_result.answer
        assert '代码位置：L1-L3' in answer_result.answer
        assert 'L1: @router.post("/login")' in answer_result.answer
        assert 'auth_service.handle_login' in answer_result.answer
        assert any(item.doc_type == 'route_handler_chunk' for item in answer_result.evidence)
    finally:
        answer_service_module.get_llm_settings = original_get_llm_settings
        shutil.rmtree(temp_dir, ignore_errors=True)
