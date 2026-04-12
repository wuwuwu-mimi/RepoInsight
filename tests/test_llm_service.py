import repoinsight.llm.service as llm_service_module
from repoinsight.answer.formatter import normalize_structured_answer
from repoinsight.llm.config import LlmSettings
from repoinsight.llm.service import _build_user_prompt, generate_answer_with_llm


def test_build_user_prompt_requires_fixed_sections() -> None:
    prompt = _build_user_prompt(
        question='这个项目怎么启动',
        repo_id='demo/sample',
        draft_answer='候选回答',
        evidence_lines=['entrypoint_summary | app.py | 启动命令：uvicorn app:app --reload'],
    )

    assert '结论：' in prompt
    assert '依据：' in prompt
    assert '不确定点：' in prompt
    assert '严格使用下面这个固定格式' in prompt


def test_normalize_llm_answer_preserves_structured_sections() -> None:
    raw_answer = (
        '### 结论\n'
        '- 可以优先使用 uvicorn 启动。\n'
        '### 依据\n'
        '- app.py 中可见 FastAPI app 对象。\n'
        '- entrypoint_summary 提供了 uvicorn app:app --reload。\n'
        '### 不确定点\n'
        '- 暂未看到生产环境命令。\n'
    )

    normalized = normalize_structured_answer(raw_answer)

    assert normalized.startswith('结论：')
    assert '\n依据：\n' in normalized
    assert '\n不确定点：\n' in normalized
    assert '- 可以优先使用 uvicorn 启动。' in normalized


def test_normalize_llm_answer_fills_missing_sections() -> None:
    raw_answer = '可以直接运行 uvicorn app:app --reload。'

    normalized = normalize_structured_answer(raw_answer)

    assert normalized.startswith('结论：')
    assert '\n依据：\n' in normalized
    assert '\n不确定点：\n' in normalized
    assert '无明显不确定点。' in normalized


def test_generate_answer_with_llm_supports_ollama_provider() -> None:
    original_client = llm_service_module.httpx.Client

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                'message': {
                    'content': '结论：可以使用 uvicorn 启动。\n依据：app.py 中可见 FastAPI app。\n不确定点：无明显不确定点。'
                }
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.endpoint = None
            self.payload = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, endpoint: str, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            assert endpoint == 'http://127.0.0.1:11434/api/chat'
            assert json['model'] == 'qwen2.5:7b'
            assert json['stream'] is False
            return FakeResponse()

    try:
        llm_service_module.httpx.Client = FakeClient
        answer = generate_answer_with_llm(
            question='这个项目怎么启动',
            repo_id='demo/sample',
            draft_answer='候选回答',
            evidence_lines=['entrypoint_summary | app.py | 启动命令：uvicorn app:app --reload'],
            settings=LlmSettings(
                provider='ollama',
                model='qwen2.5:7b',
                base_url='http://127.0.0.1:11434',
                api_key='ollama',
                timeout_seconds=30.0,
                temperature=0.2,
            ),
        )
    finally:
        llm_service_module.httpx.Client = original_client

    assert answer.startswith('结论：')
    assert '\n依据：\n' in answer


def test_generate_answer_with_llm_supports_openai_streaming() -> None:
    original_client = llm_service_module.httpx.Client
    chunks: list[str] = []

    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"结论：\\n"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"- 可以使用 uvicorn 启动。\\n依据：\\n"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"- app.py 中定义了 FastAPI app。\\n不确定点：\\n- 无明显不确定点。"}}]}'
            yield 'data: [DONE]'

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, endpoint: str, headers: dict[str, str], json: dict[str, object]):
            assert method == 'POST'
            assert endpoint == 'https://example.com/v1/chat/completions'
            assert json['stream'] is True
            return FakeStreamResponse()

    try:
        llm_service_module.httpx.Client = FakeClient
        answer = generate_answer_with_llm(
            question='这个项目怎么启动',
            repo_id='demo/sample',
            draft_answer='候选回答',
            evidence_lines=['entrypoint_summary | app.py | 启动命令：uvicorn app:app --reload'],
            settings=LlmSettings(
                provider='openai',
                model='gpt-4o-mini',
                base_url='https://example.com/v1',
                api_key='demo-key',
                timeout_seconds=30.0,
                temperature=0.2,
            ),
            stream=True,
            on_chunk=chunks.append,
        )
    finally:
        llm_service_module.httpx.Client = original_client

    assert ''.join(chunks).startswith('结论：')
    assert answer.startswith('结论：')
    assert '\n依据：\n' in answer


def test_generate_answer_with_llm_supports_ollama_streaming() -> None:
    original_client = llm_service_module.httpx.Client
    chunks: list[str] = []

    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield '{"message":{"content":"结论：\\n"}}'
            yield '{"message":{"content":"- 可以使用 uvicorn 启动。\\n依据：\\n"}}'
            yield '{"message":{"content":"- app.py 中定义了 FastAPI app。\\n不确定点：\\n- 无明显不确定点。"}}'

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, endpoint: str, headers: dict[str, str], json: dict[str, object]):
            assert method == 'POST'
            assert endpoint == 'http://127.0.0.1:11434/api/chat'
            assert json['stream'] is True
            return FakeStreamResponse()

    try:
        llm_service_module.httpx.Client = FakeClient
        answer = generate_answer_with_llm(
            question='这个项目怎么启动',
            repo_id='demo/sample',
            draft_answer='候选回答',
            evidence_lines=['entrypoint_summary | app.py | 启动命令：uvicorn app:app --reload'],
            settings=LlmSettings(
                provider='ollama',
                model='qwen2.5:7b',
                base_url='http://127.0.0.1:11434',
                api_key='ollama',
                timeout_seconds=30.0,
                temperature=0.2,
            ),
            stream=True,
            on_chunk=chunks.append,
        )
    finally:
        llm_service_module.httpx.Client = original_client

    assert ''.join(chunks).startswith('结论：')
    assert answer.startswith('结论：')
