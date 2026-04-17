import json
from collections.abc import Callable

import httpx

from repoinsight.answer.formatter import normalize_structured_answer
from repoinsight.llm.config import LlmSettings, get_llm_settings


class LlmInvocationError(RuntimeError):
    """表示 LLM 调用失败。"""


def generate_answer_with_llm(
    question: str,
    repo_id: str,
    draft_answer: str,
    evidence_lines: list[str],
    settings: LlmSettings | None = None,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """调用 OpenAI 兼容接口，根据检索证据生成最终回答。"""
    resolved_settings = settings or get_llm_settings()
    if resolved_settings is None:
        raise LlmInvocationError('LLM 配置未完成')

    messages = _build_messages(
        question=question,
        repo_id=repo_id,
        draft_answer=draft_answer,
        evidence_lines=evidence_lines,
    )
    message = _request_chat_completion(
        settings=resolved_settings,
        messages=messages,
        stream=stream,
        on_chunk=on_chunk,
    )

    if not isinstance(message, str) or not message.strip():
        raise LlmInvocationError('LLM 返回了空内容')

    return normalize_structured_answer(message)


def _request_chat_completion(
    settings: LlmSettings,
    messages: list[dict[str, str]],
    stream: bool,
    on_chunk: Callable[[str], None] | None,
) -> str:
    """根据 provider 选择合适的接口协议。"""
    if settings.provider == 'ollama':
        return _request_ollama_completion(
            settings=settings,
            messages=messages,
            stream=stream,
            on_chunk=on_chunk,
        )
    return _request_openai_compatible_completion(
        settings=settings,
        messages=messages,
        stream=stream,
        on_chunk=on_chunk,
    )


def _request_openai_compatible_completion(
    settings: LlmSettings,
    messages: list[dict[str, str]],
    stream: bool,
    on_chunk: Callable[[str], None] | None,
) -> str:
    """调用 OpenAI 兼容的 chat completions 接口。"""
    payload = {
        'model': settings.model,
        'temperature': settings.temperature,
        'messages': messages,
        'stream': stream,
    }
    endpoint = f'{settings.base_url}/chat/completions'
    headers = {
        'Authorization': f'Bearer {settings.api_key}',
        'Content-Type': 'application/json',
    }

    if stream:
        return _stream_openai_compatible_completion(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=settings.timeout_seconds,
            on_chunk=on_chunk,
        )

    response = _post_json(
        endpoint=endpoint,
        headers=headers,
        payload=payload,
        timeout_seconds=settings.timeout_seconds,
    )

    try:
        data = response.json()
        return data['choices'][0]['message']['content']
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LlmInvocationError('LLM 返回结果格式不符合预期') from exc


def _request_ollama_completion(
    settings: LlmSettings,
    messages: list[dict[str, str]],
    stream: bool,
    on_chunk: Callable[[str], None] | None,
) -> str:
    """调用 Ollama 原生 chat 接口。"""
    payload = {
        'model': settings.model,
        'messages': messages,
        'stream': stream,
        'options': {
            'temperature': settings.temperature,
        },
    }
    endpoint = f'{settings.base_url}/api/chat'

    if stream:
        return _stream_ollama_completion(
            endpoint=endpoint,
            headers={'Content-Type': 'application/json'},
            payload=payload,
            timeout_seconds=settings.timeout_seconds,
            on_chunk=on_chunk,
        )

    response = _post_json(
        endpoint=endpoint,
        headers={'Content-Type': 'application/json'},
        payload=payload,
        timeout_seconds=settings.timeout_seconds,
    )

    try:
        data = response.json()
        return data['message']['content']
    except (ValueError, KeyError, TypeError) as exc:
        raise LlmInvocationError('Ollama 返回结果格式不符合预期') from exc


def _stream_openai_compatible_completion(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: float,
    on_chunk: Callable[[str], None] | None,
) -> str:
    """以 SSE 方式流式接收 OpenAI 兼容接口输出。"""
    chunks: list[str] = []
    try:
        with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
            with client.stream('POST', endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith('data:'):
                        continue

                    data_text = line[len('data:'):].strip()
                    if data_text == '[DONE]':
                        break

                    try:
                        data = json.loads(data_text)
                        delta = data['choices'][0]['delta'].get('content', '')
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue

                    if not delta:
                        continue

                    chunks.append(delta)
                    if on_chunk is not None:
                        on_chunk(delta)
    except httpx.HTTPError as exc:
        raise LlmInvocationError(f'LLM 流式请求失败：{exc}') from exc

    return ''.join(chunks).strip()


def _stream_ollama_completion(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: float,
    on_chunk: Callable[[str], None] | None,
) -> str:
    """以 NDJSON 方式流式接收 Ollama 输出。"""
    chunks: list[str] = []
    try:
        with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
            with client.stream('POST', endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        delta = data.get('message', {}).get('content', '')
                    except ValueError:
                        continue

                    if delta:
                        chunks.append(delta)
                        if on_chunk is not None:
                            on_chunk(delta)
    except httpx.HTTPError as exc:
        raise LlmInvocationError(f'LLM 流式请求失败：{exc}') from exc

    return ''.join(chunks).strip()


def _post_json(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: float,
) -> httpx.Response:
    """统一执行 POST JSON 请求。"""
    try:
        with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            return response
    except httpx.HTTPError as exc:
        raise LlmInvocationError(f'LLM 请求失败：{exc}') from exc


def _build_messages(
    question: str,
    repo_id: str,
    draft_answer: str,
    evidence_lines: list[str],
) -> list[dict[str, str]]:
    """构造统一的聊天消息列表。"""
    return [
        {
            'role': 'system',
            'content': (
                '你是 RepoInsight 的仓库分析助手。'
                '你必须只基于提供的检索证据回答，不能编造不存在的信息。'
                '如果证据不足，要明确说信息不足。'
                '请使用简洁中文回答，并优先给出直接结论，再补充关键依据。'
            ),
        },
        {
            'role': 'user',
            'content': _build_user_prompt(
                question=question,
                repo_id=repo_id,
                draft_answer=draft_answer,
                evidence_lines=evidence_lines,
            ),
        },
    ]


def _build_user_prompt(
    question: str,
    repo_id: str,
    draft_answer: str,
    evidence_lines: list[str],
) -> str:
    """构造发给 LLM 的用户提示词。"""
    evidence_text = '\n'.join(f'- {line}' for line in evidence_lines) if evidence_lines else '- 无'
    return (
        f'仓库：{repo_id}\n'
        f'问题：{question}\n\n'
        f'程序抽取出的候选回答：\n{draft_answer}\n\n'
        f'可引用证据：\n{evidence_text}\n\n'
        '请基于这些证据输出最终回答。'
        '要求：\n'
        '1. 只基于证据回答\n'
        '2. 严格使用下面这个固定格式，不要增加别的标题：\n'
        '结论：\n'
        '- ...\n'
        '依据：\n'
        '- ...\n'
        '- ...\n'
        '不确定点：\n'
        '- ...\n'
        '3. 结论部分只写最直接的回答，1~3 条即可\n'
        '4. 依据部分只写能被证据支持的点，1~3 条即可\n'
        '5. 若没有明显不确定点，也要写“无明显不确定点”\n'
        '6. 不要输出 JSON，不要输出代码块'
    )
