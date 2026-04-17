import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import httpx
from pydantic import BaseModel, Field


class EmbeddingSettings(BaseModel):
    """描述当前 embedding 服务的运行配置。"""

    # 当前 embedding 提供方：service、ollama 或 sentence_transformers。
    provider: str = Field(..., description='embedding 提供方')

    # 当前使用的模型名，或本地模型目录。
    model: str = Field(..., description='模型名称或目录')

    # OpenAI 兼容服务或 Ollama 的基础地址；本地 sentence-transformers 可为空。
    base_url: str | None = Field(default=None, description='服务基础地址')

    # OpenAI 兼容服务的 API Key；本地服务可为空。
    api_key: str | None = Field(default=None, description='API Key')

    # 请求超时时间，单位秒。
    timeout: float = Field(default=30.0, description='请求超时秒数')

    # sentence-transformers 是否仅使用本地缓存。
    local_files_only: bool = Field(default=True, description='是否仅使用本地缓存')

    # 显式指定的本地模型目录。
    model_path: str | None = Field(default=None, description='本地模型目录')


class EmbeddingHealthResult(BaseModel):
    """描述一次 embedding 健康检查结果。"""

    # 当前使用的 embedding 提供方。
    provider: str = Field(..., description='embedding 提供方')

    # 当前使用的模型名或本地模型目录。
    model: str = Field(..., description='模型名称或目录')

    # 当前使用的基础地址。
    base_url: str | None = Field(default=None, description='服务基础地址')

    # 当前检查是否通过。
    healthy: bool = Field(default=False, description='是否健康')

    # 探测耗时，单位毫秒。
    latency_ms: float | None = Field(default=None, description='探测耗时毫秒')

    # 返回向量维度；若探测失败则为空。
    vector_size: int | None = Field(default=None, description='向量维度')

    # 健康检查说明。
    message: str = Field(default='', description='状态说明')

    # 失败时的错误详情。
    error: str | None = Field(default=None, description='错误信息')


class EmbeddingProviderSwitchError(ValueError):
    """表示命令行传入了不支持的 embedding 模式。"""



def _get_first_env(*keys: str) -> str | None:
    """按顺序读取第一个非空环境变量。"""
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None



def _parse_timeout(raw_value: str) -> float:
    """把环境变量中的超时时间转成浮点秒数。"""
    try:
        value = float(raw_value)
    except ValueError:
        return 30.0
    return value if value > 0 else 30.0



def _normalize_provider(provider: str) -> str:
    """把 provider 别名统一成内部使用的标准值。"""
    normalized = provider.strip().lower()
    aliases = {
        'service': 'service',
        'openai': 'service',
        'remote': 'service',
        'ollama': 'ollama',
        'local': 'ollama',
        'sentence-transformers': 'sentence_transformers',
        'sentence_transformers': 'sentence_transformers',
        'st': 'sentence_transformers',
    }
    if normalized not in aliases:
        raise EmbeddingProviderSwitchError(f'不支持的 embedding 模式：{provider}')
    return aliases[normalized]


# 默认使用服务商 embedding；如果用户想切到本地 Ollama，可通过命令行覆盖。
DEFAULT_EMBEDDING_PROVIDER = os.getenv('REPOINSIGHT_EMBEDDING_PROVIDER', 'service').strip().lower()
DEFAULT_EMBEDDING_MODEL = os.getenv('REPOINSIGHT_EMBEDDING_MODEL', 'text-embedding-3-small').strip()
DEFAULT_EMBEDDING_BASE_URL = os.getenv('REPOINSIGHT_EMBEDDING_BASE_URL', 'https://api.openai.com/v1').strip()
DEFAULT_EMBEDDING_API_KEY = _get_first_env('REPOINSIGHT_EMBEDDING_API_KEY', 'OPENAI_API_KEY')
DEFAULT_EMBEDDING_TIMEOUT = _parse_timeout(os.getenv('REPOINSIGHT_EMBEDDING_TIMEOUT', '30'))

# Ollama 本地 embedding 配置，通常只需要 base_url 和 model。
DEFAULT_OLLAMA_EMBEDDING_MODEL = os.getenv('REPOINSIGHT_OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text').strip()
DEFAULT_OLLAMA_BASE_URL = os.getenv('REPOINSIGHT_OLLAMA_BASE_URL', 'http://127.0.0.1:11434').strip()

# 兼容旧的 sentence-transformers 方案，便于离线或已有本地模型的用户继续使用。
DEFAULT_LOCAL_EMBEDDING_MODEL = os.getenv(
    'REPOINSIGHT_LOCAL_EMBEDDING_MODEL',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
).strip()
DEFAULT_LOCAL_EMBEDDING_MODEL_PATH = os.getenv('REPOINSIGHT_LOCAL_EMBEDDING_MODEL_PATH', '').strip()
DEFAULT_LOCAL_EMBEDDING_ONLY = (
    os.getenv('REPOINSIGHT_LOCAL_EMBEDDING_LOCAL_ONLY', '1').strip().lower()
    not in {'0', 'false', 'no', 'off'}
)

_PROVIDER_OVERRIDE: str | None = None


class OpenAICompatibleEmbeddingService:
    """基于 OpenAI 兼容接口的 embedding 服务封装。"""

    def __init__(self, model: str, base_url: str, api_key: str | None, timeout: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = (api_key or '').strip() or None
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """调用 OpenAI 兼容的 `/embeddings` 接口生成向量。"""
        if not texts:
            return []

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f'{self.base_url}/embeddings',
                headers=headers,
                json={'model': self.model, 'input': texts},
            )
            response.raise_for_status()
            payload = response.json()

        data = payload.get('data')
        if not isinstance(data, list):
            raise RuntimeError('embedding 服务返回格式异常：缺少 data 字段。')

        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get('embedding'), list):
                raise RuntimeError('embedding 服务返回格式异常：缺少 embedding 字段。')
            vectors.append([float(value) for value in item['embedding']])
        return vectors


class OllamaEmbeddingService:
    """基于 Ollama 本地服务的 embedding 服务封装。"""

    def __init__(self, model: str, base_url: str, timeout: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """优先调用新版本 `/api/embed`，必要时回退到旧版 `/api/embeddings`。"""
        if not texts:
            return []

        try:
            return self._embed_with_batch_endpoint(texts)
        except Exception:
            return [self._embed_single_text(text) for text in texts]

    def _embed_with_batch_endpoint(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f'{self.base_url}/api/embed',
                json={'model': self.model, 'input': texts},
            )
            response.raise_for_status()
            payload = response.json()

        embeddings = payload.get('embeddings')
        if not isinstance(embeddings, list):
            raise RuntimeError('Ollama /api/embed 返回格式异常。')
        return [[float(value) for value in item] for item in embeddings]

    def _embed_single_text(self, text: str) -> list[float]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f'{self.base_url}/api/embeddings',
                json={'model': self.model, 'prompt': text},
            )
            response.raise_for_status()
            payload = response.json()

        embedding = payload.get('embedding')
        if not isinstance(embedding, list):
            raise RuntimeError('Ollama /api/embeddings 返回格式异常。')
        return [float(value) for value in embedding]


class SentenceTransformerEmbeddingService:
    """基于 sentence-transformers 的本地 embedding 服务封装。"""

    def __init__(self, model_name: str, local_files_only: bool, model_path: str | None = None) -> None:
        self.model_name = model_name.strip()
        self.model_path = (model_path or '').strip() or None
        self.local_files_only = local_files_only
        self._model = None
        self._load_error: RuntimeError | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """把文本列表编码为向量列表。"""
        if not texts:
            return []

        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def _load_model(self):
        """延迟加载 sentence-transformers 模型。"""
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise self._load_error

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            error = RuntimeError('未安装 sentence-transformers，无法启用本地 embedding。')
            self._load_error = error
            raise error from exc

        model_source, source_kind = _resolve_model_source(
            model_name=self.model_name,
            model_path=self.model_path,
        )
        init_kwargs = {} if source_kind == 'local_path' else {'local_files_only': self.local_files_only}

        try:
            self._model = SentenceTransformer(model_source, **init_kwargs)
        except Exception as exc:
            if source_kind == 'local_path':
                error = RuntimeError(
                    f'加载本地 embedding 模型失败：{model_source}。'
                    '请确认目录存在且包含完整的 sentence-transformers 模型文件。'
                )
            elif self.local_files_only:
                error = RuntimeError(
                    f'加载 embedding 模型失败：{model_source}。'
                    '当前为本地缓存模式，不会联网下载；'
                    '请先手动下载模型，或切换到 service / ollama 模式。'
                )
            else:
                error = RuntimeError(
                    f'加载 embedding 模型失败：{model_source}，请确认模型名、本地目录或网络环境。'
                )
            self._load_error = error
            raise error from exc

        return self._model



def set_embedding_provider_override(provider: str | None) -> None:
    """设置本次命令运行时的 embedding 提供方覆盖值。"""
    global _PROVIDER_OVERRIDE
    if provider is None:
        _PROVIDER_OVERRIDE = None
        clear_embedding_service_cache()
        return

    normalized = _normalize_provider(provider)
    _PROVIDER_OVERRIDE = normalized
    clear_embedding_service_cache()



def get_embedding_settings() -> EmbeddingSettings:
    """读取当前 embedding 配置，并结合命令行覆盖值生成最终设置。"""
    provider = _normalize_provider(_PROVIDER_OVERRIDE or DEFAULT_EMBEDDING_PROVIDER)
    if provider == 'service':
        return EmbeddingSettings(
            provider='service',
            model=DEFAULT_EMBEDDING_MODEL,
            base_url=DEFAULT_EMBEDDING_BASE_URL,
            api_key=DEFAULT_EMBEDDING_API_KEY,
            timeout=DEFAULT_EMBEDDING_TIMEOUT,
        )
    if provider == 'ollama':
        return EmbeddingSettings(
            provider='ollama',
            model=DEFAULT_OLLAMA_EMBEDDING_MODEL,
            base_url=DEFAULT_OLLAMA_BASE_URL,
            api_key=None,
            timeout=DEFAULT_EMBEDDING_TIMEOUT,
        )
    return EmbeddingSettings(
        provider='sentence_transformers',
        model=DEFAULT_LOCAL_EMBEDDING_MODEL,
        base_url=None,
        api_key=None,
        timeout=DEFAULT_EMBEDDING_TIMEOUT,
        local_files_only=DEFAULT_LOCAL_EMBEDDING_ONLY,
        model_path=DEFAULT_LOCAL_EMBEDDING_MODEL_PATH or None,
    )



def get_embedding_service():
    """返回当前命令应该复用的 embedding 服务实例。"""
    settings = get_embedding_settings()
    return _build_embedding_service(
        settings.provider,
        settings.model,
        settings.base_url or '',
        settings.api_key or '',
        settings.timeout,
        settings.local_files_only,
        settings.model_path or '',
    )


def check_embedding_health(probe_text: str = 'health check') -> EmbeddingHealthResult:
    """执行一次最小化 embedding 探测，检查当前 provider 是否可用。"""
    settings = get_embedding_settings()
    started_at = perf_counter()
    try:
        service = get_embedding_service()
        vectors = service.embed_texts([probe_text])
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        vector_size = len(vectors[0]) if vectors and vectors[0] else 0
        return EmbeddingHealthResult(
            provider=settings.provider,
            model=settings.model_path or settings.model,
            base_url=settings.base_url,
            healthy=True,
            latency_ms=latency_ms,
            vector_size=vector_size,
            message='embedding 服务可用。',
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return EmbeddingHealthResult(
            provider=settings.provider,
            model=settings.model_path or settings.model,
            base_url=settings.base_url,
            healthy=False,
            latency_ms=latency_ms,
            vector_size=None,
            message='embedding 服务不可用。',
            error=str(exc),
        )


@lru_cache(maxsize=8)
def _build_embedding_service(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    local_files_only: bool,
    model_path: str,
):
    """按 provider + 配置维度缓存 embedding 服务实例。"""
    if provider == 'service':
        if not base_url:
            raise RuntimeError('未配置 REPOINSIGHT_EMBEDDING_BASE_URL，无法使用 service embedding。')
        return OpenAICompatibleEmbeddingService(
            model=model,
            base_url=base_url,
            api_key=api_key or None,
            timeout=timeout,
        )
    if provider == 'ollama':
        if not base_url:
            raise RuntimeError('未配置 REPOINSIGHT_OLLAMA_BASE_URL，无法使用 Ollama embedding。')
        return OllamaEmbeddingService(
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
    return SentenceTransformerEmbeddingService(
        model_name=model,
        local_files_only=local_files_only,
        model_path=model_path or None,
    )



def clear_embedding_service_cache() -> None:
    """清空 embedding 服务缓存，便于测试或重新读取环境变量。"""
    cache_clear = getattr(_build_embedding_service, 'cache_clear', None)
    if callable(cache_clear):
        cache_clear()



def _resolve_model_source(model_name: str, model_path: str | None) -> tuple[str, str]:
    """解析最终应该加载的本地模型来源，优先使用显式本地目录。"""
    if model_path:
        return str(Path(model_path).expanduser()), 'local_path'

    expanded_name = Path(model_name).expanduser()
    if expanded_name.exists():
        return str(expanded_name), 'local_path'

    return model_name, 'model_name'
