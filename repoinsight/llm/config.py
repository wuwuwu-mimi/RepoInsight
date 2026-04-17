import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LlmSettings(BaseModel):
    """表示基于 `.env` 读取到的 LLM 配置。"""

    # LLM 提供方类型，当前支持 openai 和 ollama。
    provider: str = Field(default='openai', description='LLM 提供方')

    # 当前使用的模型名称。
    model: str = Field(..., description='模型名称')

    # OpenAI 兼容接口的基础地址。
    base_url: str = Field(..., description='接口基础地址')

    # 访问模型服务所需的 API Key。
    api_key: str = Field(..., description='接口密钥')

    # 请求超时时间，单位为秒。
    timeout_seconds: float = Field(default=30.0, description='请求超时秒数')

    # 采样温度，越低越稳定。
    temperature: float = Field(default=0.2, description='采样温度')


def load_project_env() -> None:
    """从项目根目录加载 `.env`，未找到时静默跳过。"""
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path, override=False)


@lru_cache(maxsize=1)
def get_llm_settings() -> LlmSettings | None:
    """读取并缓存当前 LLM 配置；若未配置完整则返回 None。"""
    load_project_env()

    provider = (_get_first_env('REPOINSIGHT_LLM_PROVIDER', 'OPENAI_PROVIDER') or '').strip().lower()
    model = _get_first_env('REPOINSIGHT_LLM_MODEL', 'OPENAI_MODEL')
    base_url = _get_first_env('REPOINSIGHT_LLM_BASE_URL', 'OPENAI_BASE_URL')

    if not model or not base_url:
        return None

    if not provider:
        provider = _infer_provider_from_base_url(base_url)

    normalized_provider = provider if provider in {'openai', 'ollama'} else 'openai'

    # Ollama 本地模型允许不配置真实密钥，这里统一回退为固定占位值，
    # 避免误用项目里已有的 OpenAI 密钥配置。
    if normalized_provider == 'ollama':
        api_key = 'ollama'
    else:
        api_key = _get_first_env('REPOINSIGHT_LLM_API_KEY', 'OPENAI_API_KEY')
        if not api_key:
            return None

    timeout_seconds = _get_float_env(
        primary_key='REPOINSIGHT_LLM_TIMEOUT',
        fallback_key='OPENAI_TIMEOUT',
        default=30.0,
    )
    temperature = _get_float_env(
        primary_key='REPOINSIGHT_LLM_TEMPERATURE',
        fallback_key='OPENAI_TEMPERATURE',
        default=0.2,
    )

    return LlmSettings(
        provider=normalized_provider,
        model=model.strip(),
        base_url=base_url.strip().rstrip('/'),
        api_key=api_key.strip(),
        timeout_seconds=timeout_seconds,
        temperature=temperature,
    )


def clear_llm_settings_cache() -> None:
    """清空 LLM 配置缓存，便于测试或重新加载 `.env`。"""
    get_llm_settings.cache_clear()


def get_llm_config_help_text() -> str:
    """返回当前项目支持的 `.env` 配置说明。"""
    return (
        '请在项目根目录的 .env 中配置 REPOINSIGHT_LLM_PROVIDER、'
        'REPOINSIGHT_LLM_MODEL、'
        'REPOINSIGHT_LLM_BASE_URL、REPOINSIGHT_LLM_API_KEY；'
        '若 provider=ollama，则本地模型可不填 API Key。'
    )


def _get_first_env(*keys: str) -> str | None:
    """按优先顺序读取第一个非空环境变量。"""
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value
    return None


def _get_float_env(primary_key: str, fallback_key: str, default: float) -> float:
    """读取浮点型环境变量；格式不合法时回退默认值。"""
    raw_value = _get_first_env(primary_key, fallback_key)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def _infer_provider_from_base_url(base_url: str) -> str:
    """根据 base_url 轻量推断当前更像哪个提供方。"""
    lowered = base_url.lower()
    if 'localhost:11434' in lowered or '127.0.0.1:11434' in lowered or 'ollama' in lowered:
        return 'ollama'
    return 'openai'
