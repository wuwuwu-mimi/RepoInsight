import os

from repoinsight.llm.config import clear_llm_settings_cache, get_llm_settings


def test_get_llm_settings_reads_repoinsight_env_vars() -> None:
    original_values = {
        'REPOINSIGHT_LLM_PROVIDER': os.environ.get('REPOINSIGHT_LLM_PROVIDER'),
        'REPOINSIGHT_LLM_MODEL': os.environ.get('REPOINSIGHT_LLM_MODEL'),
        'REPOINSIGHT_LLM_BASE_URL': os.environ.get('REPOINSIGHT_LLM_BASE_URL'),
        'REPOINSIGHT_LLM_API_KEY': os.environ.get('REPOINSIGHT_LLM_API_KEY'),
        'REPOINSIGHT_LLM_TIMEOUT': os.environ.get('REPOINSIGHT_LLM_TIMEOUT'),
        'REPOINSIGHT_LLM_TEMPERATURE': os.environ.get('REPOINSIGHT_LLM_TEMPERATURE'),
    }
    try:
        os.environ['REPOINSIGHT_LLM_PROVIDER'] = 'openai'
        os.environ['REPOINSIGHT_LLM_MODEL'] = 'demo-model'
        os.environ['REPOINSIGHT_LLM_BASE_URL'] = 'https://example.com/v1'
        os.environ['REPOINSIGHT_LLM_API_KEY'] = 'demo-key'
        os.environ['REPOINSIGHT_LLM_TIMEOUT'] = '12'
        os.environ['REPOINSIGHT_LLM_TEMPERATURE'] = '0.4'
        clear_llm_settings_cache()

        settings = get_llm_settings()

        assert settings is not None
        assert settings.provider == 'openai'
        assert settings.model == 'demo-model'
        assert settings.base_url == 'https://example.com/v1'
        assert settings.api_key == 'demo-key'
        assert settings.timeout_seconds == 12.0
        assert settings.temperature == 0.4
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_llm_settings_cache()


def test_get_llm_settings_supports_ollama_without_real_api_key() -> None:
    original_values = {
        'REPOINSIGHT_LLM_PROVIDER': os.environ.get('REPOINSIGHT_LLM_PROVIDER'),
        'REPOINSIGHT_LLM_MODEL': os.environ.get('REPOINSIGHT_LLM_MODEL'),
        'REPOINSIGHT_LLM_BASE_URL': os.environ.get('REPOINSIGHT_LLM_BASE_URL'),
        'REPOINSIGHT_LLM_API_KEY': os.environ.get('REPOINSIGHT_LLM_API_KEY'),
    }
    try:
        os.environ['REPOINSIGHT_LLM_PROVIDER'] = 'ollama'
        os.environ['REPOINSIGHT_LLM_MODEL'] = 'qwen2.5:7b'
        os.environ['REPOINSIGHT_LLM_BASE_URL'] = 'http://127.0.0.1:11434'
        os.environ.pop('REPOINSIGHT_LLM_API_KEY', None)
        clear_llm_settings_cache()

        settings = get_llm_settings()

        assert settings is not None
        assert settings.provider == 'ollama'
        assert settings.model == 'qwen2.5:7b'
        assert settings.base_url == 'http://127.0.0.1:11434'
        assert settings.api_key == 'ollama'
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_llm_settings_cache()
