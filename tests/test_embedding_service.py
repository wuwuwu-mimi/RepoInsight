import sys
import types
from pathlib import Path
import shutil

import repoinsight.storage.embedding_service as embedding_module
from repoinsight.storage.embedding_service import (
    EmbeddingProviderSwitchError,
    SentenceTransformerEmbeddingService,
    check_embedding_health,
    clear_embedding_service_cache,
    get_embedding_service,
    get_embedding_settings,
    set_embedding_provider_override,
)


def test_embedding_settings_defaults_to_service_provider() -> None:
    clear_embedding_service_cache()
    set_embedding_provider_override(None)

    settings = get_embedding_settings()

    assert settings.provider == 'service'
    assert settings.model == 'text-embedding-3-small'
    assert settings.base_url == 'https://api.openai.com/v1'



def test_embedding_service_switches_to_ollama_provider() -> None:
    clear_embedding_service_cache()
    original_builder = embedding_module._build_embedding_service
    captured: dict[str, object] = {}

    def fake_builder(provider, model, base_url, api_key, timeout, local_files_only, model_path):
        captured['provider'] = provider
        captured['model'] = model
        captured['base_url'] = base_url
        return object()

    embedding_module._build_embedding_service = fake_builder
    try:
        set_embedding_provider_override('ollama')
        get_embedding_service()
        assert captured['provider'] == 'ollama'
        assert captured['model'] == embedding_module.DEFAULT_OLLAMA_EMBEDDING_MODEL
        assert captured['base_url'] == embedding_module.DEFAULT_OLLAMA_BASE_URL
    finally:
        set_embedding_provider_override(None)
        embedding_module._build_embedding_service = original_builder
        clear_embedding_service_cache()



def test_embedding_service_supports_explicit_local_model_path() -> None:
    captured: dict[str, object] = {}
    original_module = sys.modules.get('sentence_transformers')

    class _FakeVector:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs) -> None:
            captured['model_name'] = model_name
            captured['kwargs'] = kwargs

        def encode(self, texts: list[str], normalize_embeddings: bool = True):
            return [_FakeVector([0.3, 0.4]) for _ in texts]

    fake_module = types.ModuleType('sentence_transformers')
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    sys.modules['sentence_transformers'] = fake_module

    local_model_dir = Path('tests/.tmp_embedding_model/bge-small-zh')
    shutil.rmtree(local_model_dir.parent, ignore_errors=True)
    local_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        service = SentenceTransformerEmbeddingService(
            model_name='ignored-remote-name',
            model_path=str(local_model_dir),
            local_files_only=False,
        )
        vectors = service.embed_texts(['hello'])
        assert vectors == [[0.3, 0.4]]
        assert captured['model_name'] == str(local_model_dir)
        assert captured['kwargs'] == {}
    finally:
        shutil.rmtree(local_model_dir.parent, ignore_errors=True)
        if original_module is None:
            sys.modules.pop('sentence_transformers', None)
        else:
            sys.modules['sentence_transformers'] = original_module



def test_embedding_provider_override_rejects_invalid_mode() -> None:
    try:
        set_embedding_provider_override('invalid-provider')
    except EmbeddingProviderSwitchError as exc:
        assert '不支持的 embedding 模式' in str(exc)
    else:
        raise AssertionError('expected EmbeddingProviderSwitchError')


def test_check_embedding_health_reports_success() -> None:
    original_get_settings = embedding_module.get_embedding_settings
    original_get_service = embedding_module.get_embedding_service

    class _FakeService:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            assert texts == ['health check']
            return [[0.1, 0.2, 0.3]]

    try:
        embedding_module.get_embedding_settings = lambda: embedding_module.EmbeddingSettings(
            provider='ollama',
            model='nomic-embed-text',
            base_url='http://127.0.0.1:11434',
            timeout=30.0,
        )
        embedding_module.get_embedding_service = lambda: _FakeService()

        result = check_embedding_health()

        assert result.healthy is True
        assert result.provider == 'ollama'
        assert result.vector_size == 3
    finally:
        embedding_module.get_embedding_settings = original_get_settings
        embedding_module.get_embedding_service = original_get_service
