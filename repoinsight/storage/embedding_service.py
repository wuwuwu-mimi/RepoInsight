import os
from functools import lru_cache


# 默认使用一个支持中英文的轻量 embedding 模型，后续可通过环境变量覆盖。
DEFAULT_EMBEDDING_MODEL = os.getenv(
    'REPOINSIGHT_EMBEDDING_MODEL',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
)


class SentenceTransformerEmbeddingService:
    """基于 sentence-transformers 的 embedding 服务封装。"""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """把文本列表编码为向量列表。"""
        if not texts:
            return []

        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def _load_model(self):
        """延迟加载 embedding 模型，避免非检索路径也强制初始化。"""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                '未安装 sentence-transformers，无法启用向量检索。'
            ) from exc

        try:
            self._model = SentenceTransformer(self.model_name,)
        except Exception as exc:
            raise RuntimeError(
                f'加载 embedding 模型失败：{self.model_name}，请确认模型名称或网络环境。'
            ) from exc

        return self._model


@lru_cache(maxsize=1)
def get_embedding_service() -> SentenceTransformerEmbeddingService:
    """返回全局复用的 embedding 服务实例。"""
    return SentenceTransformerEmbeddingService()
