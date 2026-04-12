import json
import os
from pathlib import Path

from repoinsight.ingest.repo_cache import get_project_root
from repoinsight.models.rag_model import KnowledgeDocument, SearchHit
from repoinsight.storage.embedding_service import get_embedding_service


# Chroma 默认持久化目录与集合名。
DEFAULT_CHROMA_DIR = os.getenv('REPOINSIGHT_CHROMA_DIR', 'data/chroma')
DEFAULT_COLLECTION_NAME = os.getenv('REPOINSIGHT_CHROMA_COLLECTION', 'repoinsight_documents')


def index_documents_to_chroma(
    documents: list[KnowledgeDocument],
    target_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """把知识文档写入 Chroma 向量库。"""
    if not documents:
        return False

    collection = _get_collection(target_dir=target_dir, collection_name=collection_name)
    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_texts([item.content for item in documents])

    collection.upsert(
        ids=[item.doc_id for item in documents],
        documents=[item.content for item in documents],
        embeddings=embeddings,
        metadatas=[_serialize_metadata(item) for item in documents],
    )
    return True


def search_documents_in_chroma(
    query: str,
    top_k: int = 5,
    target_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[SearchHit]:
    """在 Chroma 中执行向量检索。"""
    collection = _get_collection(target_dir=target_dir, collection_name=collection_name)
    if collection.count() == 0:
        return []

    embedding_service = get_embedding_service()
    query_embedding = embedding_service.embed_texts([query])[0]
    raw_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'],
    )
    return _build_search_hits(query=query, raw_result=raw_result)


def remove_repo_documents_from_chroma(
    repo_id: str,
    target_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """从 Chroma 中删除指定仓库的所有文档。"""
    collection = _get_collection(target_dir=target_dir, collection_name=collection_name)
    collection.delete(where={'repo_id': repo_id})
    return True


def list_repo_ids_in_chroma(
    target_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> set[str]:
    """列出当前 Chroma 中出现过的仓库标识。"""
    collection = _get_collection(target_dir=target_dir, collection_name=collection_name)
    if collection.count() == 0:
        return set()

    raw_result = collection.get(include=['metadatas'])
    metadatas = raw_result.get('metadatas') or []
    repo_ids: set[str] = set()
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue
        repo_id = str(metadata.get('repo_id', '')).strip()
        if repo_id:
            repo_ids.add(repo_id)
    return repo_ids


def is_chroma_runtime_available() -> bool:
    """判断当前环境是否已安装 Chroma。"""
    try:
        import chromadb  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _get_collection(target_dir: str, collection_name: str):
    """获取 Chroma collection，不存在时自动创建。"""
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError('未安装 chromadb，无法启用向量检索。') from exc

    chroma_root = _get_chroma_root(target_dir)
    chroma_root.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_root))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'},
    )


def _get_chroma_root(target_dir: str) -> Path:
    """返回 Chroma 持久化目录。"""
    return get_project_root() / target_dir


def _serialize_metadata(document: KnowledgeDocument) -> dict[str, str | int | float | bool]:
    """把知识文档元数据转成适合写入 Chroma 的扁平结构。"""
    metadata_json = json.dumps(document.metadata, ensure_ascii=False)
    return {
        'doc_id': document.doc_id,
        'repo_id': document.repo_id,
        'doc_type': document.doc_type,
        'title': document.title,
        'source_path': document.source_path or '',
        'project_type': str(document.metadata.get('project_type', '')),
        'primary_language': str(document.metadata.get('primary_language', '')),
        'frameworks_text': _join_metadata_list(document.metadata.get('frameworks')),
        'topics_text': _join_metadata_list(document.metadata.get('topics')),
        'metadata_json': metadata_json,
    }


def _build_search_hits(query: str, raw_result: dict) -> list[SearchHit]:
    """把 Chroma query 返回结果转换为 SearchHit 列表。"""
    documents = raw_result.get('documents') or [[]]
    metadatas = raw_result.get('metadatas') or [[]]
    distances = raw_result.get('distances') or [[]]
    ids = raw_result.get('ids') or [[]]

    hits: list[SearchHit] = []
    for doc_id, content, metadata, distance in zip(
        ids[0],
        documents[0],
        metadatas[0],
        distances[0],
        strict=False,
    ):
        knowledge_document = _deserialize_document(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {},
        )
        hits.append(
            SearchHit(
                document=knowledge_document,
                score=round(_distance_to_score(distance), 4),
                snippet=_build_snippet(content=content, query=query),
            )
        )

    return hits


def _deserialize_document(doc_id: str, content: str, metadata: dict) -> KnowledgeDocument:
    """从 Chroma 元数据恢复 KnowledgeDocument。"""
    metadata_json = metadata.get('metadata_json', '{}')
    try:
        original_metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        original_metadata = {}

    return KnowledgeDocument(
        doc_id=doc_id,
        repo_id=str(metadata.get('repo_id', '')),
        doc_type=str(metadata.get('doc_type', '')),
        title=str(metadata.get('title', '')),
        content=content,
        source_path=str(metadata.get('source_path', '')) or None,
        metadata=original_metadata,
    )


def _distance_to_score(distance: float | int | None) -> float:
    """把 Chroma 返回的距离转换为越大越好的相似度分值。"""
    if distance is None:
        return 0.0
    distance_value = float(distance)
    return 1.0 / (1.0 + distance_value)


def _build_snippet(content: str, query: str, max_chars: int = 180) -> str:
    """从命中文档中提取一个便于展示的短片段。"""
    normalized_content = content.replace('\n', ' ')
    lowered_content = normalized_content.lower()
    lowered_query = query.lower()
    hit_index = lowered_content.find(lowered_query)

    if hit_index < 0:
        return normalized_content[:max_chars] + ('...' if len(normalized_content) > max_chars else '')

    start = max(0, hit_index - 40)
    end = min(len(normalized_content), hit_index + max_chars - 40)
    snippet = normalized_content[start:end].strip()
    if start > 0:
        snippet = '...' + snippet
    if end < len(normalized_content):
        snippet = snippet + '...'
    return snippet


def _join_metadata_list(value: object) -> str:
    """把 metadata 中的列表值压平为字符串，便于 Chroma 过滤和展示。"""
    if not isinstance(value, list):
        return ''
    return ', '.join(str(item) for item in value)
