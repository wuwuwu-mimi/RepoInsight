from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.models.rag_model import IndexResult
from repoinsight.storage.chroma_store import (
    DEFAULT_CHROMA_DIR,
    index_documents_to_chroma,
    remove_repo_documents_from_chroma,
)
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import (
    DEFAULT_KNOWLEDGE_DIR,
    remove_repo_documents,
    save_repo_documents,
)


def index_analysis_result(
    result: AnalysisRunResult,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> IndexResult:
    """把一次分析结果构建成知识文档，并尽量写入本地知识库与向量库。"""
    repo_id = result.repo_info.repo_model.full_name
    documents = build_knowledge_documents(result)
    local_path = save_repo_documents(repo_id=repo_id, documents=documents, target_dir=target_dir)

    try:
        vector_indexed = index_documents_to_chroma(documents)
        if vector_indexed:
            return IndexResult(
                local_path=str(local_path),
                vector_indexed=True,
                vector_backend='chroma',
                message='已同步写入 Chroma 向量索引。',
            )
    except Exception as exc:
        return IndexResult(
            local_path=str(local_path),
            vector_indexed=False,
            vector_backend='none',
            message=f'已保存本地知识文档，但向量索引未启用：{exc}',
        )

    return IndexResult(
        local_path=str(local_path),
        vector_indexed=False,
        vector_backend='none',
        message='已保存本地知识文档，但未启用向量索引。',
    )


def remove_indexed_repo(
    repo_id: str,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> bool:
    """删除指定仓库在本地知识库和向量库中的索引。"""
    local_removed = remove_repo_documents(repo_id=repo_id, target_dir=target_dir)

    try:
        vector_removed = remove_repo_documents_from_chroma(repo_id=repo_id)
    except Exception:
        vector_removed = False

    return local_removed or vector_removed


def remove_vector_indexed_repo(
    repo_id: str,
    target_dir: str = DEFAULT_CHROMA_DIR,
) -> bool:
    """仅删除指定仓库在向量库中的索引。"""
    try:
        return remove_repo_documents_from_chroma(repo_id=repo_id, target_dir=target_dir)
    except Exception:
        return False
