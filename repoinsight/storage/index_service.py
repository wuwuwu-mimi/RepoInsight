import os
import shutil
import stat
import time
from pathlib import Path

from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.models.rag_model import IndexResult
from repoinsight.models.vector_store_model import VectorStoreHealthResult, VectorStoreRebuildResult
from repoinsight.storage.chroma_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    get_chroma_document_count,
    index_documents_to_chroma,
    is_chroma_runtime_available,
    list_repo_ids_in_chroma,
    remove_repo_documents_from_chroma,
)
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import (
    DEFAULT_KNOWLEDGE_DIR,
    load_all_documents,
    remove_repo_documents,
    save_repo_documents,
)
from repoinsight.ingest.repo_cache import get_project_root


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


def check_vector_store_health(
    target_dir: str = DEFAULT_CHROMA_DIR,
    knowledge_dir: str = DEFAULT_KNOWLEDGE_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStoreHealthResult:
    """检查当前 Chroma 向量库是否健康，并返回知识库与索引的概览。"""
    store_path = get_project_root() / target_dir
    knowledge_documents = load_all_documents(target_dir=knowledge_dir)
    knowledge_repo_count = len({item.repo_id for item in knowledge_documents})
    knowledge_document_count = len(knowledge_documents)

    if not is_chroma_runtime_available():
        return VectorStoreHealthResult(
            runtime_available=False,
            store_exists=store_path.exists(),
            store_path=str(store_path),
            knowledge_repo_count=knowledge_repo_count,
            knowledge_document_count=knowledge_document_count,
            healthy=False,
            message='未检测到 chromadb 运行时，当前无法使用向量检索。',
            error='chromadb runtime unavailable',
        )

    try:
        indexed_document_count = get_chroma_document_count(
            target_dir=target_dir,
            collection_name=collection_name,
        )
        indexed_repo_ids = list_repo_ids_in_chroma(
            target_dir=target_dir,
            collection_name=collection_name,
        )
    except Exception as exc:
        return VectorStoreHealthResult(
            runtime_available=True,
            store_exists=store_path.exists(),
            store_path=str(store_path),
            knowledge_repo_count=knowledge_repo_count,
            knowledge_document_count=knowledge_document_count,
            healthy=False,
            message='向量库探测失败，建议执行重建。',
            error=str(exc),
        )

    return VectorStoreHealthResult(
        runtime_available=True,
        store_exists=store_path.exists(),
        store_path=str(store_path),
        knowledge_repo_count=knowledge_repo_count,
        knowledge_document_count=knowledge_document_count,
        indexed_repo_count=len(indexed_repo_ids),
        indexed_document_count=indexed_document_count,
        healthy=True,
        message='向量库状态正常。',
    )


def get_vector_store_rebuild_overview(
    target_dir: str = DEFAULT_CHROMA_DIR,
    knowledge_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> VectorStoreHealthResult:
    """返回重建前的轻量概览，避免因为损坏索引的深度探测拖慢重建命令。"""
    store_path = get_project_root() / target_dir
    knowledge_documents = load_all_documents(target_dir=knowledge_dir)
    return VectorStoreHealthResult(
        runtime_available=is_chroma_runtime_available(),
        store_exists=store_path.exists(),
        store_path=str(store_path),
        knowledge_repo_count=len({item.repo_id for item in knowledge_documents}),
        knowledge_document_count=len(knowledge_documents),
        healthy=False,
        message='已跳过深度探测，准备直接重建向量库。',
    )


def rebuild_vector_store(
    target_dir: str = DEFAULT_CHROMA_DIR,
    knowledge_dir: str = DEFAULT_KNOWLEDGE_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    health_snapshot: VectorStoreHealthResult | None = None,
) -> VectorStoreRebuildResult:
    """删除现有 Chroma 数据目录，并基于本地知识文档重新建立向量索引。"""
    health = health_snapshot or get_vector_store_rebuild_overview(
        target_dir=target_dir,
        knowledge_dir=knowledge_dir,
    )
    store_path = Path(health.store_path)

    if not health.runtime_available:
        return VectorStoreRebuildResult(
            healthy_before=health.healthy,
            removed_existing_store=False,
            success=False,
            indexed_repo_count=0,
            indexed_document_count=0,
            store_path=str(store_path),
            message='重建失败：当前环境未安装 chromadb。',
            error=health.error,
        )

    removed_existing_store = False
    if store_path.exists():
        _shutdown_chroma_runtime()
        removed_existing_store = _remove_directory_with_retry(store_path)
        if not removed_existing_store:
            return VectorStoreRebuildResult(
                healthy_before=health.healthy,
                removed_existing_store=False,
                success=False,
                indexed_repo_count=0,
                indexed_document_count=0,
                store_path=str(store_path),
                message='重建失败：无法删除旧的向量库目录，可能仍被占用。',
                error='failed to remove vector store directory',
            )

    knowledge_documents = load_all_documents(target_dir=knowledge_dir)
    if not knowledge_documents:
        return VectorStoreRebuildResult(
            healthy_before=health.healthy,
            removed_existing_store=removed_existing_store,
            success=True,
            indexed_repo_count=0,
            indexed_document_count=0,
            store_path=str(store_path),
            message='已清空旧向量库，但当前没有可重建的本地知识文档。',
        )

    try:
        index_documents_to_chroma(
            knowledge_documents,
            target_dir=target_dir,
            collection_name=collection_name,
        )
    except Exception as exc:
        return VectorStoreRebuildResult(
            healthy_before=health.healthy,
            removed_existing_store=removed_existing_store,
            success=False,
            indexed_repo_count=0,
            indexed_document_count=0,
            store_path=str(store_path),
            message='重建失败：写入 Chroma 向量库时出现异常。',
            error=str(exc),
        )

    repo_count = len({item.repo_id for item in knowledge_documents})
    return VectorStoreRebuildResult(
        healthy_before=health.healthy,
        removed_existing_store=removed_existing_store,
        success=True,
        indexed_repo_count=repo_count,
        indexed_document_count=len(knowledge_documents),
        store_path=str(store_path),
        message='向量库已根据本地知识文档重建完成。',
    )


def _remove_directory_with_retry(target_path: Path, retries: int = 3, delay_seconds: float = 0.2) -> bool:
    """在 Windows 上删除目录时做几次重试，降低文件句柄暂未释放导致的失败概率。"""
    for _ in range(retries):
        try:
            if not target_path.exists():
                return True
            _ensure_directory_writable(target_path)
            shutil.rmtree(target_path)
            return True
        except PermissionError:
            time.sleep(delay_seconds)
        except OSError:
            time.sleep(delay_seconds)
    return not target_path.exists()


def _ensure_directory_writable(target_path: Path) -> None:
    """递归取消只读属性，避免 Windows 删除目录时因为权限位导致失败。"""
    if not target_path.exists():
        return
    try:
        os.chmod(target_path, stat.S_IWRITE)
    except OSError:
        pass
    for root, dirs, files in os.walk(target_path):
        for name in dirs:
            path = Path(root) / name
            try:
                os.chmod(path, stat.S_IWRITE)
            except OSError:
                continue
        for name in files:
            path = Path(root) / name
            try:
                os.chmod(path, stat.S_IWRITE)
            except OSError:
                continue


def _shutdown_chroma_runtime() -> None:
    """尝试主动关闭当前进程里的 Chroma 运行时，释放 SQLite 等文件句柄。"""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except Exception:
        return

    try:
        systems = list(getattr(SharedSystemClient, '_identifier_to_system', {}).values())
        for system in systems:
            try:
                system.stop()
            except Exception:
                continue
        SharedSystemClient.clear_system_cache()
    except Exception:
        return
