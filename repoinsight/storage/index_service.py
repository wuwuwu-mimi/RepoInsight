from pathlib import Path

from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.local_knowledge_store import (
    DEFAULT_KNOWLEDGE_DIR,
    remove_repo_documents,
    save_repo_documents,
)


def index_analysis_result(
    result: AnalysisRunResult,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> Path:
    """把一次分析结果构建成知识文档并写入本地知识库。"""
    repo_id = result.repo_info.repo_model.full_name
    documents = build_knowledge_documents(result)
    return save_repo_documents(repo_id=repo_id, documents=documents, target_dir=target_dir)


def remove_indexed_repo(
    repo_id: str,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> bool:
    """删除指定仓库在本地知识库中的索引。"""
    return remove_repo_documents(repo_id=repo_id, target_dir=target_dir)
