import json
import os
import stat
import time
from pathlib import Path

from repoinsight.ingest.repo_cache import get_project_root, parse_repo_id
from repoinsight.models.rag_model import KnowledgeDocument


# RAG MVP 的本地知识库存放在 data/knowledge 目录中。
DEFAULT_KNOWLEDGE_DIR = 'data/knowledge'


def get_knowledge_root(target_dir: str = DEFAULT_KNOWLEDGE_DIR) -> Path:
    """返回本地知识库根目录。"""
    return get_project_root() / target_dir


def save_repo_documents(
    repo_id: str,
    documents: list[KnowledgeDocument],
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> Path:
    """把单个仓库的知识文档写入本地 JSON 文件。"""
    owner, repo = parse_repo_id(repo_id)
    knowledge_root = get_knowledge_root(target_dir)
    knowledge_root.mkdir(parents=True, exist_ok=True)

    owner_dir = knowledge_root / owner
    owner_dir.mkdir(parents=True, exist_ok=True)

    target_path = owner_dir / f'{repo}.json'
    payload = {
        'repo_id': repo_id,
        'document_count': len(documents),
        'documents': [item.model_dump() for item in documents],
    }
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return target_path


def load_all_documents(target_dir: str = DEFAULT_KNOWLEDGE_DIR) -> list[KnowledgeDocument]:
    """读取知识库中所有仓库的知识文档。"""
    knowledge_root = get_knowledge_root(target_dir)
    if not knowledge_root.exists():
        return []

    documents: list[KnowledgeDocument] = []
    for file_path in knowledge_root.rglob('*.json'):
        documents.extend(load_repo_documents_from_path(file_path))

    return documents


def load_repo_documents(
    repo_id: str,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> list[KnowledgeDocument]:
    """读取指定仓库的知识文档。"""
    target_path = get_repo_knowledge_path(repo_id, target_dir=target_dir)
    if not target_path.exists():
        return []

    return load_repo_documents_from_path(target_path)


def remove_repo_documents(repo_id: str, target_dir: str = DEFAULT_KNOWLEDGE_DIR) -> bool:
    """删除指定仓库的知识文档。"""
    target_path = get_repo_knowledge_path(repo_id, target_dir=target_dir)
    if not target_path.exists():
        return False

    removed = _remove_file_with_retry(target_path)
    if removed:
        _cleanup_empty_parents(target_path.parent, get_knowledge_root(target_dir))
    else:
        _write_tombstone_file(target_path, repo_id)
    return True


def get_repo_knowledge_path(repo_id: str, target_dir: str = DEFAULT_KNOWLEDGE_DIR) -> Path:
    """根据仓库标识返回对应的知识文档文件路径。"""
    owner, repo = parse_repo_id(repo_id)
    return get_knowledge_root(target_dir) / owner / f'{repo}.json'


def load_repo_documents_from_path(file_path: Path) -> list[KnowledgeDocument]:
    """从单个 JSON 文件中加载知识文档。"""
    try:
        payload = json.loads(file_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []

    raw_documents = payload.get('documents')
    if not isinstance(raw_documents, list):
        return []

    documents: list[KnowledgeDocument] = []
    for item in raw_documents:
        if not isinstance(item, dict):
            continue
        try:
            documents.append(KnowledgeDocument.model_validate(item))
        except Exception:
            continue

    return documents


def _cleanup_empty_parents(start_path: Path, stop_path: Path) -> None:
    """删除知识文件后，顺手清理空目录，但不会删除知识库根目录。"""
    current = start_path
    while current != stop_path:
        if not current.exists():
            current = current.parent
            continue

        try:
            next(current.iterdir())
            break
        except StopIteration:
            current.rmdir()
            current = current.parent


def _remove_file_with_retry(target_path: Path, retries: int = 3, delay_seconds: float = 0.2) -> bool:
    """在 Windows 上删除知识文件时做几次重试，成功返回 True，失败返回 False。"""
    for _ in range(retries):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            target_path.unlink()
            return True
        except PermissionError:
            time.sleep(delay_seconds)

    return False


def _write_tombstone_file(target_path: Path, repo_id: str) -> None:
    """当文件被占用无法删除时，退化为写入空文档，避免检索结果继续命中。"""
    payload = {
        'repo_id': repo_id,
        'document_count': 0,
        'documents': [],
        'deleted': True,
    }
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
