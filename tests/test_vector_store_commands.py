import io
from contextlib import redirect_stdout
from pathlib import Path

import repoinsight.cli.main as cli_main
import repoinsight.storage.index_service as index_service_module
from repoinsight.models.rag_model import KnowledgeDocument
from repoinsight.models.vector_store_model import VectorStoreHealthResult, VectorStoreRebuildResult


def test_check_vector_store_health_reports_probe_error() -> None:
    original_runtime_available = index_service_module.is_chroma_runtime_available
    original_load_all_documents = index_service_module.load_all_documents
    original_get_project_root = index_service_module.get_project_root
    original_get_chroma_document_count = index_service_module.get_chroma_document_count
    try:
        index_service_module.is_chroma_runtime_available = lambda: True
        index_service_module.load_all_documents = lambda target_dir='data/knowledge': [
            KnowledgeDocument(
                doc_id='demo/sample::repo_summary',
                repo_id='demo/sample',
                doc_type='repo_summary',
                title='demo',
                content='demo',
                metadata={},
            )
        ]
        index_service_module.get_project_root = lambda: Path('E:/PythonProject/RepoInsight')

        def fail_count(target_dir='data/chroma', collection_name='repoinsight_documents'):
            raise RuntimeError('disk I/O error')

        index_service_module.get_chroma_document_count = fail_count

        result = index_service_module.check_vector_store_health()

        assert result.healthy is False
        assert result.runtime_available is True
        assert result.knowledge_repo_count == 1
        assert '重建' in result.message
        assert 'disk I/O error' in (result.error or '')
    finally:
        index_service_module.is_chroma_runtime_available = original_runtime_available
        index_service_module.load_all_documents = original_load_all_documents
        index_service_module.get_project_root = original_get_project_root
        index_service_module.get_chroma_document_count = original_get_chroma_document_count


def test_rebuild_vector_store_reindexes_local_documents() -> None:
    original_overview = index_service_module.get_vector_store_rebuild_overview
    original_load_all_documents = index_service_module.load_all_documents
    original_remove_directory = index_service_module._remove_directory_with_retry
    original_index_documents = index_service_module.index_documents_to_chroma
    try:
        index_service_module.get_vector_store_rebuild_overview = lambda **kwargs: VectorStoreHealthResult(
            runtime_available=True,
            store_exists=True,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            knowledge_repo_count=1,
            knowledge_document_count=2,
            healthy=False,
            message='向量库探测失败，建议执行重建。',
            error='disk I/O error',
        )
        documents = [
            KnowledgeDocument(
                doc_id='demo/sample::repo_summary',
                repo_id='demo/sample',
                doc_type='repo_summary',
                title='demo',
                content='demo',
                metadata={},
            ),
            KnowledgeDocument(
                doc_id='demo/sample::readme_summary',
                repo_id='demo/sample',
                doc_type='readme_summary',
                title='readme',
                content='readme',
                metadata={},
            ),
        ]
        captured = {'count': 0}
        index_service_module.load_all_documents = lambda target_dir='data/knowledge': documents
        index_service_module._remove_directory_with_retry = lambda path: True

        def fake_index_documents(items, target_dir='data/chroma', collection_name='repoinsight_documents'):
            captured['count'] = len(items)
            return True

        index_service_module.index_documents_to_chroma = fake_index_documents

        result = index_service_module.rebuild_vector_store()

        assert result.success is True
        assert result.removed_existing_store is True
        assert result.indexed_repo_count == 1
        assert result.indexed_document_count == 2
        assert captured['count'] == 2
    finally:
        index_service_module.get_vector_store_rebuild_overview = original_overview
        index_service_module.load_all_documents = original_load_all_documents
        index_service_module._remove_directory_with_retry = original_remove_directory
        index_service_module.index_documents_to_chroma = original_index_documents


def test_vector_health_command_prints_summary() -> None:
    original_health = cli_main.check_vector_store_health
    try:
        cli_main.check_vector_store_health = lambda: VectorStoreHealthResult(
            runtime_available=True,
            store_exists=True,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            knowledge_repo_count=1,
            knowledge_document_count=41,
            indexed_repo_count=1,
            indexed_document_count=41,
            healthy=True,
            message='向量库状态正常。',
        )
        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.vector_health()
        rendered = output.getvalue()
        assert '向量库健康检查' in rendered
        assert '向量库状态正常' in rendered
    finally:
        cli_main.check_vector_store_health = original_health


def test_rebuild_vector_command_prints_success_message() -> None:
    original_health = cli_main.check_vector_store_health
    original_overview = cli_main.get_vector_store_rebuild_overview
    original_rebuild = cli_main.rebuild_vector_store
    try:
        cli_main.get_vector_store_rebuild_overview = lambda: VectorStoreHealthResult(
            runtime_available=True,
            store_exists=True,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            knowledge_repo_count=1,
            knowledge_document_count=41,
            indexed_repo_count=0,
            indexed_document_count=0,
            healthy=False,
            message='已跳过深度探测，准备直接重建向量库。',
            error=None,
        )
        cli_main.check_vector_store_health = lambda: VectorStoreHealthResult(
            runtime_available=True,
            store_exists=True,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            knowledge_repo_count=1,
            knowledge_document_count=41,
            indexed_repo_count=0,
            indexed_document_count=0,
            healthy=False,
            message='向量库探测失败，建议执行重建。',
            error='disk I/O error',
        )
        cli_main.rebuild_vector_store = lambda health_snapshot=None: VectorStoreRebuildResult(
            healthy_before=False,
            removed_existing_store=True,
            success=True,
            indexed_repo_count=1,
            indexed_document_count=41,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            message='向量库已根据本地知识文档重建完成。',
        )

        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.rebuild_vector()

        rendered = output.getvalue()
        assert '重建前状态' in rendered
        assert '重建后状态' in rendered
        assert '向量库已根据本地知识文档重建完成' in rendered
    finally:
        cli_main.check_vector_store_health = original_health
        cli_main.get_vector_store_rebuild_overview = original_overview
        cli_main.rebuild_vector_store = original_rebuild


def test_rebuild_vector_command_applies_ollama_embedding_mode() -> None:
    original_apply_mode = cli_main._apply_embedding_mode
    original_overview = cli_main.get_vector_store_rebuild_overview
    original_rebuild = cli_main.rebuild_vector_store
    original_health = cli_main.check_vector_store_health
    captured = {'mode': None}
    try:
        cli_main._apply_embedding_mode = lambda mode: captured.__setitem__('mode', mode) or True
        cli_main.get_vector_store_rebuild_overview = lambda: VectorStoreHealthResult(
            runtime_available=True,
            store_exists=False,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            knowledge_repo_count=0,
            knowledge_document_count=0,
            indexed_repo_count=0,
            indexed_document_count=0,
            healthy=False,
            message='skip',
            error=None,
        )
        cli_main.rebuild_vector_store = lambda health_snapshot=None: VectorStoreRebuildResult(
            healthy_before=False,
            removed_existing_store=False,
            success=False,
            indexed_repo_count=0,
            indexed_document_count=0,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            message='stop',
            error='x',
        )
        cli_main.check_vector_store_health = lambda: VectorStoreHealthResult(
            runtime_available=True,
            store_exists=False,
            store_path='E:/PythonProject/RepoInsight/data/chroma',
            knowledge_repo_count=0,
            knowledge_document_count=0,
            indexed_repo_count=0,
            indexed_document_count=0,
            healthy=False,
            message='skip',
            error=None,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.rebuild_vector(embedding_mode='ollama')

        assert captured['mode'] == 'ollama'
    finally:
        cli_main._apply_embedding_mode = original_apply_mode
        cli_main.get_vector_store_rebuild_overview = original_overview
        cli_main.rebuild_vector_store = original_rebuild
        cli_main.check_vector_store_health = original_health


def test_embedding_health_command_renders_result() -> None:
    original_apply_mode = cli_main._apply_embedding_mode
    original_check_embedding_health = cli_main.check_embedding_health
    try:
        cli_main._apply_embedding_mode = lambda mode: True
        cli_main.check_embedding_health = lambda: type(
            'EmbeddingHealthResultStub',
            (),
            {
                'healthy': True,
                'provider': 'ollama',
                'model': 'nomic-embed-text',
                'base_url': 'http://127.0.0.1:11434',
                'latency_ms': 12.5,
                'vector_size': 768,
                'message': 'embedding 服务可用。',
                'error': None,
            },
        )()

        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.embedding_health(embedding_mode='ollama')

        rendered = output.getvalue()
        assert 'Embedding 健康检查' in rendered
        assert 'nomic-embed-text' in rendered
        assert 'ollama' in rendered
    finally:
        cli_main._apply_embedding_mode = original_apply_mode
        cli_main.check_embedding_health = original_check_embedding_health
