import io
from contextlib import redirect_stdout

import repoinsight.cli.main as cli_main
import repoinsight.storage.index_service as index_service_module


def test_remove_vector_indexed_repo_returns_true_when_chroma_delete_succeeds() -> None:
    original_remove = index_service_module.remove_repo_documents_from_chroma
    try:
        index_service_module.remove_repo_documents_from_chroma = lambda repo_id, target_dir='data/chroma': repo_id == 'demo/sample'
        assert index_service_module.remove_vector_indexed_repo('demo/sample') is True
        assert index_service_module.remove_vector_indexed_repo('demo/missing') is False
    finally:
        index_service_module.remove_repo_documents_from_chroma = original_remove


def test_remove_vector_command_prints_success_message() -> None:
    original_remove = cli_main.remove_vector_indexed_repo
    try:
        cli_main.remove_vector_indexed_repo = lambda repo_id: repo_id == 'demo/sample'
        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.remove_vector('demo/sample')
        assert '向量索引已删除' in output.getvalue()
    finally:
        cli_main.remove_vector_indexed_repo = original_remove


def test_remove_command_deletes_artifacts_by_default() -> None:
    original_remove_cloned_repo = cli_main.remove_cloned_repo
    original_remove_markdown_report = cli_main.remove_markdown_report
    original_remove_json_report = cli_main.remove_json_report
    original_remove_pdf_report = cli_main.remove_pdf_report
    original_remove_llm_context_text = cli_main.remove_llm_context_text
    original_remove_indexed_repo = cli_main.remove_indexed_repo
    calls: list[str] = []
    try:
        cli_main.remove_cloned_repo = lambda repo_id: calls.append(f'clone:{repo_id}') or True
        cli_main.remove_markdown_report = lambda repo_id, output_dir='reports': calls.append(f'md:{repo_id}:{output_dir}') or True
        cli_main.remove_json_report = lambda repo_id, output_dir='reports': calls.append(f'json:{repo_id}:{output_dir}') or True
        cli_main.remove_pdf_report = lambda repo_id, output_dir='reports': calls.append(f'pdf:{repo_id}:{output_dir}') or True
        cli_main.remove_llm_context_text = lambda repo_id, output_dir='reports': calls.append(f'llm:{repo_id}:{output_dir}') or True
        cli_main.remove_indexed_repo = lambda repo_id: calls.append(f'knowledge:{repo_id}') or True

        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.remove('demo/sample')

        rendered = output.getvalue()
        assert 'demo/sample' in rendered
        assert '分析产物已删除' in rendered
        assert calls == [
            'clone:demo/sample',
            'md:demo/sample:reports',
            'json:demo/sample:reports',
            'pdf:demo/sample:reports',
            'llm:demo/sample:reports',
            'knowledge:demo/sample',
        ]
    finally:
        cli_main.remove_cloned_repo = original_remove_cloned_repo
        cli_main.remove_markdown_report = original_remove_markdown_report
        cli_main.remove_json_report = original_remove_json_report
        cli_main.remove_pdf_report = original_remove_pdf_report
        cli_main.remove_llm_context_text = original_remove_llm_context_text
        cli_main.remove_indexed_repo = original_remove_indexed_repo
