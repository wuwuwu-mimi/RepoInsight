from pathlib import Path
import json
import shutil

import repoinsight.ingest.repo_cache as repo_cache_module
import repoinsight.storage.chroma_store as chroma_store_module


def test_list_cloned_repos_includes_artifact_and_vector_statuses() -> None:
    temp_root = Path('data/test_repo_list_status')
    shutil.rmtree(temp_root, ignore_errors=True)
    clone_root = temp_root / 'clone'
    report_root = temp_root / 'reports'
    knowledge_root = temp_root / 'data' / 'knowledge'

    try:
        (clone_root / 'demo' / 'sample' / '.git').mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
        (knowledge_root / 'demo').mkdir(parents=True, exist_ok=True)

        (report_root / 'demo__sample.md').write_text('# sample\n', encoding='utf-8')
        (report_root / 'demo__sample.json').write_text('{}\n', encoding='utf-8')
        (report_root / 'demo__sample.llm.txt').write_text('context\n', encoding='utf-8')
        (knowledge_root / 'demo' / 'sample.json').write_text(
            json.dumps({'repo_id': 'demo/sample', 'documents': []}, ensure_ascii=False),
            encoding='utf-8',
        )

        original_get_clone_root = repo_cache_module.get_clone_root
        original_get_project_root = repo_cache_module.get_project_root
        original_get_report_path = repo_cache_module.get_report_path
        original_list_repo_ids_in_chroma = chroma_store_module.list_repo_ids_in_chroma
        try:
            repo_cache_module.get_clone_root = lambda target_dir='clone': clone_root
            repo_cache_module.get_project_root = lambda: temp_root
            repo_cache_module.get_report_path = lambda repo_id: report_root / f"{repo_id.replace('/', '__')}.md"
            chroma_store_module.list_repo_ids_in_chroma = lambda: {'demo/sample', 'ghost/repo'}

            result = repo_cache_module.list_cloned_repos()

            assert result.total_count == 2
            sample = next(item for item in result.repos if item.repo_id == 'demo/sample')
            ghost = next(item for item in result.repos if item.repo_id == 'ghost/repo')

            assert sample.has_clone is True
            assert sample.is_git_repo is True
            assert sample.has_markdown_report is True
            assert sample.has_json_report is True
            assert sample.has_llm_context is True
            assert sample.has_knowledge is True
            assert sample.has_vector_index is True
            assert sample.asset_status == '完整'
            assert sample.local_path

            assert ghost.has_clone is False
            assert ghost.has_vector_index is True
            assert ghost.asset_status == '仅索引残留'
            assert ghost.local_path == ''
            assert ghost.has_markdown_report is False
        finally:
            repo_cache_module.get_clone_root = original_get_clone_root
            repo_cache_module.get_project_root = original_get_project_root
            repo_cache_module.get_report_path = original_get_report_path
            chroma_store_module.list_repo_ids_in_chroma = original_list_repo_ids_in_chroma
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cleanup_orphans_command_removes_orphan_assets() -> None:
    import io
    from contextlib import redirect_stdout

    import repoinsight.cli.main as cli_main
    from repoinsight.models.cache_model import CachedRepoEntry, CachedRepoListResult

    orphan_repo = CachedRepoEntry(
        repo_id='ghost/repo',
        owner='ghost',
        repo='repo',
        local_path='',
        has_clone=False,
        is_git_repo=False,
        has_report=False,
        has_markdown_report=True,
        has_json_report=True,
        has_llm_context=True,
        has_knowledge=True,
        has_vector_index=True,
        asset_status='孤儿索引+资产',
    )
    normal_repo = CachedRepoEntry(
        repo_id='demo/sample',
        owner='demo',
        repo='sample',
        local_path='E:/clone/demo/sample',
        has_clone=True,
        is_git_repo=True,
        has_report=True,
        has_markdown_report=True,
        has_json_report=True,
        has_llm_context=True,
        has_knowledge=True,
        has_vector_index=True,
        asset_status='完整',
    )

    original_list_cloned_repos = cli_main.list_cloned_repos
    original_remove_markdown_report = cli_main.remove_markdown_report
    original_remove_json_report = cli_main.remove_json_report
    original_remove_llm_context_text = cli_main.remove_llm_context_text
    original_remove_indexed_repo = cli_main.remove_indexed_repo
    calls: list[str] = []
    try:
        cli_main.list_cloned_repos = lambda: CachedRepoListResult(
            clone_root='E:/PythonProject/RepoInsight/clone',
            repos=[orphan_repo, normal_repo],
            total_count=2,
        )
        cli_main.remove_markdown_report = lambda repo_id, output_dir='reports': calls.append(f'md:{repo_id}') or True
        cli_main.remove_json_report = lambda repo_id, output_dir='reports': calls.append(f'json:{repo_id}') or True
        cli_main.remove_llm_context_text = lambda repo_id, output_dir='reports': calls.append(f'llm:{repo_id}') or True
        cli_main.remove_indexed_repo = lambda repo_id: calls.append(f'knowledge:{repo_id}') or True

        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.cleanup_orphans()

        rendered = output.getvalue()
        assert 'ghost/repo' in rendered
        assert 'demo/sample' not in rendered
        assert calls == [
            'md:ghost/repo',
            'json:ghost/repo',
            'llm:ghost/repo',
            'knowledge:ghost/repo',
        ]
    finally:
        cli_main.list_cloned_repos = original_list_cloned_repos
        cli_main.remove_markdown_report = original_remove_markdown_report
        cli_main.remove_json_report = original_remove_json_report
        cli_main.remove_llm_context_text = original_remove_llm_context_text
        cli_main.remove_indexed_repo = original_remove_indexed_repo
