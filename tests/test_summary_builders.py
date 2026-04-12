from repoinsight.models.analysis_model import (
    AnalysisRunResult,
    CodeSymbol,
    KeyFileContent,
    ModuleRelation,
    ProjectProfile,
    SubprojectSummary,
)
from repoinsight.models.file_model import FileEntry, ScanResult, ScanStats
from repoinsight.models.repo_model import RepoInfo, RepoModel
from repoinsight.storage.document_builder import build_knowledge_documents
from repoinsight.storage.summary_builders import build_config_summaries, build_entrypoint_summaries


def _build_result(key_file_contents: list[KeyFileContent], entry_paths: list[str]) -> AnalysisRunResult:
    """构造测试用的最小分析结果。"""
    scan_result = ScanResult(
        root_path='E:/PythonProject/RepoInsight/clones/demo',
        all_files=[
            FileEntry(
                path=path,
                name=path.split('/')[-1],
                extension=f'.{path.split(".")[-1]}' if '.' in path.split('/')[-1] else None,
                size_bytes=128,
                parent_dir='/'.join(path.split('/')[:-1]),
                is_key_file=True,
            )
            for path in entry_paths
        ],
        key_files=[
            FileEntry(
                path=path,
                name=path.split('/')[-1],
                extension=f'.{path.split(".")[-1]}' if '.' in path.split('/')[-1] else None,
                size_bytes=128,
                parent_dir='/'.join(path.split('/')[:-1]),
                is_key_file=True,
            )
            for path in entry_paths
        ],
        tree_preview=entry_paths,
        stats=ScanStats(total_seen=len(entry_paths), kept_count=len(entry_paths), key_file_count=len(entry_paths)),
    )

    return AnalysisRunResult(
        repo_info=RepoInfo(
            repo_model=RepoModel(
                owner='demo',
                name='sample',
                full_name='demo/sample',
                html_url='https://github.com/demo/sample',
                default_branch='main',
                primary_language='Python',
                languages={'Python': 1000},
            ),
            readme='demo readme',
        ),
        clone_path='E:/PythonProject/RepoInsight/clones/demo__sample',
        scan_result=scan_result,
        key_file_contents=key_file_contents,
        project_profile=ProjectProfile(
            primary_language='Python',
            languages=['Python', 'TypeScript'],
            runtimes=['Node.js'],
            frameworks=['FastAPI', 'React'],
            build_tools=['Vite'],
            package_managers=['npm', 'pip'],
            test_tools=['Pytest', 'Vitest'],
            deploy_tools=['Docker'],
            entrypoints=['app.py', 'src/main.tsx'],
        ),
    )


def test_build_config_summaries_extracts_scripts_env_and_paths() -> None:
    package_json = KeyFileContent(
        path='package.json',
        size_bytes=320,
        content=(
            '{'
            '"name":"demo-web",'
            '"packageManager":"pnpm@9.0.0",'
            '"scripts":{"dev":"vite","start":"node server.js --token ${OPENAI_API_KEY}"},'
            '"dependencies":{"react":"^18.0.0","redis":"^4.0.0"},'
            '"workspaces":["packages/*"],'
            '"main":"server.js"'
            '}'
        ),
    )
    result = _build_result([package_json], ['package.json'])

    summaries = build_config_summaries(result)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.config_kind == 'package_manager'
    assert '定义脚本：dev, start' in summary.key_points
    assert '显式声明包管理器：pnpm@9.0.0' in summary.key_points
    assert 'start: node server.js --token ${OPENAI_API_KEY}' in summary.scripts_or_commands
    assert 'Redis' in summary.service_dependencies
    assert 'OPENAI_API_KEY' in summary.env_vars
    assert 'packages/*' in summary.related_paths
    assert 'server.js' in summary.related_paths


def test_build_entrypoint_summaries_extracts_web_api_signals() -> None:
    pyproject = KeyFileContent(
        path='pyproject.toml',
        size_bytes=240,
        content=(
            '[project]\n'
            'name = "demo-api"\n'
            'requires-python = ">=3.11"\n'
            '[project.scripts]\n'
            'demo = "app:app"\n'
        ),
    )
    app_file = KeyFileContent(
        path='app.py',
        size_bytes=420,
        content=(
            'from fastapi import FastAPI\n'
            'from api.router import router\n'
            'import redis\n\n'
            'app = FastAPI()\n\n'
            'def create_app():\n'
            '    return app\n'
            'app.include_router(router)\n\n'
            'if __name__ == "__main__":\n'
            '    import uvicorn\n'
            '    uvicorn.run(app)\n'
        ),
    )
    result = _build_result([pyproject, app_file], ['pyproject.toml', 'app.py'])
    result.project_profile.subprojects = [
        SubprojectSummary(
            root_path='.',
            language_scope='python',
            project_kind='service',
            config_paths=['pyproject.toml'],
            entrypoint_paths=['app.py'],
            markers=['service'],
        )
    ]
    result.project_profile.code_symbols = [
        CodeSymbol(
            name='create_app',
            symbol_type='function',
            source_path='app.py',
            line_number=7,
        )
    ]
    result.project_profile.module_relations = [
        ModuleRelation(
            source_path='app.py',
            target='api.router',
            relation_type='import',
            line_number=2,
        )
    ]

    summaries = build_entrypoint_summaries(result)
    documents = build_knowledge_documents(result)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.entrypoint_kind == 'web_api'
    assert 'uvicorn app:app --reload' in summary.startup_commands
    assert 'pyproject.toml' in summary.dependent_configs
    assert 'ASGI app: app' in summary.exposed_interfaces
    assert 'Redis' in summary.service_dependencies
    assert summary.subproject_root == '.'
    assert 'service' in summary.subproject_markers
    assert 'function create_app @L7' in summary.code_symbols
    assert 'import api.router @L2' in summary.module_relations
    assert any(component in summary.related_components for component in ('FastAPI', 'api.router'))
    assert any(document.doc_type == 'entrypoint_summary' for document in documents)
    assert any(document.doc_type == 'subproject_summary' for document in documents)
    assert any(
        document.doc_type == 'entrypoint_summary'
        and '关键符号：function create_app @L7' in document.content
        for document in documents
    )
