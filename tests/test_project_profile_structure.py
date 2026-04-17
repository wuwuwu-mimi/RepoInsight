from repoinsight.analyze.project_profile_inference import infer_project_profile
from repoinsight.models.analysis_model import KeyFileContent
from repoinsight.models.file_model import FileEntry, ScanResult, ScanStats
from repoinsight.models.repo_model import RepoInfo, RepoModel


def _build_file_entry(path: str) -> FileEntry:
    """快速构造测试用文件条目。"""
    name = path.split('/')[-1]
    parent_dir = '/'.join(path.split('/')[:-1])
    extension = f'.{name.split(".")[-1]}' if '.' in name else None
    return FileEntry(
        path=path,
        name=name,
        extension=extension,
        size_bytes=256,
        parent_dir=parent_dir,
        is_key_file=True,
    )


def test_infer_project_profile_extracts_subprojects_and_code_structure() -> None:
    repo_info = RepoInfo(
        repo_model=RepoModel(
            owner='demo',
            name='monorepo',
            full_name='demo/monorepo',
            html_url='https://github.com/demo/monorepo',
            default_branch='main',
            primary_language='TypeScript',
            languages={'TypeScript': 1200, 'Python': 900},
        ),
        readme='demo monorepo',
    )
    scan_result = ScanResult(
        root_path='E:/PythonProject/RepoInsight/clones/demo__monorepo',
        all_files=[
            _build_file_entry('package.json'),
            _build_file_entry('packages/web/package.json'),
            _build_file_entry('packages/web/src/main.tsx'),
            _build_file_entry('services/api/pyproject.toml'),
            _build_file_entry('services/api/app.py'),
        ],
        key_files=[
            _build_file_entry('package.json'),
            _build_file_entry('packages/web/package.json'),
            _build_file_entry('packages/web/src/main.tsx'),
            _build_file_entry('services/api/pyproject.toml'),
            _build_file_entry('services/api/app.py'),
        ],
        tree_preview=['packages/web', 'services/api'],
        stats=ScanStats(total_seen=5, kept_count=5, key_file_count=5),
    )
    key_file_contents = [
        KeyFileContent(
            path='package.json',
            size_bytes=180,
            content='{"name":"demo-root","workspaces":["packages/*","services/*"]}',
        ),
        KeyFileContent(
            path='packages/web/package.json',
            size_bytes=320,
            content=(
                '{'
                '"name":"web-app",'
                '"scripts":{"dev":"vite","build":"vite build"},'
                '"dependencies":{"react":"^18.0.0"}'
                '}'
            ),
        ),
        KeyFileContent(
            path='packages/web/src/main.tsx',
            size_bytes=300,
            content=(
                'import React from "react"\n'
                'import { createRoot } from "react-dom/client"\n\n'
                'export function bootstrap() {\n'
                '  return createRoot(document.getElementById("root")!)\n'
                '}\n'
            ),
        ),
        KeyFileContent(
            path='services/api/pyproject.toml',
            size_bytes=220,
            content=(
                '[project]\n'
                'name = "demo-api"\n'
                'dependencies = ["fastapi", "uvicorn"]\n'
            ),
        ),
        KeyFileContent(
            path='services/api/app.py',
            size_bytes=420,
            content=(
                'from fastapi import FastAPI\n'
                'from api.router import router\n\n'
                'class Settings:\n'
                '    pass\n\n'
                'def create_app():\n'
                '    return FastAPI()\n\n'
                'app = create_app()\n'
                'app.include_router(router)\n'
            ),
        ),
    ]

    profile = infer_project_profile(repo_info, scan_result, key_file_contents)

    subproject_roots = {item.root_path for item in profile.subprojects}
    assert 'packages/web' in subproject_roots
    assert 'services/api' in subproject_roots

    web_subproject = next(item for item in profile.subprojects if item.root_path == 'packages/web')
    api_subproject = next(item for item in profile.subprojects if item.root_path == 'services/api')
    assert web_subproject.language_scope == 'nodejs'
    assert 'packages/web/package.json' in web_subproject.config_paths
    assert api_subproject.language_scope == 'python'
    assert 'services/api/app.py' in api_subproject.entrypoint_paths
    assert 'service' in api_subproject.markers

    symbol_names = {(item.source_path, item.name, item.symbol_type) for item in profile.code_symbols}
    assert ('services/api/app.py', 'Settings', 'class') in symbol_names
    assert ('services/api/app.py', 'create_app', 'function') in symbol_names
    assert ('packages/web/src/main.tsx', 'bootstrap', 'function') in symbol_names

    relation_targets = {(item.source_path, item.target, item.relation_type) for item in profile.module_relations}
    assert ('services/api/app.py', 'fastapi', 'import') in relation_targets
    assert ('services/api/app.py', 'api.router', 'import') in relation_targets
    assert ('packages/web/src/main.tsx', 'react-dom/client', 'import') in relation_targets
    assert any(item.entity_kind == 'function' and item.qualified_name == 'create_app' for item in profile.code_entities)
    assert any(item.entity_kind == 'class' and item.qualified_name == 'Settings' for item in profile.code_entities)
    assert any(
        item.source_ref == 'services/api/app.py' and item.target_ref == 'api.router' and item.relation_type == 'import'
        for item in profile.code_relation_edges
    )
    assert any(
        item.source_ref == 'services/api/app.py' and item.target_ref == 'Settings' and item.relation_type == 'contain_symbol'
        for item in profile.code_relation_edges
    )
    assert any(
        item.source_ref == 'services/api/app.py' and item.target_ref == 'create_app' and item.relation_type == 'contain_symbol'
        for item in profile.code_relation_edges
    )
    assert any(
        item.source_ref == 'create_app' and item.target_ref == 'FastAPI' and item.relation_type == 'call'
        for item in profile.code_relation_edges
    )
    assert any(item.name == 'FastAPI' and item.evidence_level == 'strong' for item in profile.confirmed_signals)


def test_infer_project_profile_does_not_promote_plain_name_mentions_to_confirmed_stack() -> None:
    repo_info = RepoInfo(
        repo_model=RepoModel(
            owner='demo',
            name='agents',
            full_name='demo/agents',
            html_url='https://github.com/demo/agents',
            default_branch='main',
            primary_language='Python',
            languages={'Python': 800},
        ),
        readme='demo agents',
    )
    scan_result = ScanResult(
        root_path='E:/PythonProject/RepoInsight/clones/demo__agents',
        all_files=[_build_file_entry('main.py')],
        key_files=[_build_file_entry('main.py')],
        tree_preview=['main.py'],
        stats=ScanStats(total_seen=1, kept_count=1, key_file_count=1),
    )
    key_file_contents = [
        KeyFileContent(
            path='main.py',
            size_bytes=180,
            content=(
                'class ReactAgent:\n'
                '    pass\n\n'
                'def main():\n'
                '    agent = ReactAgent()\n'
                '    return agent\n'
            ),
        )
    ]

    profile = infer_project_profile(repo_info, scan_result, key_file_contents)

    assert 'React' not in profile.frameworks
    assert all(item.name != 'React' for item in profile.confirmed_signals)
    assert all(item.name != 'React' for item in profile.weak_signals)
