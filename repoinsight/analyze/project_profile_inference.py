import json
import tomllib

from repoinsight.models.analysis_model import KeyFileContent, ProjectProfile, TechStackItem
from repoinsight.models.file_model import ScanResult
from repoinsight.models.repo_model import RepoInfo


# 关键字到结构化信号的轻量映射，先覆盖当前最常见的语言与框架。
CONTENT_SIGNAL_RULES: dict[str, tuple[str, str]] = {
    'fastapi': ('FastAPI', 'framework'),
    'flask': ('Flask', 'framework'),
    'django': ('Django', 'framework'),
    'langchain': ('LangChain', 'framework'),
    'langgraph': ('LangGraph', 'framework'),
    'llama_index': ('LlamaIndex', 'framework'),
    'llamaindex': ('LlamaIndex', 'framework'),
    'chromadb': ('Chroma', 'database'),
    'qdrant': ('Qdrant', 'database'),
    'faiss': ('FAISS', 'database'),
    'streamlit': ('Streamlit', 'framework'),
    'gradio': ('Gradio', 'framework'),
    'typer': ('Typer', 'framework'),
    'click': ('Click', 'framework'),
    'pydantic': ('Pydantic', 'tool'),
    'uvicorn': ('Uvicorn', 'runtime'),
    'sqlalchemy': ('SQLAlchemy', 'database'),
    'redis': ('Redis', 'database'),
    'celery': ('Celery', 'tool'),
    'react': ('React', 'framework'),
    'next': ('Next.js', 'framework'),
    'vue': ('Vue', 'framework'),
    'vite': ('Vite', 'build_tool'),
    'express': ('Express', 'framework'),
    'nestjs': ('NestJS', 'framework'),
    'typescript': ('TypeScript', 'language'),
    'jest': ('Jest', 'test_tool'),
    'vitest': ('Vitest', 'test_tool'),
    'pytest': ('Pytest', 'test_tool'),
    'docker': ('Docker', 'deploy_tool'),
}

# package.json 中常见依赖名到结构化信号的映射。
PACKAGE_JSON_DEPENDENCY_RULES: dict[str, tuple[str, str]] = {
    'react': ('React', 'framework'),
    'next': ('Next.js', 'framework'),
    'vue': ('Vue', 'framework'),
    'express': ('Express', 'framework'),
    '@nestjs/core': ('NestJS', 'framework'),
    'typescript': ('TypeScript', 'language'),
    'vite': ('Vite', 'build_tool'),
    'jest': ('Jest', 'test_tool'),
    'vitest': ('Vitest', 'test_tool'),
    'tsx': ('TSX', 'runtime'),
}


def infer_project_profile(
    repo_info: RepoInfo,
    scan_result: ScanResult,
    key_file_contents: list[KeyFileContent],
) -> ProjectProfile:
    """综合元数据、关键文件名和关键文件内容，生成结构化项目画像。"""
    inferred: dict[str, TechStackItem] = {}
    entrypoints: set[str] = set()
    project_markers: set[str] = set()

    _infer_from_repo_metadata(repo_info, inferred)
    _infer_from_file_paths(scan_result, inferred, entrypoints, project_markers)
    _infer_from_file_contents(key_file_contents, inferred, entrypoints, project_markers)

    signals = sorted(inferred.values(), key=lambda item: (item.category, item.name.lower()))
    return ProjectProfile(
        primary_language=_choose_primary_language(repo_info, signals),
        languages=_collect_names(signals, 'language'),
        runtimes=_collect_names(signals, 'runtime'),
        frameworks=_collect_names(signals, 'framework'),
        build_tools=_collect_names(signals, 'build_tool'),
        package_managers=_collect_names(signals, 'package_manager'),
        test_tools=_collect_names(signals, 'test_tool'),
        ci_cd_tools=_collect_names(signals, 'ci_cd'),
        deploy_tools=_collect_names(signals, 'deploy_tool'),
        entrypoints=sorted(entrypoints),
        project_markers=sorted(project_markers),
        signals=signals,
    )


def _infer_from_repo_metadata(repo_info: RepoInfo, inferred: dict[str, TechStackItem]) -> None:
    """根据 GitHub 元数据补充语言级别的识别结果。"""
    repo_model = repo_info.repo_model

    if repo_model.primary_language:
        _add_signal(
            inferred,
            name=repo_model.primary_language,
            category='language',
            evidence=f'GitHub 主要语言字段：{repo_model.primary_language}',
        )

    for language in repo_model.languages.keys():
        _add_signal(
            inferred,
            name=language,
            category='language',
            evidence=f'GitHub 语言分布包含：{language}',
        )


def _infer_from_file_paths(
    scan_result: ScanResult,
    inferred: dict[str, TechStackItem],
    entrypoints: set[str],
    project_markers: set[str],
) -> None:
    """根据关键文件路径和目录特征补充项目画像。"""
    for entry in scan_result.key_files:
        lower_name = entry.name.lower()
        lower_path = entry.path.lower()

        if lower_name in {'pyproject.toml', 'requirements.txt', 'setup.py', 'setup.cfg'}:
            _add_signal(inferred, 'Python', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'pip', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_name == 'poetry.lock':
            _add_signal(inferred, 'Python', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Poetry', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_name == 'package.json':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'npm', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_name == 'package-lock.json':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'npm', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_name == 'pnpm-lock.yaml':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'pnpm', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_name == 'yarn.lock':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Yarn', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_name == 'tsconfig.json':
            _add_signal(inferred, 'TypeScript', 'language', f'检测到关键文件：{entry.path}')

        if lower_name.startswith('vite.config.'):
            _add_signal(inferred, 'Vite', 'build_tool', f'检测到关键文件：{entry.path}')

        if lower_name.startswith('next.config.'):
            _add_signal(inferred, 'Next.js', 'framework', f'检测到关键文件：{entry.path}')

        if lower_name in {'dockerfile', 'docker-compose.yml', 'docker-compose.yaml'}:
            _add_signal(inferred, 'Docker', 'deploy_tool', f'检测到关键文件：{entry.path}')

        if lower_name in {'docker-compose.yml', 'docker-compose.yaml'}:
            _add_signal(inferred, 'Docker Compose', 'deploy_tool', f'检测到关键文件：{entry.path}')

        if lower_name in {'pom.xml'}:
            _add_signal(inferred, 'Java', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Maven', 'build_tool', f'检测到关键文件：{entry.path}')

        if lower_name in {'build.gradle', 'build.gradle.kts', 'settings.gradle'}:
            _add_signal(inferred, 'Java', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Gradle', 'build_tool', f'检测到关键文件：{entry.path}')

        if lower_name == 'go.mod':
            _add_signal(inferred, 'Go', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Go Modules', 'build_tool', f'检测到关键文件：{entry.path}')

        if lower_name == 'cargo.toml':
            _add_signal(inferred, 'Rust', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Cargo', 'build_tool', f'检测到关键文件：{entry.path}')

        if lower_name == 'composer.json':
            _add_signal(inferred, 'PHP', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Composer', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_name == 'gemfile':
            _add_signal(inferred, 'Ruby', 'language', f'检测到关键文件：{entry.path}')
            _add_signal(inferred, 'Bundler', 'package_manager', f'检测到关键文件：{entry.path}')

        if lower_path.startswith('.github/workflows/'):
            _add_signal(inferred, 'GitHub Actions', 'ci_cd', f'检测到工作流文件：{entry.path}')

        if lower_name in {'main.py', 'app.py', 'manage.py', 'cli.py', '__main__.py', 'main.go'}:
            entrypoints.add(entry.path)

        if lower_path in {'src/main.rs', 'src/lib.rs'}:
            entrypoints.add(entry.path)

    all_paths = {item.path.lower() for item in scan_result.all_files}
    if any(path.startswith('packages/') for path in all_paths):
        project_markers.add('monorepo')

    if any(path.startswith('cmd/') for path in all_paths):
        project_markers.add('go_cmd_layout')


def _infer_from_file_contents(
    key_file_contents: list[KeyFileContent],
    inferred: dict[str, TechStackItem],
    entrypoints: set[str],
    project_markers: set[str],
) -> None:
    """根据关键文件内容补充框架、工具和入口信息。"""
    for file_content in key_file_contents:
        lower_path = file_content.path.lower()
        lower_content = file_content.content.lower()

        for keyword, (display_name, category) in CONTENT_SIGNAL_RULES.items():
            if keyword not in lower_content:
                continue

            _add_signal(
                inferred,
                name=display_name,
                category=category,
                evidence=f'在 {file_content.path} 中命中关键字：{keyword}',
            )

        if any(keyword in lower_content for keyword in ('argparse', 'typer', 'click')):
            project_markers.add('cli')
            entrypoints.add(file_content.path)

        if 'if __name__ == "__main__"' in lower_content:
            entrypoints.add(file_content.path)

        if lower_path == 'package.json':
            _infer_from_package_json(file_content, inferred, project_markers)

        if lower_path == 'pyproject.toml':
            _infer_from_pyproject(file_content, inferred, project_markers)

        if lower_path == 'requirements.txt':
            _infer_from_requirements(file_content, inferred)


def _infer_from_package_json(
    file_content: KeyFileContent,
    inferred: dict[str, TechStackItem],
    project_markers: set[str],
) -> None:
    """解析 package.json，提取 Node.js / TypeScript 项目的基础画像。"""
    try:
        package_data = json.loads(file_content.content)
    except json.JSONDecodeError:
        return

    package_manager = package_data.get('packageManager')
    if isinstance(package_manager, str):
        manager_name = package_manager.split('@', maxsplit=1)[0].strip()
        if manager_name:
            _add_signal(
                inferred,
                name=manager_name,
                category='package_manager',
                evidence=f'在 {file_content.path} 的 packageManager 字段中识别到：{package_manager}',
            )

    scripts = package_data.get('scripts')
    if isinstance(scripts, dict):
        if 'test' in scripts:
            project_markers.add('has_test_script')
        if 'build' in scripts:
            project_markers.add('has_build_script')
        if 'dev' in scripts:
            project_markers.add('has_dev_script')

    for section in ('dependencies', 'devDependencies', 'peerDependencies'):
        dependencies = package_data.get(section)
        if not isinstance(dependencies, dict):
            continue

        for package_name in dependencies.keys():
            rule = PACKAGE_JSON_DEPENDENCY_RULES.get(package_name.lower())
            if not rule:
                continue

            display_name, category = rule
            _add_signal(
                inferred,
                name=display_name,
                category=category,
                evidence=f'在 {file_content.path} 的 {section} 中识别到依赖：{package_name}',
            )


def _infer_from_pyproject(
    file_content: KeyFileContent,
    inferred: dict[str, TechStackItem],
    project_markers: set[str],
) -> None:
    """解析 pyproject.toml，提取 Python 项目的基础画像。"""
    try:
        pyproject = tomllib.loads(file_content.content)
    except tomllib.TOMLDecodeError:
        return

    project_section = pyproject.get('project')
    if isinstance(project_section, dict):
        if project_section.get('scripts'):
            project_markers.add('python_scripts')

        dependencies = project_section.get('dependencies')
        if isinstance(dependencies, list):
            for dependency in dependencies:
                if isinstance(dependency, str):
                    _infer_from_dependency_text(
                        dependency_text=dependency,
                        source_path=file_content.path,
                        inferred=inferred,
                    )

    tool_section = pyproject.get('tool')
    if isinstance(tool_section, dict):
        if 'poetry' in tool_section:
            _add_signal(inferred, 'Poetry', 'package_manager', f'在 {file_content.path} 中检测到 tool.poetry')


def _infer_from_requirements(
    file_content: KeyFileContent,
    inferred: dict[str, TechStackItem],
) -> None:
    """从 requirements.txt 中提取基础依赖信号。"""
    for line in file_content.content.splitlines():
        dependency_text = line.strip()
        if not dependency_text or dependency_text.startswith('#'):
            continue

        _infer_from_dependency_text(
            dependency_text=dependency_text,
            source_path=file_content.path,
            inferred=inferred,
        )


def _infer_from_dependency_text(
    dependency_text: str,
    source_path: str,
    inferred: dict[str, TechStackItem],
) -> None:
    """根据依赖声明文本补充常见 Python 框架与工具。"""
    normalized = dependency_text.lower()
    for keyword, (display_name, category) in CONTENT_SIGNAL_RULES.items():
        if keyword not in normalized:
            continue

        _add_signal(
            inferred,
            name=display_name,
            category=category,
            evidence=f'在 {source_path} 的依赖声明中命中关键字：{keyword}',
        )


def _collect_names(signals: list[TechStackItem], category: str) -> list[str]:
    """从信号列表中提取某一类别的名称，并保持稳定排序。"""
    return [item.name for item in signals if item.category == category]


def _choose_primary_language(repo_info: RepoInfo, signals: list[TechStackItem]) -> str | None:
    """优先使用 GitHub 元数据，否则退回到规则识别出的语言结果。"""
    if repo_info.repo_model.primary_language:
        return repo_info.repo_model.primary_language

    for item in signals:
        if item.category == 'language':
            return item.name

    return None


def _add_signal(
    inferred: dict[str, TechStackItem],
    name: str,
    category: str,
    evidence: str,
) -> None:
    """向结果集中加入一条识别信号，已存在时保留首次证据。"""
    key = f'{category}:{name.lower()}'
    if key in inferred:
        return

    inferred[key] = TechStackItem(
        name=name,
        category=category,
        evidence=evidence,
    )
