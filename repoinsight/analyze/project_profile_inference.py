import json
import re
import tomllib

from repoinsight.models.analysis_model import (
    CodeSymbol,
    KeyFileContent,
    ModuleRelation,
    ProjectProfile,
    SubprojectSummary,
    TechStackItem,
)
from repoinsight.models.file_model import ScanResult
from repoinsight.models.repo_model import RepoInfo


# 常见依赖名 / import 目标到结构化信号的映射。
SIGNAL_RULES: dict[str, tuple[str, str]] = {
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
    'openai': ('OpenAI', 'tool'),
    'react': ('React', 'framework'),
    'react-dom': ('React', 'framework'),
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

# 不同证据来源的强弱分级。
EVIDENCE_PRIORITY = {
    'strong': 3,
    'medium': 2,
    'weak': 1,
}

# 这些来源能直接确认技术实际被使用，优先作为最终技术栈。
CONFIRMED_EVIDENCE_LEVELS = {'strong', 'medium'}

# 弱证据候选只用于线索提示，不直接进入最终技术栈。
WEAK_KEYWORD_RULES: dict[str, tuple[str, str]] = {
    'fastapi': ('FastAPI', 'framework'),
    'flask': ('Flask', 'framework'),
    'django': ('Django', 'framework'),
    'langchain': ('LangChain', 'framework'),
    'langgraph': ('LangGraph', 'framework'),
    'llamaindex': ('LlamaIndex', 'framework'),
    'llama_index': ('LlamaIndex', 'framework'),
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
    'vue': ('Vue', 'framework'),
    'express': ('Express', 'framework'),
    'pytest': ('Pytest', 'test_tool'),
    'vitest': ('Vitest', 'test_tool'),
    'jest': ('Jest', 'test_tool'),
    'docker': ('Docker', 'deploy_tool'),
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
    subprojects = _infer_subprojects(scan_result, key_file_contents, sorted(entrypoints))
    code_symbols, module_relations = _extract_code_structure(key_file_contents)
    _infer_from_module_relations(module_relations, inferred)

    signals = sorted(inferred.values(), key=_signal_sort_key)
    confirmed_signals = [item for item in signals if item.evidence_level in CONFIRMED_EVIDENCE_LEVELS]
    weak_signals = [item for item in signals if item.evidence_level == 'weak']
    return ProjectProfile(
        primary_language=_choose_primary_language(repo_info, confirmed_signals, signals),
        languages=_collect_names(confirmed_signals, 'language'),
        runtimes=_collect_names(confirmed_signals, 'runtime'),
        frameworks=_collect_names(confirmed_signals, 'framework'),
        build_tools=_collect_names(confirmed_signals, 'build_tool'),
        package_managers=_collect_names(confirmed_signals, 'package_manager'),
        test_tools=_collect_names(confirmed_signals, 'test_tool'),
        ci_cd_tools=_collect_names(confirmed_signals, 'ci_cd'),
        deploy_tools=_collect_names(confirmed_signals, 'deploy_tool'),
        entrypoints=sorted(entrypoints),
        project_markers=sorted(project_markers),
        subprojects=subprojects,
        code_symbols=code_symbols,
        module_relations=module_relations,
        confirmed_signals=confirmed_signals,
        weak_signals=weak_signals,
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
            evidence_level='strong',
            evidence_source='metadata',
        )

    for language in repo_model.languages.keys():
        _add_signal(
            inferred,
            name=language,
            category='language',
            evidence=f'GitHub 语言分布包含：{language}',
            evidence_level='strong',
            evidence_source='metadata',
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
            _add_signal(
                inferred,
                'Python',
                'language',
                f'检测到关键文件：{entry.path}',
                evidence_level='strong',
                evidence_source='config',
                source_path=entry.path,
            )
            _add_signal(
                inferred,
                'pip',
                'package_manager',
                f'检测到关键文件：{entry.path}',
                evidence_level='medium',
                evidence_source='config',
                source_path=entry.path,
            )

        if lower_name == 'poetry.lock':
            _add_signal(inferred, 'Python', 'language', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)
            _add_signal(inferred, 'Poetry', 'package_manager', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)

        if lower_name == 'package.json':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)
            _add_signal(inferred, 'npm', 'package_manager', f'检测到关键文件：{entry.path}', 'medium', 'config', entry.path)

        if lower_name == 'package-lock.json':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)
            _add_signal(inferred, 'npm', 'package_manager', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)

        if lower_name == 'pnpm-lock.yaml':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)
            _add_signal(inferred, 'pnpm', 'package_manager', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)

        if lower_name == 'yarn.lock':
            _add_signal(inferred, 'Node.js', 'runtime', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)
            _add_signal(inferred, 'Yarn', 'package_manager', f'检测到关键文件：{entry.path}', 'strong', 'lockfile', entry.path)

        if lower_name == 'tsconfig.json':
            _add_signal(inferred, 'TypeScript', 'language', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name.startswith('vite.config.'):
            _add_signal(inferred, 'Vite', 'build_tool', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name.startswith('next.config.'):
            _add_signal(inferred, 'Next.js', 'framework', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name in {'dockerfile', 'docker-compose.yml', 'docker-compose.yaml'}:
            _add_signal(inferred, 'Docker', 'deploy_tool', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name in {'docker-compose.yml', 'docker-compose.yaml'}:
            _add_signal(inferred, 'Docker Compose', 'deploy_tool', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name in {'pom.xml'}:
            _add_signal(inferred, 'Java', 'language', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)
            _add_signal(inferred, 'Maven', 'build_tool', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name in {'build.gradle', 'build.gradle.kts', 'settings.gradle'}:
            _add_signal(inferred, 'Java', 'language', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)
            _add_signal(inferred, 'Gradle', 'build_tool', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name == 'go.mod':
            _add_signal(inferred, 'Go', 'language', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)
            _add_signal(inferred, 'Go Modules', 'build_tool', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name == 'cargo.toml':
            _add_signal(inferred, 'Rust', 'language', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)
            _add_signal(inferred, 'Cargo', 'build_tool', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name == 'composer.json':
            _add_signal(inferred, 'PHP', 'language', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)
            _add_signal(inferred, 'Composer', 'package_manager', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_name == 'gemfile':
            _add_signal(inferred, 'Ruby', 'language', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)
            _add_signal(inferred, 'Bundler', 'package_manager', f'检测到关键文件：{entry.path}', 'strong', 'config', entry.path)

        if lower_path.startswith('.github/workflows/'):
            _add_signal(inferred, 'GitHub Actions', 'ci_cd', f'检测到工作流文件：{entry.path}', 'strong', 'config', entry.path)

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

        if _contains_explicit_cli_signal(lower_content):
            project_markers.add('cli')
            entrypoints.add(file_content.path)
            if _contains_word(lower_content, 'argparse'):
                _add_signal(
                    inferred,
                    name='argparse',
                    category='tool',
                    evidence=f'在 {file_content.path} 中检测到 argparse 参数解析逻辑',
                    evidence_level='medium',
                    evidence_source='runtime_call',
                    source_path=file_content.path,
                )

        _infer_runtime_construction_signals(file_content, inferred)
        _infer_weak_keyword_candidates(file_content, inferred)

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
                evidence_level='strong',
                evidence_source='config',
                source_path=file_content.path,
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
            resolved_signal = _resolve_signal_from_reference(package_name)
            if resolved_signal is None:
                continue

            display_name, category = resolved_signal
            _add_signal(
                inferred,
                name=display_name,
                category=category,
                evidence=f'在 {file_content.path} 的 {section} 中识别到依赖：{package_name}',
                evidence_level='strong',
                evidence_source='dependency',
                source_path=file_content.path,
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
            _add_signal(
                inferred,
                'Poetry',
                'package_manager',
                f'在 {file_content.path} 中检测到 tool.poetry',
                'strong',
                'config',
                file_content.path,
            )


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
    normalized = _normalize_reference_text(dependency_text)
    resolved_signal = _resolve_signal_from_reference(normalized)
    if resolved_signal is None:
        return

    display_name, category = resolved_signal
    _add_signal(
        inferred,
        name=display_name,
        category=category,
        evidence=f'在 {source_path} 的依赖声明中识别到依赖：{dependency_text}',
        evidence_level='strong',
        evidence_source='dependency',
        source_path=source_path,
    )


def _collect_names(signals: list[TechStackItem], category: str) -> list[str]:
    """从信号列表中提取某一类别的名称，并保持稳定排序。"""
    return [item.name for item in signals if item.category == category]


def _choose_primary_language(
    repo_info: RepoInfo,
    confirmed_signals: list[TechStackItem],
    all_signals: list[TechStackItem],
) -> str | None:
    """优先使用 GitHub 元数据，否则退回到规则识别出的语言结果。"""
    if repo_info.repo_model.primary_language:
        return repo_info.repo_model.primary_language

    for item in confirmed_signals:
        if item.category == 'language':
            return item.name

    for item in all_signals:
        if item.category == 'language':
            return item.name

    return None


def _add_signal(
    inferred: dict[str, TechStackItem],
    name: str,
    category: str,
    evidence: str,
    evidence_level: str = 'medium',
    evidence_source: str = 'unknown',
    source_path: str | None = None,
) -> None:
    """向结果集中加入一条识别信号，已存在时优先保留更强证据。"""
    key = f'{category}:{name.lower()}'
    existing = inferred.get(key)
    candidate = TechStackItem(
        name=name,
        category=category,
        evidence=evidence,
        evidence_level=evidence_level,
        evidence_source=evidence_source,
        source_path=source_path,
    )
    if existing is not None and _should_keep_existing_signal(existing, candidate):
        return

    inferred[key] = candidate


def _should_keep_existing_signal(existing: TechStackItem, candidate: TechStackItem) -> bool:
    """比较同一技术的两条证据，优先保留强证据。"""
    existing_priority = EVIDENCE_PRIORITY.get(existing.evidence_level, 0)
    candidate_priority = EVIDENCE_PRIORITY.get(candidate.evidence_level, 0)
    if existing_priority != candidate_priority:
        return existing_priority > candidate_priority

    # 同强度下，优先保留更直接的依赖 / import / 配置类证据。
    source_priority = {
        'dependency': 4,
        'import': 4,
        'config': 3,
        'lockfile': 3,
        'metadata': 3,
        'runtime_call': 2,
        'keyword': 1,
        'unknown': 0,
    }
    return source_priority.get(existing.evidence_source, 0) >= source_priority.get(candidate.evidence_source, 0)


def _signal_sort_key(item: TechStackItem) -> tuple[int, str, str]:
    """让更强证据在排序时靠前。"""
    return (-EVIDENCE_PRIORITY.get(item.evidence_level, 0), item.category, item.name.lower())


def _contains_word(text: str, keyword: str) -> bool:
    """判断文本中是否出现完整单词，避免 ReactAgent 这类误判。"""
    return re.search(rf'(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])', text) is not None


def _contains_explicit_cli_signal(lower_content: str) -> bool:
    """检测明显的 CLI 参数解析或命令注册逻辑。"""
    return any(
        pattern.search(lower_content)
        for pattern in (
            re.compile(r'(?<![a-z0-9_])argparse(?![a-z0-9_])'),
            re.compile(r'(?<![a-z0-9_])typer(?![a-z0-9_])'),
            re.compile(r'(?<![a-z0-9_])click(?![a-z0-9_])'),
            re.compile(r'commander\.'),
            re.compile(r'(?<![a-z0-9_])yargs(?![a-z0-9_])'),
        )
    )


def _infer_runtime_construction_signals(
    file_content: KeyFileContent,
    inferred: dict[str, TechStackItem],
) -> None:
    """从显式构造 / 调用代码中提取中强度技术信号。"""
    content = file_content.content
    lower_content = content.lower()

    construction_rules = (
        (re.compile(r'\bFastAPI\s*\('), 'FastAPI', 'framework'),
        (re.compile(r'\bFlask\s*\('), 'Flask', 'framework'),
        (re.compile(r'\btyper\.Typer\s*\('), 'Typer', 'framework'),
        (re.compile(r'@click\.(?:command|group)\s*\('), 'Click', 'framework'),
        (re.compile(r'\buvicorn\.run\s*\('), 'Uvicorn', 'runtime'),
        (re.compile(r'\bexpress\s*\('), 'Express', 'framework'),
        (re.compile(r'\bcreateRoot\s*\('), 'React', 'framework'),
        (re.compile(r'\bcreateApp\s*\('), 'Vue', 'framework'),
    )
    for pattern, display_name, category in construction_rules:
        if pattern.search(content) is None:
            continue
        _add_signal(
            inferred,
            name=display_name,
            category=category,
            evidence=f'在 {file_content.path} 中检测到显式构造或运行调用：{pattern.pattern}',
            evidence_level='medium',
            evidence_source='runtime_call',
            source_path=file_content.path,
        )

    if _contains_word(lower_content, 'pytest'):
        _add_signal(
            inferred,
            name='Pytest',
            category='test_tool',
            evidence=f'在 {file_content.path} 中检测到 pytest 相关调用',
            evidence_level='medium',
            evidence_source='runtime_call',
            source_path=file_content.path,
        )


def _infer_weak_keyword_candidates(
    file_content: KeyFileContent,
    inferred: dict[str, TechStackItem],
) -> None:
    """记录仅来自普通文本命中的候选技术，供人工复核。"""
    lower_content = file_content.content.lower()
    for keyword, (display_name, category) in WEAK_KEYWORD_RULES.items():
        if not _contains_word(lower_content, keyword):
            continue
        _add_signal(
            inferred,
            name=display_name,
            category=category,
            evidence=f'在 {file_content.path} 中出现关键字：{keyword}',
            evidence_level='weak',
            evidence_source='keyword',
            source_path=file_content.path,
        )


def _infer_from_module_relations(
    module_relations: list[ModuleRelation],
    inferred: dict[str, TechStackItem],
) -> None:
    """根据 import / require / use 结果补充更可信的技术栈信号。"""
    for relation in module_relations:
        resolved_signal = _resolve_signal_from_reference(relation.target)
        if resolved_signal is None:
            continue

        display_name, category = resolved_signal
        _add_signal(
            inferred,
            name=display_name,
            category=category,
            evidence=(
                f'在 {relation.source_path} 中检测到 {relation.relation_type} 依赖：'
                f'{relation.target}{_format_line_hint(relation.line_number)}'
            ),
            evidence_level='strong',
            evidence_source='import',
            source_path=relation.source_path,
        )


def _resolve_signal_from_reference(reference: str) -> tuple[str, str] | None:
    """把依赖名或 import 目标解析成结构化技术信号。"""
    normalized = _normalize_reference_text(reference)
    if not normalized:
        return None

    if normalized in SIGNAL_RULES:
        return SIGNAL_RULES[normalized]

    for key, signal in SIGNAL_RULES.items():
        if (
            normalized.startswith(f'{key}.')
            or normalized.startswith(f'{key}/')
            or normalized.startswith(f'{key}-')
            or normalized.startswith(f'{key}_')
        ):
            return signal

    if normalized.startswith('@nestjs/'):
        return 'NestJS', 'framework'
    if normalized.startswith('react-dom'):
        return 'React', 'framework'
    if normalized.startswith('next/'):
        return 'Next.js', 'framework'
    if normalized.startswith('@vue/'):
        return 'Vue', 'framework'

    return None


def _normalize_reference_text(reference: str) -> str:
    """把依赖 / import 文本规范化，便于做准确匹配。"""
    normalized = reference.strip().lower()
    normalized = normalized.split('[', maxsplit=1)[0]
    normalized = re.split(r'[<>=~! ]', normalized, maxsplit=1)[0]
    return normalized.strip()


def _format_line_hint(line_number: int | None) -> str:
    """把可选行号格式化为简短提示。"""
    if line_number is None:
        return ''
    return f' @L{line_number}'


def _infer_subprojects(
    scan_result: ScanResult,
    key_file_contents: list[KeyFileContent],
    entrypoints: list[str],
) -> list[SubprojectSummary]:
    """根据关键配置文件和目录布局提取子项目信息。"""
    config_entries = {
        'package.json': ('nodejs', 'workspace'),
        'pyproject.toml': ('python', 'package'),
        'requirements.txt': ('python', 'package'),
        'go.mod': ('go', 'service'),
        'cargo.toml': ('rust', 'package'),
        'pom.xml': ('java', 'service'),
        'build.gradle': ('java', 'service'),
        'build.gradle.kts': ('java', 'service'),
        'composer.json': ('php', 'package'),
        'gemfile': ('ruby', 'package'),
    }
    subprojects_by_root: dict[str, SubprojectSummary] = {}

    for entry in scan_result.key_files:
        file_name = entry.name.lower()
        config_info = config_entries.get(file_name)
        if config_info is None:
            continue

        language_scope, project_kind = config_info
        root_path = entry.parent_dir or '.'
        markers = _infer_subproject_markers(root_path)
        summary = subprojects_by_root.get(root_path)
        if summary is None:
            summary = SubprojectSummary(
                root_path=root_path,
                language_scope=language_scope,
                project_kind=project_kind,
                config_paths=[],
                entrypoint_paths=[],
                markers=markers,
            )
            subprojects_by_root[root_path] = summary

        if entry.path not in summary.config_paths:
            summary.config_paths.append(entry.path)

    for entrypoint in entrypoints:
        root_path = _guess_subproject_root(entrypoint, subprojects_by_root.keys())
        if root_path is None:
            continue

        summary = subprojects_by_root[root_path]
        if entrypoint not in summary.entrypoint_paths:
            summary.entrypoint_paths.append(entrypoint)

    for file_content in key_file_contents:
        if file_content.path.lower().endswith('package.json'):
            root_path = file_content.path.rsplit('/', maxsplit=1)[0] if '/' in file_content.path else '.'
            summary = subprojects_by_root.get(root_path)
            if summary is None:
                continue

            try:
                package_data = json.loads(file_content.content)
            except json.JSONDecodeError:
                continue

            workspaces = package_data.get('workspaces')
            if isinstance(workspaces, list) and workspaces:
                if 'monorepo_workspace' not in summary.markers:
                    summary.markers.append('monorepo_workspace')

    return sorted(subprojects_by_root.values(), key=lambda item: (item.root_path, item.language_scope))


def _extract_code_structure(
    key_file_contents: list[KeyFileContent],
) -> tuple[list[CodeSymbol], list[ModuleRelation]]:
    """从关键文件中抽取符号定义与模块关系。"""
    code_symbols: list[CodeSymbol] = []
    module_relations: list[ModuleRelation] = []

    for file_content in key_file_contents:
        lower_path = file_content.path.lower()
        if lower_path.endswith('.py'):
            _extract_python_structure(file_content, code_symbols, module_relations)
        elif lower_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
            _extract_javascript_structure(file_content, code_symbols, module_relations)
        elif lower_path.endswith('.go'):
            _extract_go_structure(file_content, code_symbols, module_relations)
        elif lower_path.endswith('.rs'):
            _extract_rust_structure(file_content, code_symbols, module_relations)

    return code_symbols, module_relations


def _extract_python_structure(
    file_content: KeyFileContent,
    code_symbols: list[CodeSymbol],
    module_relations: list[ModuleRelation],
) -> None:
    """抽取 Python 关键文件中的函数、类和导入。"""
    for index, raw_line in enumerate(file_content.content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        function_match = re.match(r'def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if function_match:
            code_symbols.append(
                CodeSymbol(
                    name=function_match.group(1),
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=index,
                )
            )

        class_match = re.match(r'class\s+([A-Za-z_][A-Za-z0-9_]*)', line)
        if class_match:
            code_symbols.append(
                CodeSymbol(
                    name=class_match.group(1),
                    symbol_type='class',
                    source_path=file_content.path,
                    line_number=index,
                )
            )

        import_match = re.match(r'import\s+([A-Za-z0-9_\.]+)', line)
        if import_match:
            module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=import_match.group(1),
                    relation_type='import',
                    line_number=index,
                )
            )

        from_match = re.match(r'from\s+([A-Za-z0-9_\.]+)\s+import', line)
        if from_match:
            module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=from_match.group(1),
                    relation_type='import',
                    line_number=index,
                )
            )


def _extract_javascript_structure(
    file_content: KeyFileContent,
    code_symbols: list[CodeSymbol],
    module_relations: list[ModuleRelation],
) -> None:
    """抽取 JS/TS 关键文件中的函数、类和导入。"""
    for index, raw_line in enumerate(file_content.content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        function_match = re.match(r'(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if function_match:
            code_symbols.append(
                CodeSymbol(
                    name=function_match.group(1),
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=index,
                )
            )

        class_match = re.match(r'(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)', line)
        if class_match:
            code_symbols.append(
                CodeSymbol(
                    name=class_match.group(1),
                    symbol_type='class',
                    source_path=file_content.path,
                    line_number=index,
                )
            )

        const_match = re.match(r'(?:export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=', line)
        if const_match:
            code_symbols.append(
                CodeSymbol(
                    name=const_match.group(1),
                    symbol_type='variable',
                    source_path=file_content.path,
                    line_number=index,
                )
            )

        import_match = re.search(r'from\s+[\'"]([^\'"]+)[\'"]', line)
        if import_match:
            module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=import_match.group(1),
                    relation_type='import',
                    line_number=index,
                )
            )

        require_match = re.search(r'require\(\s*[\'"]([^\'"]+)[\'"]\s*\)', line)
        if require_match:
            module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=require_match.group(1),
                    relation_type='require',
                    line_number=index,
                )
            )


def _extract_go_structure(
    file_content: KeyFileContent,
    code_symbols: list[CodeSymbol],
    module_relations: list[ModuleRelation],
) -> None:
    """抽取 Go 关键文件中的函数与依赖。"""
    for index, raw_line in enumerate(file_content.content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        function_match = re.match(r'func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if function_match:
            code_symbols.append(
                CodeSymbol(
                    name=function_match.group(1),
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=index,
                )
            )

        import_match = re.match(r'"([^"]+)"', line)
        if import_match:
            module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=import_match.group(1),
                    relation_type='import',
                    line_number=index,
                )
            )


def _extract_rust_structure(
    file_content: KeyFileContent,
    code_symbols: list[CodeSymbol],
    module_relations: list[ModuleRelation],
) -> None:
    """抽取 Rust 关键文件中的函数与依赖。"""
    for index, raw_line in enumerate(file_content.content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        function_match = re.match(r'fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if function_match:
            code_symbols.append(
                CodeSymbol(
                    name=function_match.group(1),
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=index,
                )
            )

        use_match = re.match(r'use\s+([^;]+);', line)
        if use_match:
            module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=use_match.group(1),
                    relation_type='use',
                    line_number=index,
                )
            )


def _infer_subproject_markers(root_path: str) -> list[str]:
    """根据路径模式推断子项目标签。"""
    markers: list[str] = []
    normalized = root_path.lower()
    if normalized.startswith('packages/'):
        markers.append('package')
    if normalized.startswith('apps/'):
        markers.append('app')
    if normalized.startswith('services/'):
        markers.append('service')
    if normalized.startswith('cmd/'):
        markers.append('command')
    return markers


def _guess_subproject_root(entrypoint: str, roots: object) -> str | None:
    """根据入口路径找到最可能所属的子项目根。"""
    root_candidates = sorted((root for root in roots if root != '.'), key=len, reverse=True)
    for root in root_candidates:
        if entrypoint == root or entrypoint.startswith(f'{root}/'):
            return root
    return '.' if '.' in roots else None
