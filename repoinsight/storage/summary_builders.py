import json
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from repoinsight.models.analysis_model import (
    AnalysisRunResult,
    CodeSymbol,
    KeyFileContent,
    ModuleRelation,
    SubprojectSummary,
)
from repoinsight.models.rag_model import ConfigSummary, EntrypointSummary


# 不同语言常见配置文件，用于给通用摘要逻辑提供基础分类。
CONFIG_KIND_BY_NAME = {
    'pyproject.toml': ('package_manager', 'python'),
    'requirements.txt': ('package_manager', 'python'),
    'package.json': ('package_manager', 'nodejs'),
    'pnpm-lock.yaml': ('package_manager', 'nodejs'),
    'yarn.lock': ('package_manager', 'nodejs'),
    'package-lock.json': ('package_manager', 'nodejs'),
    'tsconfig.json': ('build', 'nodejs'),
    'vite.config.ts': ('build', 'nodejs'),
    'vite.config.js': ('build', 'nodejs'),
    'vite.config.mjs': ('build', 'nodejs'),
    'next.config.js': ('build', 'nodejs'),
    'next.config.mjs': ('build', 'nodejs'),
    'next.config.ts': ('build', 'nodejs'),
    'pom.xml': ('build', 'java'),
    'build.gradle': ('build', 'java'),
    'build.gradle.kts': ('build', 'java'),
    'settings.gradle': ('build', 'java'),
    'go.mod': ('package_manager', 'go'),
    'cargo.toml': ('package_manager', 'rust'),
    'composer.json': ('package_manager', 'php'),
    'gemfile': ('package_manager', 'ruby'),
    'dockerfile': ('deploy', 'generic'),
    'docker-compose.yml': ('deploy', 'generic'),
    'docker-compose.yaml': ('deploy', 'generic'),
}

ENTRYPOINT_KIND_BY_NAME = {
    'main.py': ('cli', 'python'),
    'app.py': ('web_api', 'python'),
    'manage.py': ('web_api', 'python'),
    'cli.py': ('cli', 'python'),
    '__main__.py': ('cli', 'python'),
    'main.go': ('web_api', 'go'),
    'main.rs': ('cli', 'rust'),
    'lib.rs': ('library', 'rust'),
    'program.cs': ('web_api', 'csharp'),
    'server.ts': ('web_api', 'nodejs'),
    'server.js': ('web_api', 'nodejs'),
    'main.ts': ('frontend', 'nodejs'),
    'main.tsx': ('frontend', 'nodejs'),
    'main.jsx': ('frontend', 'nodejs'),
    'workflow.py': ('workflow', 'python'),
    'graph.py': ('workflow', 'python'),
}

SERVICE_KEYWORDS = {
    'postgres': 'PostgreSQL',
    'postgresql': 'PostgreSQL',
    'psycopg': 'PostgreSQL',
    'mysql': 'MySQL',
    'mariadb': 'MariaDB',
    'sqlite': 'SQLite',
    'redis': 'Redis',
    'mongo': 'MongoDB',
    'mongodb': 'MongoDB',
    'mongoose': 'MongoDB',
    'rabbitmq': 'RabbitMQ',
    'amqp': 'RabbitMQ',
    'kafka': 'Kafka',
    'bullmq': 'Redis/BullMQ',
    'celery': 'Celery',
    's3': 'S3',
    'minio': 'MinIO',
    'elasticsearch': 'Elasticsearch',
    'meilisearch': 'Meilisearch',
}

ENTRYPOINT_KIND_KEYWORDS = {
    'cli': ('argparse', 'typer', 'click', 'commander', 'yargs'),
    'web_api': ('fastapi', 'flask', 'django', 'uvicorn', 'express', 'nestjs', 'http.handlefunc'),
    'frontend': ('reactdom.createroot', 'createroot(', 'createapp(', 'render(', 'vite/client'),
    'worker': ('celery', 'worker', 'consumer', 'bullmq', 'rq worker'),
    'workflow': ('langgraph', 'stategraph', 'workflow'),
}

ENV_VAR_PATTERNS = (
    re.compile(r'process\.env\.([A-Z][A-Z0-9_]{1,})'),
    re.compile(r'os\.getenv\(\s*[\'\"]([A-Z][A-Z0-9_]{1,})[\'\"]'),
    re.compile(r'getenv\(\s*[\'\"]([A-Z][A-Z0-9_]{1,})[\'\"]'),
    re.compile(r'\$\{([A-Z][A-Z0-9_]{1,})\}'),
    re.compile(r'^\s*([A-Z][A-Z0-9_]{1,})\s*=', re.MULTILINE),
    re.compile(r'^\s*ENV\s+([A-Z][A-Z0-9_]{1,})=', re.MULTILINE),
)


def build_config_summaries(result: AnalysisRunResult) -> list[ConfigSummary]:
    """构建多语言通用的配置摘要列表。"""
    repo_id = result.repo_info.repo_model.full_name
    summaries: list[ConfigSummary] = []

    for item in result.key_file_contents:
        config_identity = _resolve_config_identity(item)
        if config_identity is None:
            continue

        config_kind, language_scope = config_identity
        key_points, scripts_or_commands, service_dependencies, env_vars, related_paths = (
            _extract_config_signals(result, item, config_kind, language_scope)
        )
        frameworks = _pick_related_values(result.project_profile.frameworks, item.content)
        package_managers = _pick_related_values(result.project_profile.package_managers, item.content)
        build_tools = _pick_related_values(result.project_profile.build_tools, item.content)
        test_tools = _pick_related_values(result.project_profile.test_tools, item.content)
        deploy_tools = _pick_related_values(result.project_profile.deploy_tools, item.content)
        subproject_root, subproject_markers = _resolve_subproject_context(result, item.path)
        code_symbols = _extract_file_code_symbols(result, item.path)
        module_relations = _extract_file_module_relations(result, item.path)
        evidence = _build_config_evidence(
            item,
            key_points,
            scripts_or_commands,
            env_vars,
            code_symbols,
            module_relations,
        )

        summaries.append(
            ConfigSummary(
                repo_id=repo_id,
                source_path=item.path,
                config_kind=config_kind,
                language_scope=language_scope,
                frameworks=frameworks,
                package_managers=package_managers,
                build_tools=build_tools,
                test_tools=test_tools,
                deploy_tools=deploy_tools,
                key_points=key_points,
                scripts_or_commands=scripts_or_commands,
                service_dependencies=service_dependencies,
                env_vars=env_vars,
                related_paths=related_paths,
                subproject_root=subproject_root,
                subproject_markers=subproject_markers,
                code_symbols=code_symbols,
                module_relations=module_relations,
                summary=_build_config_summary_text(
                    item=item,
                    config_kind=config_kind,
                    language_scope=language_scope,
                    key_points=key_points,
                    frameworks=frameworks,
                    package_managers=package_managers,
                    build_tools=build_tools,
                    test_tools=test_tools,
                    deploy_tools=deploy_tools,
                    scripts_or_commands=scripts_or_commands,
                    service_dependencies=service_dependencies,
                    env_vars=env_vars,
                    related_paths=related_paths,
                    subproject_root=subproject_root,
                    subproject_markers=subproject_markers,
                    code_symbols=code_symbols,
                    module_relations=module_relations,
                ),
                evidence=evidence,
            )
        )

    return summaries


def build_entrypoint_summaries(result: AnalysisRunResult) -> list[EntrypointSummary]:
    """构建多语言通用的入口摘要列表。"""
    repo_id = result.repo_info.repo_model.full_name
    summaries: list[EntrypointSummary] = []

    for item in result.key_file_contents:
        entrypoint_identity = _resolve_entrypoint_identity(item)
        if entrypoint_identity is None:
            continue

        default_kind, language_scope = entrypoint_identity
        entrypoint_kind = _refine_entrypoint_kind(default_kind, item.content)
        startup_hints = _build_startup_hints(result, item, entrypoint_kind, language_scope)
        startup_commands = _build_startup_commands(result, item, entrypoint_kind, language_scope)
        related_components = _extract_related_components(result, item)
        dependent_configs = _infer_dependent_configs(result, language_scope)
        exposed_interfaces = _extract_exposed_interfaces(item, entrypoint_kind)
        service_dependencies = _extract_service_dependencies(item.content)
        subproject_root, subproject_markers = _resolve_subproject_context(result, item.path)
        code_symbols = _extract_file_code_symbols(result, item.path)
        module_relations = _extract_file_module_relations(result, item.path)
        evidence = _build_entrypoint_evidence(
            item,
            entrypoint_kind,
            startup_commands,
            exposed_interfaces,
            code_symbols,
            module_relations,
        )

        summaries.append(
            EntrypointSummary(
                repo_id=repo_id,
                source_path=item.path,
                entrypoint_kind=entrypoint_kind,
                language_scope=language_scope,
                responsibility=_infer_entrypoint_responsibility(entrypoint_kind, language_scope, item.content),
                startup_hints=startup_hints,
                startup_commands=startup_commands,
                related_components=related_components,
                dependent_configs=dependent_configs,
                exposed_interfaces=exposed_interfaces,
                service_dependencies=service_dependencies,
                subproject_root=subproject_root,
                subproject_markers=subproject_markers,
                code_symbols=code_symbols,
                module_relations=module_relations,
                summary=_build_entrypoint_summary_text(
                    item=item,
                    entrypoint_kind=entrypoint_kind,
                    language_scope=language_scope,
                    related_components=related_components,
                    dependent_configs=dependent_configs,
                    startup_commands=startup_commands,
                    exposed_interfaces=exposed_interfaces,
                    service_dependencies=service_dependencies,
                    subproject_root=subproject_root,
                    subproject_markers=subproject_markers,
                    code_symbols=code_symbols,
                    module_relations=module_relations,
                ),
                evidence=evidence,
            )
        )

    return summaries


def _resolve_config_identity(item: KeyFileContent) -> tuple[str, str] | None:
    """识别配置文件所属的配置类型与语言范围。"""
    lower_name = Path(item.path).name.lower()
    if lower_name in CONFIG_KIND_BY_NAME:
        return CONFIG_KIND_BY_NAME[lower_name]

    if item.path.lower().startswith('.github/workflows/'):
        return 'ci', 'generic'

    return None


def _resolve_entrypoint_identity(item: KeyFileContent) -> tuple[str, str] | None:
    """识别入口文件的基础类型与语言范围。"""
    lower_name = Path(item.path).name.lower()
    return ENTRYPOINT_KIND_BY_NAME.get(lower_name)

def _extract_config_signals(
    result: AnalysisRunResult,
    item: KeyFileContent,
    config_kind: str,
    language_scope: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """从配置文件中提炼结构化检索信号。"""
    lower_name = Path(item.path).name.lower()
    key_points: list[str] = [f'该文件主要承担 {_describe_config_kind(config_kind)} 配置职责']
    scripts_or_commands: list[str] = []
    service_dependencies = _extract_service_dependencies(item.content)
    env_vars = _extract_env_vars(item.content)
    related_paths: list[str] = []

    if lower_name == 'package.json':
        package_data = _load_json(item.content)
        if isinstance(package_data, dict):
            scripts = package_data.get('scripts')
            if isinstance(scripts, dict):
                script_names = list(scripts.keys())
                if script_names:
                    key_points.append(f'定义脚本：{", ".join(script_names[:6])}')
                for script_name, command in list(scripts.items())[:6]:
                    scripts_or_commands.append(f'{script_name}: {str(command).strip()}')

            package_manager = package_data.get('packageManager')
            if isinstance(package_manager, str) and package_manager.strip():
                key_points.append(f'显式声明包管理器：{package_manager.strip()}')

            workspaces = package_data.get('workspaces')
            if isinstance(workspaces, list) and workspaces:
                key_points.append('声明 workspaces，仓库可能采用 monorepo 结构')
                related_paths.extend(str(item) for item in workspaces[:6] if isinstance(item, str))

            bin_field = package_data.get('bin')
            if isinstance(bin_field, str):
                scripts_or_commands.append(f'bin: {bin_field}')
            elif isinstance(bin_field, dict):
                for bin_name, bin_path in list(bin_field.items())[:5]:
                    scripts_or_commands.append(f'bin {bin_name}: {bin_path}')
                    if isinstance(bin_path, str):
                        related_paths.append(bin_path)

            for field_name in ('main', 'module', 'types'):
                field_value = package_data.get(field_name)
                if isinstance(field_value, str) and field_value.strip():
                    related_paths.append(field_value.strip())
                    key_points.append(f'{field_name} 指向 {field_value.strip()}')

    elif lower_name == 'pyproject.toml':
        pyproject = _load_toml(item.content)
        if isinstance(pyproject, dict):
            project_section = pyproject.get('project')
            if isinstance(project_section, dict):
                requires_python = project_section.get('requires-python')
                if isinstance(requires_python, str):
                    key_points.append(f'声明 Python 版本要求：{requires_python}')

                scripts = project_section.get('scripts')
                if isinstance(scripts, dict):
                    key_points.append('声明 project.scripts，可直接暴露命令行入口')
                    for script_name, target in list(scripts.items())[:6]:
                        scripts_or_commands.append(f'{script_name}: {target}')

                optional_dependencies = project_section.get('optional-dependencies')
                if isinstance(optional_dependencies, dict) and optional_dependencies:
                    key_points.append(f'定义可选依赖分组：{", ".join(list(optional_dependencies.keys())[:6])}')

            build_system = pyproject.get('build-system')
            if isinstance(build_system, dict):
                build_backend = build_system.get('build-backend')
                if isinstance(build_backend, str):
                    key_points.append(f'构建后端：{build_backend}')

            tool_section = pyproject.get('tool')
            if isinstance(tool_section, dict):
                if 'poetry' in tool_section:
                    key_points.append('检测到 Poetry 配置')
                if 'pytest' in tool_section:
                    key_points.append('检测到 pytest 配置')
                if 'ruff' in tool_section:
                    key_points.append('检测到 Ruff 配置')
                if 'mypy' in tool_section:
                    key_points.append('检测到 Mypy 配置')

    elif lower_name == 'requirements.txt':
        dependencies = _extract_dependency_names_from_lines(item.content)
        if dependencies:
            key_points.append(f'列出运行依赖：{", ".join(dependencies[:8])}')

    elif lower_name == 'tsconfig.json':
        tsconfig = _load_json(item.content)
        if isinstance(tsconfig, dict):
            compiler_options = tsconfig.get('compilerOptions')
            if isinstance(compiler_options, dict):
                for field_name in ('target', 'module', 'jsx', 'baseUrl', 'rootDir', 'outDir'):
                    field_value = compiler_options.get(field_name)
                    if isinstance(field_value, str) and field_value.strip():
                        key_points.append(f'{field_name}={field_value.strip()}')
                paths = compiler_options.get('paths')
                if isinstance(paths, dict) and paths:
                    key_points.append('声明路径别名映射')

            extends = tsconfig.get('extends')
            if isinstance(extends, str) and extends.strip():
                related_paths.append(extends.strip())
                key_points.append(f'扩展配置：{extends.strip()}')

            include = tsconfig.get('include')
            if isinstance(include, list) and include:
                related_paths.extend(str(item) for item in include[:6] if isinstance(item, str))

    elif lower_name.startswith('vite.config.'):
        plugins = _extract_regex_values(item.content, r'([A-Za-z0-9_]+)\s*\(')
        port = _extract_regex_values(item.content, r'\bport\s*:\s*(\d+)')
        if plugins:
            key_points.append(f'Vite 配置包含插件或工厂调用：{", ".join(plugins[:5])}')
        if port:
            key_points.append(f'开发端口候选：{port[0]}')

    elif lower_name.startswith('next.config.'):
        if 'output:' in item.content:
            key_points.append('声明 Next.js 输出模式')
        if 'images:' in item.content:
            key_points.append('声明图片域名或图片优化配置')

    elif lower_name == 'pom.xml':
        pom_points, pom_paths = _extract_pom_signals(item.content)
        key_points.extend(pom_points)
        related_paths.extend(pom_paths)

    elif lower_name in {'build.gradle', 'build.gradle.kts', 'settings.gradle'}:
        gradle_points, gradle_paths = _extract_gradle_signals(item.content)
        key_points.extend(gradle_points)
        related_paths.extend(gradle_paths)

    elif lower_name == 'go.mod':
        module_name = _extract_regex_values(item.content, r'^\s*module\s+([^\s]+)', multiline=True)
        if module_name:
            key_points.append(f'Go 模块名：{module_name[0]}')
        require_lines = _extract_regex_values(item.content, r'^\s*require\s+([^\s]+)', multiline=True)
        if require_lines:
            key_points.append(f'包含依赖声明：{", ".join(require_lines[:5])}')

    elif lower_name == 'cargo.toml':
        cargo_data = _load_toml(item.content)
        if isinstance(cargo_data, dict):
            package_section = cargo_data.get('package')
            if isinstance(package_section, dict):
                package_name = package_section.get('name')
                if isinstance(package_name, str):
                    key_points.append(f'Rust crate 名称：{package_name}')

            if 'workspace' in cargo_data:
                key_points.append('包含 Cargo workspace 配置')

            if 'bin' in cargo_data:
                key_points.append('声明可执行二进制目标')

    elif lower_name == 'composer.json':
        composer_data = _load_json(item.content)
        if isinstance(composer_data, dict):
            autoload = composer_data.get('autoload')
            if isinstance(autoload, dict):
                key_points.append('声明 Composer autoload 规则')
            scripts = composer_data.get('scripts')
            if isinstance(scripts, dict):
                for script_name, script_value in list(scripts.items())[:6]:
                    scripts_or_commands.append(f'{script_name}: {script_value}')

    elif lower_name == 'gemfile':
        gem_names = _extract_regex_values(item.content, r'^\s*gem\s+[\'\"]([^\'\"]+)[\'\"]', multiline=True)
        if gem_names:
            key_points.append(f'声明 Ruby gems：{", ".join(gem_names[:8])}')

    elif lower_name == 'dockerfile':
        images = _extract_regex_values(item.content, r'^\s*FROM\s+([^\s]+)', multiline=True)
        exposes = _extract_regex_values(item.content, r'^\s*EXPOSE\s+([^\s]+)', multiline=True)
        commands = _extract_regex_values(item.content, r'^\s*(CMD|ENTRYPOINT)\s+(.+)$', multiline=True, group=0)
        if images:
            key_points.append(f'基础镜像：{", ".join(images[:3])}')
        if exposes:
            key_points.append(f'暴露端口：{", ".join(exposes[:4])}')
        scripts_or_commands.extend(commands[:4])

    elif lower_name in {'docker-compose.yml', 'docker-compose.yaml'}:
        compose_services = _extract_compose_services(item.content)
        if compose_services:
            key_points.append(f'定义容器服务：{", ".join(compose_services[:6])}')
        related_paths.extend(_extract_regex_values(item.content, r'^\s*-\s+\.?/?([A-Za-z0-9_./-]+)', multiline=True))

    elif item.path.lower().startswith('.github/workflows/'):
        workflow_points, workflow_commands = _extract_workflow_signals(item.content)
        key_points.extend(workflow_points)
        scripts_or_commands.extend(workflow_commands)

    if not scripts_or_commands and language_scope == 'nodejs':
        scripts_or_commands.extend(_guess_node_commands(result))

    if not key_points:
        key_points.append(f'该文件属于 {language_scope} 生态中的 {config_kind} 配置')

    return (
        _unique_keep_order(key_points),
        _unique_keep_order(scripts_or_commands),
        _unique_keep_order(service_dependencies),
        _unique_keep_order(env_vars),
        _unique_keep_order(related_paths),
    )


def _build_config_summary_text(
    item: KeyFileContent,
    config_kind: str,
    language_scope: str,
    key_points: list[str],
    frameworks: list[str],
    package_managers: list[str],
    build_tools: list[str],
    test_tools: list[str],
    deploy_tools: list[str],
    scripts_or_commands: list[str],
    service_dependencies: list[str],
    env_vars: list[str],
    related_paths: list[str],
    subproject_root: str | None,
    subproject_markers: list[str],
    code_symbols: list[str],
    module_relations: list[str],
) -> str:
    """生成适合 RAG 检索的配置摘要文本。"""
    sentences = [
        f'文件 {item.path} 属于 {language_scope} 生态下的 {config_kind} 配置。',
        f'关键结论：{_join_or_none(key_points)}。',
        f'相关框架：{_join_or_none(frameworks)}；包管理器：{_join_or_none(package_managers)}；构建工具：{_join_or_none(build_tools)}。',
    ]
    if subproject_root:
        sentences.append(f'所属子项目根目录：{subproject_root}。')
    if subproject_markers:
        sentences.append(f'子项目标签：{_join_or_none(subproject_markers)}。')
    if test_tools:
        sentences.append(f'测试工具线索：{_join_or_none(test_tools)}。')
    if deploy_tools:
        sentences.append(f'部署工具线索：{_join_or_none(deploy_tools)}。')
    if scripts_or_commands:
        sentences.append(f'文件中暴露的脚本或命令包括：{_join_or_none(scripts_or_commands)}。')
    if service_dependencies:
        sentences.append(f'可见的外部服务依赖包括：{_join_or_none(service_dependencies)}。')
    if env_vars:
        sentences.append(f'涉及的环境变量包括：{_join_or_none(env_vars)}。')
    if related_paths:
        sentences.append(f'相关路径包括：{_join_or_none(related_paths)}。')
    if code_symbols:
        sentences.append(f'关键符号：{_join_or_none(code_symbols)}。')
    if module_relations:
        sentences.append(f'模块依赖：{_join_or_none(module_relations)}。')
    return ' '.join(sentences)


def _refine_entrypoint_kind(default_kind: str, content: str) -> str:
    """结合文件内容对入口类型做一次轻量修正。"""
    lowered = content.lower()
    for entrypoint_kind, keywords in ENTRYPOINT_KIND_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return entrypoint_kind
    return default_kind

def _build_entrypoint_summary_text(
    item: KeyFileContent,
    entrypoint_kind: str,
    language_scope: str,
    related_components: list[str],
    dependent_configs: list[str],
    startup_commands: list[str],
    exposed_interfaces: list[str],
    service_dependencies: list[str],
    subproject_root: str | None,
    subproject_markers: list[str],
    code_symbols: list[str],
    module_relations: list[str],
) -> str:
    """生成适合 RAG 检索的入口摘要文本。"""
    sentences = [
        f'文件 {item.path} 被识别为 {language_scope} 生态下的 {entrypoint_kind} 类型入口。',
        f'入口相关组件包括：{_join_or_none(related_components)}。',
    ]
    if subproject_root:
        sentences.append(f'所属子项目根目录：{subproject_root}。')
    if subproject_markers:
        sentences.append(f'子项目标签：{_join_or_none(subproject_markers)}。')
    if dependent_configs:
        sentences.append(f'该入口通常需要结合这些配置一起理解：{_join_or_none(dependent_configs)}。')
    if startup_commands:
        sentences.append(f'可优先尝试的启动方式：{_join_or_none(startup_commands)}。')
    if exposed_interfaces:
        sentences.append(f'入口显式暴露的接口线索：{_join_or_none(exposed_interfaces)}。')
    if service_dependencies:
        sentences.append(f'入口依赖的外部服务线索：{_join_or_none(service_dependencies)}。')
    if code_symbols:
        sentences.append(f'关键符号：{_join_or_none(code_symbols)}。')
    if module_relations:
        sentences.append(f'模块依赖：{_join_or_none(module_relations)}。')
    return ' '.join(sentences)


def _infer_entrypoint_responsibility(
    entrypoint_kind: str,
    language_scope: str,
    content: str,
) -> str:
    """根据入口类型与文件内容给出职责描述。"""
    lowered = content.lower()
    responsibility_map = {
        'cli': '负责命令行入口、参数解析或任务调度',
        'web_api': '负责启动服务、注册路由或初始化接口层',
        'frontend': '负责挂载前端应用或初始化页面入口',
        'worker': '负责后台任务消费或异步处理',
        'library': '负责导出库入口或核心公共接口',
        'workflow': '负责组装工作流、状态机或 Agent 图',
    }
    responsibility = responsibility_map.get(entrypoint_kind, f'负责 {language_scope} 项目的核心启动入口')
    if entrypoint_kind == 'web_api' and 'router' in lowered:
        responsibility += '，并且看起来承担了路由装配职责'
    if entrypoint_kind == 'cli' and any(keyword in lowered for keyword in ('typer', 'click', 'argparse')):
        responsibility += '，文件内可见 CLI 框架或参数解析逻辑'
    return responsibility


def _build_startup_hints(
    result: AnalysisRunResult,
    item: KeyFileContent,
    entrypoint_kind: str,
    language_scope: str,
) -> list[str]:
    """结合项目画像和入口类型生成启动线索。"""
    hints: list[str] = []
    if entrypoint_kind == 'cli':
        hints.append(f'可优先从入口文件 `{item.path}` 查看命令注册与参数解析。')
    elif entrypoint_kind == 'web_api':
        hints.append(f'可优先从入口文件 `{item.path}` 查看服务初始化、路由注册与端口设置。')
    elif entrypoint_kind == 'frontend':
        hints.append(f'可优先从入口文件 `{item.path}` 查看前端挂载方式和构建入口。')
    elif entrypoint_kind == 'workflow':
        hints.append(f'可优先从入口文件 `{item.path}` 查看工作流节点装配与执行图定义。')

    dependent_configs = _infer_dependent_configs(result, language_scope)
    if dependent_configs:
        hints.append(f'建议与配置文件 {_join_or_none(dependent_configs)} 联合阅读。')

    package_manager_commands = _guess_node_commands(result)
    if language_scope == 'nodejs' and package_manager_commands:
        hints.append(f'可先查看 package.json 脚本，对应命令候选：{_join_or_none(package_manager_commands)}。')

    return _unique_keep_order(hints)


def _build_startup_commands(
    result: AnalysisRunResult,
    item: KeyFileContent,
    entrypoint_kind: str,
    language_scope: str,
) -> list[str]:
    """为入口文件推断一组可操作的启动命令。"""
    lower_path = item.path.lower()
    commands: list[str] = []

    if language_scope == 'python':
        if lower_path.endswith('manage.py'):
            commands.append(f'python {item.path} runserver')
        elif entrypoint_kind == 'web_api':
            app_var = _extract_python_app_variable(item.content)
            module_name = item.path[:-3].replace('/', '.')
            if app_var:
                commands.append(f'uvicorn {module_name}:{app_var} --reload')
            commands.append(f'python {item.path}')
        elif lower_path.endswith('__main__.py'):
            package_dir = Path(item.path).parent.as_posix().replace('/', '.')
            if package_dir and package_dir != '.':
                commands.append(f'python -m {package_dir}')
            commands.append(f'python {item.path}')
        else:
            commands.append(f'python {item.path}')

    elif language_scope == 'nodejs':
        commands.extend(_guess_node_commands(result))
        if lower_path.endswith(('.ts', '.tsx')):
            commands.append(f'tsx {item.path}')
        if lower_path.endswith(('.js', '.jsx')):
            commands.append(f'node {item.path}')

    elif language_scope == 'go':
        parent_dir = Path(item.path).parent.as_posix()
        commands.append(f'go run ./{parent_dir}' if parent_dir not in {'', '.'} else 'go run .')

    elif language_scope == 'rust':
        commands.append('cargo run')

    elif language_scope == 'java':
        if _has_key_file(result, 'pom.xml'):
            commands.append('mvn spring-boot:run')
        if _has_key_file(result, 'build.gradle') or _has_key_file(result, 'build.gradle.kts'):
            commands.append('./gradlew bootRun')

    return _unique_keep_order(commands)


def _extract_related_components(result: AnalysisRunResult, item: KeyFileContent) -> list[str]:
    """从入口文件中提取与职责最相关的组件线索。"""
    components: list[str] = []
    components.extend(_pick_related_values(result.project_profile.frameworks, item.content))

    lower_path = item.path.lower()
    if lower_path.endswith('.py'):
        components.extend(_extract_regex_values(item.content, r'^\s*from\s+([A-Za-z0-9_\.]+)\s+import', multiline=True))
        components.extend(_extract_regex_values(item.content, r'^\s*import\s+([A-Za-z0-9_\.]+)', multiline=True))
    elif lower_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
        components.extend(_extract_regex_values(item.content, r'from\s+[\'\"]([^\'\"]+)[\'\"]'))
        components.extend(_extract_regex_values(item.content, r'require\(\s*[\'\"]([^\'\"]+)[\'\"]\s*\)'))
    elif lower_path.endswith('.go'):
        components.extend(_extract_regex_values(item.content, r'^\s*"([^"]+)"', multiline=True))
    elif lower_path.endswith('.rs'):
        components.extend(_extract_regex_values(item.content, r'^\s*use\s+([^;]+);', multiline=True))
        components.extend(_extract_regex_values(item.content, r'^\s*mod\s+([A-Za-z0-9_]+)', multiline=True))

    return _unique_keep_order(components)[:8]


def _infer_dependent_configs(result: AnalysisRunResult, language_scope: str) -> list[str]:
    """根据语言范围推断与入口协作的配置文件。"""
    path_candidates = [item.path for item in result.key_file_contents]
    preferred_names_by_language = {
        'python': ('pyproject.toml', 'requirements.txt', 'setup.py', 'setup.cfg', '.env.example', 'dockerfile', 'docker-compose.yml', 'docker-compose.yaml'),
        'nodejs': ('package.json', 'tsconfig.json', 'vite.config.ts', 'vite.config.js', 'vite.config.mjs', 'next.config.ts', 'next.config.js', 'next.config.mjs', '.env.example', 'dockerfile'),
        'go': ('go.mod', 'go.sum', '.env.example', 'dockerfile'),
        'rust': ('cargo.toml', 'cargo.lock', '.env.example', 'dockerfile'),
        'java': ('pom.xml', 'build.gradle', 'build.gradle.kts', 'settings.gradle', '.env.example', 'dockerfile'),
        'generic': ('dockerfile', 'docker-compose.yml', 'docker-compose.yaml', '.env.example'),
    }
    preferred_names = preferred_names_by_language.get(language_scope, preferred_names_by_language['generic'])
    matched = [
        path
        for path in path_candidates
        if Path(path).name.lower() in preferred_names or path.lower().startswith('.github/workflows/')
    ]
    return _unique_keep_order(matched)[:8]


def _extract_exposed_interfaces(item: KeyFileContent, entrypoint_kind: str) -> list[str]:
    """从入口文件中抽取对外暴露接口或对象。"""
    interfaces: list[str] = []
    lower_path = item.path.lower()

    if lower_path.endswith('.py'):
        fastapi_apps = _extract_regex_values(item.content, r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*FastAPI\(', multiline=True)
        flask_apps = _extract_regex_values(item.content, r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Flask\(', multiline=True)
        typer_apps = _extract_regex_values(item.content, r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*typer\.Typer\(', multiline=True)
        click_groups = _extract_regex_values(item.content, r'@click\.group\(\)\s*\ndef\s+([A-Za-z_][A-Za-z0-9_]*)', multiline=True)
        interfaces.extend(f'ASGI app: {name}' for name in fastapi_apps)
        interfaces.extend(f'Flask app: {name}' for name in flask_apps)
        interfaces.extend(f'CLI app: {name}' for name in typer_apps)
        interfaces.extend(f'Click group: {name}' for name in click_groups)
        if 'if __name__ == "__main__"' in item.content:
            interfaces.append('包含直接运行入口 if __name__ == "__main__"')

    elif lower_path.endswith(('.ts', '.js')):
        express_apps = _extract_regex_values(item.content, r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*express\(')
        interfaces.extend(f'Express app: {name}' for name in express_apps)
    elif lower_path.endswith(('.tsx', '.jsx')):
        if 'createRoot(' in item.content or 'ReactDOM.createRoot' in item.content:
            interfaces.append('前端挂载根节点')
        if 'createApp(' in item.content:
            interfaces.append('Vue 应用挂载入口')
    elif lower_path.endswith('.go'):
        routes = _extract_regex_values(item.content, r'HandleFunc\(\s*"([^"]+)"')
        interfaces.extend(f'HTTP route: {route}' for route in routes[:6])
    elif lower_path.endswith('.rs') and entrypoint_kind == 'cli':
        if 'fn main()' in item.content:
            interfaces.append('Rust 可执行入口 main()')

    return _unique_keep_order(interfaces)


def _build_config_evidence(
    item: KeyFileContent,
    key_points: list[str],
    scripts_or_commands: list[str],
    env_vars: list[str],
    code_symbols: list[str],
    module_relations: list[str],
) -> list[str]:
    """构建配置摘要的证据列表。"""
    evidence = [f'来源文件：{item.path}', f'文件大小：{item.size_bytes} 字节']
    if key_points:
        evidence.append(f'关键线索：{key_points[0]}')
    if scripts_or_commands:
        evidence.append(f'脚本线索：{scripts_or_commands[0]}')
    if env_vars:
        evidence.append(f'环境变量线索：{env_vars[0]}')
    if code_symbols:
        evidence.append(f'关键符号线索：{code_symbols[0]}')
    if module_relations:
        evidence.append(f'模块依赖线索：{module_relations[0]}')
    if item.truncated:
        evidence.append('文件内容在读取阶段发生截断')
    return evidence[:6]


def _build_entrypoint_evidence(
    item: KeyFileContent,
    entrypoint_kind: str,
    startup_commands: list[str],
    exposed_interfaces: list[str],
    code_symbols: list[str],
    module_relations: list[str],
) -> list[str]:
    """构建入口摘要的证据列表。"""
    evidence = [
        f'来源文件：{item.path}',
        f'入口类型识别结果：{entrypoint_kind}',
        f'文件大小：{item.size_bytes} 字节',
    ]
    if startup_commands:
        evidence.append(f'启动命令线索：{startup_commands[0]}')
    if exposed_interfaces:
        evidence.append(f'暴露接口线索：{exposed_interfaces[0]}')
    if code_symbols:
        evidence.append(f'关键符号线索：{code_symbols[0]}')
    if module_relations:
        evidence.append(f'模块依赖线索：{module_relations[0]}')
    if item.truncated:
        evidence.append('文件内容在读取阶段发生截断')
    return evidence[:6]

def _extract_service_dependencies(content: str) -> list[str]:
    """从文本中提取数据库、缓存、消息队列等服务依赖。"""
    lowered = content.lower()
    matched = [display_name for keyword, display_name in SERVICE_KEYWORDS.items() if keyword in lowered]
    return _unique_keep_order(matched)


def _extract_env_vars(content: str) -> list[str]:
    """从文本中提取环境变量名。"""
    env_vars: list[str] = []
    for pattern in ENV_VAR_PATTERNS:
        env_vars.extend(pattern.findall(content))
    return _unique_keep_order(env_vars)[:10]


def _extract_dependency_names_from_lines(content: str) -> list[str]:
    """从 requirements 一类的行文本中抽取依赖名。"""
    dependency_names: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        name = re.split(r'[<>=\[\]~! ]', stripped, maxsplit=1)[0].strip()
        if name:
            dependency_names.append(name)
    return _unique_keep_order(dependency_names)


def _extract_pom_signals(content: str) -> tuple[list[str], list[str]]:
    """从 pom.xml 提取构建和依赖信号。"""
    key_points: list[str] = []
    related_paths: list[str] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return key_points, related_paths

    namespace_match = re.match(r'\{(.+)\}', root.tag)
    namespace = {'m': namespace_match.group(1)} if namespace_match else {}
    prefix = 'm:' if namespace else ''

    artifact_id = root.findtext(f'.//{prefix}artifactId', namespaces=namespace) or ''
    packaging = root.findtext(f'.//{prefix}packaging', namespaces=namespace) or ''
    if artifact_id.strip():
        key_points.append(f'构建产物 artifactId：{artifact_id.strip()}')
    if packaging.strip():
        key_points.append(f'打包类型：{packaging.strip()}')

    dependencies = root.findall(f'.//{prefix}dependency/{prefix}artifactId', namespaces=namespace)
    dependency_names = [item.text.strip() for item in dependencies if item.text and item.text.strip()]
    if dependency_names:
        key_points.append(f'依赖工件：{", ".join(dependency_names[:6])}')

    plugins = root.findall(f'.//{prefix}plugin/{prefix}artifactId', namespaces=namespace)
    plugin_names = [item.text.strip() for item in plugins if item.text and item.text.strip()]
    if plugin_names:
        key_points.append(f'构建插件：{", ".join(plugin_names[:5])}')

    return key_points, related_paths


def _extract_gradle_signals(content: str) -> tuple[list[str], list[str]]:
    """从 Gradle 文件中提取插件、依赖与入口线索。"""
    key_points: list[str] = []
    related_paths: list[str] = []

    plugin_ids = _extract_regex_values(content, r'id\s+[\'\"]([^\'\"]+)[\'\"]')
    if plugin_ids:
        key_points.append(f'Gradle 插件：{", ".join(plugin_ids[:6])}')

    dependencies = _extract_regex_values(content, r'(?:implementation|api|testImplementation)\s*\(?\s*[\'\"]([^\'\"]+)[\'\"]')
    if dependencies:
        key_points.append(f'依赖声明：{", ".join(dependencies[:6])}')

    main_class = _extract_regex_values(content, r'mainClass(?:Name)?\s*=\s*[\'\"]([^\'\"]+)[\'\"]')
    if main_class:
        key_points.append(f'主类：{main_class[0]}')

    root_project_name = _extract_regex_values(content, r'rootProject\.name\s*=\s*[\'\"]([^\'\"]+)[\'\"]')
    if root_project_name:
        key_points.append(f'根项目名：{root_project_name[0]}')

    include_projects = _extract_regex_values(content, r'include\(([^)]+)\)')
    if include_projects:
        related_paths.extend(include_projects[:4])

    return key_points, related_paths


def _extract_workflow_signals(content: str) -> tuple[list[str], list[str]]:
    """从 GitHub Actions 工作流中提取执行线索。"""
    key_points: list[str] = []
    commands: list[str] = []

    if 'on:' in content:
        key_points.append('定义 GitHub Actions 触发条件')
    if 'jobs:' in content:
        key_points.append('包含自动化任务作业定义')

    runs_on = _extract_regex_values(content, r'runs-on:\s*([A-Za-z0-9_\-]+)')
    if runs_on:
        key_points.append(f'执行环境：{", ".join(runs_on[:4])}')

    commands.extend(_extract_regex_values(content, r'run:\s*(.+)'))
    return key_points, commands[:6]


def _extract_compose_services(content: str) -> list[str]:
    """用轻量规则提取 docker-compose 的服务名。"""
    services_block = re.search(r'^\s*services:\s*(.*)$', content, flags=re.MULTILINE)
    if not services_block:
        return []

    service_names = _extract_regex_values(
        content,
        r'^\s{2,}([A-Za-z0-9_.-]+):\s*$',
        multiline=True,
    )
    return [name for name in service_names if name not in {'services', 'volumes', 'networks'}]


def _extract_python_app_variable(content: str) -> str | None:
    """抽取 Python Web 入口中的 app 变量名。"""
    for pattern in (
        r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*FastAPI\(',
        r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Flask\(',
    ):
        matches = _extract_regex_values(content, pattern, multiline=True)
        if matches:
            return matches[0]
    return None


def _guess_node_commands(result: AnalysisRunResult) -> list[str]:
    """根据 package.json 和包管理器推断 Node.js 项目的常见命令。"""
    package_json = _find_key_file_content(result, 'package.json')
    if package_json is None:
        return []

    package_data = _load_json(package_json.content)
    if not isinstance(package_data, dict):
        return []

    scripts = package_data.get('scripts')
    if not isinstance(scripts, dict):
        return []

    package_manager = _choose_node_package_manager(result.project_profile.package_managers)
    commands = [f'{package_manager} run {script_name}' for script_name in list(scripts.keys())[:6]]
    if package_manager == 'yarn':
        commands.extend(script_name for script_name in list(scripts.keys())[:3])
    return _unique_keep_order(commands)


def _choose_node_package_manager(package_managers: list[str]) -> str:
    """从项目画像中选出最合适的 Node.js 包管理器命令前缀。"""
    lowered = {item.lower(): item for item in package_managers}
    if 'pnpm' in lowered:
        return 'pnpm'
    if 'yarn' in lowered:
        return 'yarn'
    return 'npm'


def _resolve_subproject_context(result: AnalysisRunResult, source_path: str) -> tuple[str | None, list[str]]:
    """为文件找到最匹配的子项目根目录与标签。"""
    matched: SubprojectSummary | None = None
    for subproject in result.project_profile.subprojects:
        root_path = subproject.root_path
        if root_path == '.':
            if matched is None:
                matched = subproject
            continue
        if source_path == root_path or source_path.startswith(f'{root_path}/'):
            if matched is None or len(root_path) > len(matched.root_path):
                matched = subproject

    if matched is None:
        return None, []
    return matched.root_path, matched.markers


def _extract_file_code_symbols(result: AnalysisRunResult, source_path: str) -> list[str]:
    """提取属于指定文件的关键符号，并转换成便于展示的文本。"""
    related_symbols = [
        item
        for item in result.project_profile.code_symbols
        if item.source_path == source_path
    ]
    return [_format_symbol_evidence(item) for item in related_symbols[:8]]


def _extract_file_module_relations(result: AnalysisRunResult, source_path: str) -> list[str]:
    """提取属于指定文件的模块依赖关系，并转换成便于展示的文本。"""
    related_relations = [
        item
        for item in result.project_profile.module_relations
        if item.source_path == source_path
    ]
    return [_format_relation_evidence(item) for item in related_relations[:8]]


def _format_symbol_evidence(symbol: CodeSymbol) -> str:
    """把符号对象格式化为便于报告与检索复用的证据文本。"""
    if symbol.line_number is None:
        return f'{symbol.symbol_type} {symbol.name}'
    return f'{symbol.symbol_type} {symbol.name} @L{symbol.line_number}'


def _format_relation_evidence(relation: ModuleRelation) -> str:
    """把依赖关系格式化为便于报告与检索复用的证据文本。"""
    if relation.line_number is None:
        return f'{relation.relation_type} {relation.target}'
    return f'{relation.relation_type} {relation.target} @L{relation.line_number}'


def _find_key_file_content(result: AnalysisRunResult, file_name: str) -> KeyFileContent | None:
    """按文件名查找已读取的关键文件内容。"""
    target = file_name.lower()
    for item in result.key_file_contents:
        if Path(item.path).name.lower() == target:
            return item
    return None


def _has_key_file(result: AnalysisRunResult, file_name: str) -> bool:
    """判断分析结果中是否存在指定关键文件。"""
    return _find_key_file_content(result, file_name) is not None


def _load_json(content: str) -> dict | list | None:
    """安全解析 JSON 文本。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _load_toml(content: str) -> dict | None:
    """安全解析 TOML 文本。"""
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _describe_config_kind(config_kind: str) -> str:
    """把配置类型转换成更自然的中文描述。"""
    kind_map = {
        'package_manager': '依赖与包管理',
        'build': '构建与编译',
        'deploy': '部署与运行环境',
        'ci': '持续集成',
    }
    return kind_map.get(config_kind, config_kind)


def _pick_related_values(values: list[str], content: str) -> list[str]:
    """只保留与当前文件内容更相关的一组结构化值。"""
    lowered = content.lower()
    matched = [item for item in values if item.lower() in lowered]
    return matched or values[:3]


def _extract_regex_values(
    text: str,
    pattern: str,
    *,
    multiline: bool = False,
    group: int = 1,
) -> list[str]:
    """用统一方式执行正则提取，并保留出现顺序。"""
    flags = re.MULTILINE if multiline else 0
    results: list[str] = []
    for match in re.finditer(pattern, text, flags):
        try:
            value = match.group(group)
        except IndexError:
            value = match.group(0)
        cleaned = value.strip()
        if cleaned:
            results.append(cleaned)
    return _unique_keep_order(results)


def _unique_keep_order(items: list[str]) -> list[str]:
    """对列表做稳定去重。"""
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_items.append(normalized)
    return unique_items


def _join_or_none(items: list[str]) -> str:
    """把列表转换成适合摘要展示的文本。"""
    return ', '.join(items) if items else '无'
