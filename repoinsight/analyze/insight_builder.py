from repoinsight.models.analysis_model import KeyFileContent, ProjectProfile
from repoinsight.models.file_model import ScanResult
from repoinsight.models.repo_model import RepoInfo


# 这些文件通常能反映项目是否具备基础工程化配置。
ENGINEERING_FILE_NAMES = {
    'dockerfile',
    'docker-compose.yml',
    'docker-compose.yaml',
    'makefile',
    'pyproject.toml',
    'package.json',
    'requirements.txt',
    '.env.example',
}

# 这类文件名常出现在命令行工具项目中，可作为辅助判断依据。
CLI_ENTRY_FILE_NAMES = {
    'main.py',
    'cli.py',
    '__main__.py',
    'app.py',
}


def infer_project_type(
    repo_info: RepoInfo,
    project_profile: ProjectProfile,
    scan_result: ScanResult,
    key_file_contents: list[KeyFileContent],
) -> tuple[str | None, str | None]:
    """根据技术栈、关键文件和 README 粗略推断项目类型。"""
    stack_names = {item.name.lower() for item in project_profile.signals}
    key_file_names = {item.path.split('/')[-1].lower() for item in scan_result.key_files}
    readme_text = (repo_info.readme or '').lower()
    key_text = '\n'.join(item.content.lower() for item in key_file_contents)
    markers = {item.lower() for item in project_profile.project_markers}

    # 优先识别 AI / RAG 类项目，因为这类仓库往往也会同时具备 Web 或 CLI 特征。
    if {'langgraph', 'langchain', 'llamaindex', 'openai'} & stack_names or 'rag' in readme_text:
        return 'AI / RAG 项目', '检测到 LangGraph / LangChain / LlamaIndex / OpenAI / RAG 相关特征'

    if {'streamlit', 'gradio'} & stack_names:
        return '交互式 AI 应用', '检测到 Streamlit 或 Gradio 等交互式应用框架'

    if {'fastapi', 'flask', 'django', 'express', 'nestjs'} & stack_names:
        return 'Web 服务', '检测到 Web 框架或服务端应用特征'

    if {'react', 'vue', 'next.js'} & stack_names:
        return '前端或全栈项目', '检测到前端框架相关特征'

    # 除了框架外，也尝试从入口文件名和源码片段中判断是否为命令行工具。
    if {'typer', 'click'} & stack_names or CLI_ENTRY_FILE_NAMES & key_file_names or 'cli' in markers:
        return 'CLI 工具', '检测到 CLI 相关框架或典型入口文件'

    if 'argparse' in key_text or 'typer' in key_text or 'click' in key_text:
        return 'CLI 工具', '在关键文件内容中检测到 argparse / Typer / Click 相关调用'

    # 如果更偏向 Docker、Compose、Makefile 一类配置文件，可先归为部署或基础设施项目。
    if 'docker' in stack_names and len(key_file_contents) <= 3:
        return '部署或基础设施项目', '检测到 Docker 相关文件，且关键文件偏部署配置'

    return None, None


def build_strengths(
    repo_info: RepoInfo,
    scan_result: ScanResult,
    project_profile: ProjectProfile,
) -> list[str]:
    """根据已有信息生成一组偏确定性的正向观察。"""
    strengths: list[str] = []
    stack_names = {item.name.lower() for item in project_profile.signals}
    key_file_names = {item.path.split('/')[-1].lower() for item in scan_result.key_files}

    if repo_info.readme and repo_info.readme.strip():
        strengths.append('README 已成功获取，说明项目基础文档相对完整。')

    if repo_info.repo_model.license_name:
        strengths.append('仓库包含许可证信息，开源使用边界更清晰。')

    if project_profile.primary_language:
        strengths.append(f'已识别出主语言为 {project_profile.primary_language}，后续可以做更细粒度的语言专项分析。')

    if key_file_names & ENGINEERING_FILE_NAMES:
        strengths.append('检测到工程化配置文件，说明项目结构相对规范。')

    if 'docker' in stack_names:
        strengths.append('检测到 Docker 相关配置，通常说明项目具备一定部署友好性。')

    if scan_result.stats.key_file_count >= 3:
        strengths.append('关键文件数量较充足，便于快速理解项目结构和入口。')

    return strengths


def build_risks(
    repo_info: RepoInfo,
    scan_result: ScanResult,
    project_profile: ProjectProfile,
) -> list[str]:
    """根据已有信息生成一组偏保守的风险提示。"""
    risks: list[str] = []
    stack_names = {item.name.lower() for item in project_profile.signals}

    if not repo_info.readme or not repo_info.readme.strip():
        risks.append('README 缺失或未成功获取，项目上手成本可能较高。')

    if not repo_info.repo_model.license_name:
        risks.append('未检测到许可证信息，直接复用前建议确认开源使用边界。')

    if scan_result.stats.key_file_count <= 1:
        risks.append('关键文件较少，当前自动分析可能无法完整反映项目结构。')

    if scan_result.stats.kept_count > 200 and scan_result.stats.key_file_count < 5:
        risks.append('候选文件较多但关键文件较少，项目结构可能较复杂或入口不明显。')

    # Python 项目很常见，但若没有容器化配置，后续部署方式可能还需要人工补充。
    if 'docker' not in stack_names and project_profile.primary_language == 'Python':
        risks.append('暂未检测到容器化配置，部署方式可能需要进一步手动确认。')

    if not project_profile.frameworks and not project_profile.build_tools:
        risks.append('暂未识别出明显框架或构建工具，项目类型判断可能还不够稳定。')

    return risks


def build_observations(
    repo_info: RepoInfo,
    scan_result: ScanResult,
    project_profile: ProjectProfile,
    project_type: str | None,
) -> list[str]:
    """汇总一些中性、偏描述性的初步观察。"""
    observations: list[str] = []
    stack_names = [item.name for item in project_profile.signals]

    if project_type:
        observations.append(f'该仓库初步被归类为：{project_type}。')

    if stack_names:
        observations.append(f'当前规则识别出的技术栈包括：{", ".join(stack_names)}。')

    if project_profile.package_managers:
        observations.append(
            f'当前识别出的包管理器包括：{", ".join(project_profile.package_managers)}。'
        )

    if project_profile.entrypoints:
        observations.append(
            f'当前识别出的入口文件包括：{", ".join(project_profile.entrypoints[:5])}。'
        )

    if repo_info.repo_model.topics:
        observations.append(
            f'GitHub Topics 显示项目关注方向包括：{", ".join(repo_info.repo_model.topics)}。'
        )

    observations.append(
        f'扫描阶段保留了 {scan_result.stats.kept_count} 个候选文件，并识别出 '
        f'{scan_result.stats.key_file_count} 个关键文件。'
    )

    return observations


def build_repo_insights(
    repo_info: RepoInfo,
    scan_result: ScanResult,
    project_profile: ProjectProfile,
    key_file_contents: list[KeyFileContent],
) -> tuple[str | None, str | None, list[str], list[str], list[str]]:
    """统一构建项目类型、优势、风险和初步观察。"""
    project_type, project_type_evidence = infer_project_type(
        repo_info=repo_info,
        project_profile=project_profile,
        scan_result=scan_result,
        key_file_contents=key_file_contents,
    )
    strengths = build_strengths(
        repo_info=repo_info,
        scan_result=scan_result,
        project_profile=project_profile,
    )
    risks = build_risks(
        repo_info=repo_info,
        scan_result=scan_result,
        project_profile=project_profile,
    )
    observations = build_observations(
        repo_info=repo_info,
        scan_result=scan_result,
        project_profile=project_profile,
        project_type=project_type,
    )
    return project_type, project_type_evidence, strengths, risks, observations
