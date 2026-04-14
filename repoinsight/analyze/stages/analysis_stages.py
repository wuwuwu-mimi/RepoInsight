import asyncio
from collections.abc import Callable

from repoinsight.analyze.file_reader import read_key_files
from repoinsight.analyze.insight_builder import build_repo_insights
from repoinsight.analyze.project_profile_inference import infer_project_profile
from repoinsight.analyze.stack_inference import infer_tech_stack
from repoinsight.ingest.file_scanner import scan_repo
from repoinsight.ingest.git_loader import clone_repo
from repoinsight.ingest.repo_service import get_repo_info
from repoinsight.models.analysis_model import AnalysisState
from repoinsight.utils.check_util import check_url


AnalysisStage = Callable[[AnalysisState], AnalysisState]


def build_state_from_url(url: str) -> AnalysisState:
    """根据输入 URL 初始化一份分析状态。"""
    return AnalysisState(url=url)


def validate_analysis_url_stage(state: AnalysisState) -> AnalysisState:
    """校验输入仓库地址是否合法。"""
    if not check_url(state.url):
        raise ValueError('不是合法的 GitHub 仓库地址')
    return state


def fetch_repo_metadata_stage(state: AnalysisState) -> AnalysisState:
    """获取仓库元数据与 README。"""
    state.repo_info = get_repo_info(state.url)
    return state


def clone_repository_stage(state: AnalysisState) -> AnalysisState:
    """克隆仓库到本地分析目录。"""
    clone_path = asyncio.run(clone_repo(url=state.url))
    if not clone_path:
        raise RuntimeError('仓库克隆失败')
    state.clone_path = clone_path
    return state


def scan_repository_stage(state: AnalysisState) -> AnalysisState:
    """扫描仓库目录并筛选关键文件。"""
    if not state.clone_path:
        raise RuntimeError('仓库尚未完成克隆，无法执行扫描阶段')
    state.scan_result = scan_repo(root_path=state.clone_path)
    return state


def read_key_files_stage(state: AnalysisState) -> AnalysisState:
    """读取扫描阶段选出的关键文件内容。"""
    if not state.clone_path or state.scan_result is None:
        raise RuntimeError('仓库扫描结果缺失，无法读取关键文件')
    state.key_file_contents = read_key_files(
        root_path=state.clone_path,
        key_files=state.scan_result.key_files,
    )
    return state


def build_project_profile_stage(state: AnalysisState) -> AnalysisState:
    """根据元数据、扫描结果和关键文件构建项目画像。"""
    if state.repo_info is None or state.scan_result is None:
        raise RuntimeError('项目画像阶段缺少元数据或扫描结果')
    state.project_profile = infer_project_profile(
        repo_info=state.repo_info,
        scan_result=state.scan_result,
        key_file_contents=state.key_file_contents,
    )
    return state


def build_tech_stack_stage(state: AnalysisState) -> AnalysisState:
    """根据项目画像推断技术栈。"""
    state.tech_stack = infer_tech_stack(state.project_profile)
    return state


def build_repo_insights_stage(state: AnalysisState) -> AnalysisState:
    """生成项目类型、优点、风险和观察结论。"""
    if state.repo_info is None or state.scan_result is None:
        raise RuntimeError('洞察阶段缺少元数据或扫描结果')
    (
        state.project_type,
        state.project_type_evidence,
        state.strengths,
        state.risks,
        state.observations,
    ) = build_repo_insights(
        repo_info=state.repo_info,
        scan_result=state.scan_result,
        project_profile=state.project_profile,
        key_file_contents=state.key_file_contents,
    )
    return state


ANALYSIS_STAGES: tuple[AnalysisStage, ...] = (
    validate_analysis_url_stage,
    fetch_repo_metadata_stage,
    clone_repository_stage,
    scan_repository_stage,
    read_key_files_stage,
    build_project_profile_stage,
    build_tech_stack_stage,
    build_repo_insights_stage,
)
