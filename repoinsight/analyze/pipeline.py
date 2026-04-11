import asyncio

from repoinsight.analyze.file_reader import read_key_files
from repoinsight.analyze.insight_builder import build_repo_insights
from repoinsight.analyze.project_profile_inference import infer_project_profile
from repoinsight.analyze.stack_inference import infer_tech_stack
from repoinsight.ingest.file_scanner import scan_repo
from repoinsight.ingest.git_loader import clone_repo
from repoinsight.ingest.repo_service import get_repo_info
from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.utils.check_util import check_url



def run_analysis(url: str) -> AnalysisRunResult:
    """串联 analyze 主流程：元数据 -> 克隆 -> 扫描 -> 读取关键文件。"""
    if not check_url(url):
        raise ValueError('不是合法的 GitHub 仓库地址')

    repo_info = get_repo_info(url)
    clone_path = asyncio.run(clone_repo(url=url))
    if not clone_path:
        raise RuntimeError('仓库克隆失败')

    scan_result = scan_repo(root_path=clone_path)
    key_file_contents = read_key_files(
        root_path=clone_path,
        key_files=scan_result.key_files,
    )
    project_profile = infer_project_profile(
        repo_info=repo_info,
        scan_result=scan_result,
        key_file_contents=key_file_contents,
    )
    tech_stack = infer_tech_stack(project_profile)
    project_type, project_type_evidence, strengths, risks, observations = build_repo_insights(
        repo_info=repo_info,
        scan_result=scan_result,
        project_profile=project_profile,
        key_file_contents=key_file_contents,
    )

    return AnalysisRunResult(
        repo_info=repo_info,
        clone_path=clone_path,
        scan_result=scan_result,
        key_file_contents=key_file_contents,
        tech_stack=tech_stack,
        project_profile=project_profile,
        project_type=project_type,
        project_type_evidence=project_type_evidence,
        strengths=strengths,
        risks=risks,
        observations=observations,
    )
