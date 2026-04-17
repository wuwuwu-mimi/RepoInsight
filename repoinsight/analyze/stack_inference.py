from repoinsight.models.analysis_model import ProjectProfile, TechStackItem


def infer_tech_stack(project_profile: ProjectProfile) -> list[TechStackItem]:
    """从结构化项目画像中提取技术栈列表。"""
    if project_profile.confirmed_signals:
        return list(project_profile.confirmed_signals)
    return list(project_profile.signals)
