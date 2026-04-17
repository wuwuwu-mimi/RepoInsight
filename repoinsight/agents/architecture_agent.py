from repoinsight.agents.code_agent import (
    build_code_investigation_context_lines,
    investigate_code_hits,
    should_use_code_investigation_context,
)
from repoinsight.agents.models import CodeInvestigationResult
from repoinsight.models.rag_model import SearchHit
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR


def investigate_architecture_hits(
    question: str,
    hits: list[SearchHit],
    *,
    repo_id: str | None = None,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
    max_hits: int = 4,
    max_follow_steps: int = 6,
    max_follow_depth: int = 2,
) -> CodeInvestigationResult | None:
    """以 architecture 视角复用现有代码调查能力，提炼模块与依赖链路。"""
    return investigate_code_hits(
        question,
        hits,
        'architecture',
        repo_id=repo_id,
        target_dir=target_dir,
        max_hits=max_hits,
        max_follow_steps=max_follow_steps,
        max_follow_depth=max_follow_depth,
    )


def build_architecture_investigation_context_lines(
    investigation: CodeInvestigationResult,
    *,
    max_lines: int = 6,
) -> list[str]:
    """把架构调查结果转换为 synthesis_agent 可直接吸收的上下文。"""
    lines = build_code_investigation_context_lines(investigation, max_lines=max_lines)
    normalized_lines: list[str] = []
    for line in lines:
        if line.startswith('[code_agent]'):
            normalized_lines.append(line.replace('[code_agent]', '[architecture_agent]', 1))
        else:
            normalized_lines.append(line)
    return normalized_lines


def should_use_architecture_investigation_context(
    result: CodeInvestigationResult | None,
) -> bool:
    """沿用既有质量阈值，判断架构调查结果是否值得注入回答上下文。"""
    return should_use_code_investigation_context(result)
