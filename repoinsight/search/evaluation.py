from pydantic import BaseModel, Field

from repoinsight.models.rag_model import SearchHit, SearchResult
from repoinsight.search.service import search_knowledge_base
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR


class RagEvaluationCase(BaseModel):
    """表示一条检索验证问题。"""

    # 用于唯一标识问题，便于后续做回归对比。
    case_id: str = Field(..., description='问题标识')

    # 面向 RAG 检索的自然语言问题。
    query: str = Field(..., description='检索问题')

    # 期望命中的仓库标识；为空时不限制。
    expected_repo_id: str | None = Field(default=None, description='期望仓库标识')

    # 期望命中的文档类型，例如 config_summary、entrypoint_summary。
    expected_doc_types: list[str] = Field(default_factory=list, description='期望文档类型')

    # 期望命中的源文件路径，例如 package.json、app.py。
    expected_source_paths: list[str] = Field(default_factory=list, description='期望源文件路径')

    # 期望在命中文档里至少出现一个的关键词。
    expected_terms_any: list[str] = Field(default_factory=list, description='期望关键词')

    # 当前问题允许检查的命中数量。
    top_k: int = Field(default=5, description='检查命中数量')


class RagEvaluationCaseResult(BaseModel):
    """表示单条检索问题的验证结果。"""

    # 对应的原始验证问题。
    case: RagEvaluationCase = Field(..., description='验证问题')

    # 当前问题是否通过。
    passed: bool = Field(..., description='是否通过')

    # 实际使用的检索后端。
    backend: str = Field(default='local', description='检索后端')

    # 命中的文档类型列表，便于调试排序效果。
    hit_doc_types: list[str] = Field(default_factory=list, description='命中文档类型')

    # 命中的源文件路径列表，便于调试排序效果。
    hit_source_paths: list[str] = Field(default_factory=list, description='命中源文件路径')

    # 本次判定的说明文字。
    reason: str = Field(..., description='判定原因')


class RagEvaluationReport(BaseModel):
    """表示一轮问题集回归验证结果。"""

    # 总问题数。
    total_cases: int = Field(default=0, description='总问题数')

    # 通过的问题数。
    passed_cases: int = Field(default=0, description='通过问题数')

    # 未通过的问题数。
    failed_cases: int = Field(default=0, description='失败问题数')

    # 单题明细结果。
    results: list[RagEvaluationCaseResult] = Field(default_factory=list, description='明细结果')


def build_mvp_rag_eval_cases(repo_id: str) -> list[RagEvaluationCase]:
    """生成一组适合当前 MVP 的通用检索问题模板。"""
    return [
        RagEvaluationCase(
            case_id='repo-purpose',
            query='这个项目是做什么的',
            expected_repo_id=repo_id,
            expected_doc_types=['repo_summary', 'readme_summary'],
        ),
        RagEvaluationCase(
            case_id='startup-command',
            query='这个项目怎么启动',
            expected_repo_id=repo_id,
            expected_doc_types=['entrypoint_summary', 'readme_summary'],
        ),
        RagEvaluationCase(
            case_id='entrypoint-location',
            query='主入口文件在哪里',
            expected_repo_id=repo_id,
            expected_doc_types=['entrypoint_summary'],
        ),
        RagEvaluationCase(
            case_id='config-runtime',
            query='项目的关键配置文件有哪些',
            expected_repo_id=repo_id,
            expected_doc_types=['config_summary', 'repo_fact'],
        ),
        RagEvaluationCase(
            case_id='env-vars',
            query='这个项目依赖哪些环境变量',
            expected_repo_id=repo_id,
            expected_doc_types=['config_summary'],
        ),
    ]


def build_code_rag_eval_cases(repo_id: str) -> list[RagEvaluationCase]:
    """生成一组面向代码级问答的检索验证问题。"""
    return [
        RagEvaluationCase(
            case_id='implementation-handle-login',
            query='AuthService.handle_login 是怎么实现的？',
            expected_repo_id=repo_id,
            expected_doc_types=['function_body_chunk', 'function_summary'],
            expected_terms_any=['create_session_token'],
        ),
        RagEvaluationCase(
            case_id='api-post-login',
            query='POST /login 是怎么实现的？',
            expected_repo_id=repo_id,
            expected_doc_types=['route_handler_chunk', 'api_route_summary'],
            expected_terms_any=['handle_login'],
        ),
        RagEvaluationCase(
            case_id='api-post-session',
            query='POST /session 是怎么实现的？',
            expected_repo_id=repo_id,
            expected_doc_types=['route_handler_chunk', 'function_body_chunk', 'api_route_summary'],
            expected_terms_any=['auth_service.login_user', 'session_repo.persist_session'],
            top_k=6,
        ),
    ]


def build_full_rag_eval_cases(repo_id: str) -> list[RagEvaluationCase]:
    """生成覆盖通用问题与代码级问题的完整评测集合。"""
    return [
        *build_mvp_rag_eval_cases(repo_id),
        *build_code_rag_eval_cases(repo_id),
    ]


def build_multilang_code_rag_eval_cases(repo_id: str) -> list[RagEvaluationCase]:
    """生成覆盖 TypeScript、Go、Rust 的多语言代码级检索评测集合。"""
    return [
        RagEvaluationCase(
            case_id='ts-implementation-create-session',
            query='createSession 是怎么实现的？',
            expected_repo_id=repo_id,
            expected_doc_types=['function_body_chunk', 'function_summary'],
            expected_source_paths=['src/routes.ts'],
            expected_terms_any=['sessionService.create'],
        ),
        RagEvaluationCase(
            case_id='ts-api-post-session',
            query='POST /session 是怎么实现的？',
            expected_repo_id=repo_id,
            expected_doc_types=['route_handler_chunk', 'api_route_summary'],
            expected_source_paths=['src/routes.ts'],
            expected_terms_any=['createSession'],
        ),
        RagEvaluationCase(
            case_id='go-implementation-build-login-token',
            query='BuildLoginToken 是怎么实现的？',
            expected_repo_id=repo_id,
            expected_doc_types=['function_body_chunk', 'function_summary'],
            expected_source_paths=['cmd/server/main.go'],
            expected_terms_any=['signToken'],
        ),
        RagEvaluationCase(
            case_id='rust-implementation-persist-session',
            query='persist_session 是怎么实现的？',
            expected_repo_id=repo_id,
            expected_doc_types=['function_body_chunk', 'function_summary'],
            expected_source_paths=['src/lib.rs'],
            expected_terms_any=['write_store'],
        ),
    ]


def evaluate_search_cases(
    cases: list[RagEvaluationCase],
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> RagEvaluationReport:
    """逐条执行检索问题，并按照期望规则判定是否通过。"""
    results = [
        evaluate_single_case(case=case, target_dir=target_dir)
        for case in cases
    ]
    passed_cases = sum(1 for item in results if item.passed)
    return RagEvaluationReport(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=len(results) - passed_cases,
        results=results,
    )


def evaluate_single_case(
    case: RagEvaluationCase,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> RagEvaluationCaseResult:
    """执行单条检索问题验证。"""
    result = search_knowledge_base(query=case.query, top_k=case.top_k, target_dir=target_dir)
    matched_hit = _find_first_matching_hit(case, result)

    hit_doc_types = [item.document.doc_type for item in result.hits]
    hit_source_paths = [item.document.source_path or '' for item in result.hits]

    if matched_hit is not None:
        return RagEvaluationCaseResult(
            case=case,
            passed=True,
            backend=result.backend,
            hit_doc_types=hit_doc_types,
            hit_source_paths=hit_source_paths,
            reason=_build_pass_reason(case, matched_hit),
        )

    return RagEvaluationCaseResult(
        case=case,
        passed=False,
        backend=result.backend,
        hit_doc_types=hit_doc_types,
        hit_source_paths=hit_source_paths,
        reason=_build_fail_reason(case, result),
    )


def _find_first_matching_hit(case: RagEvaluationCase, result: SearchResult) -> SearchHit | None:
    """从命中结果里找到第一个满足验证条件的文档。"""
    for hit in result.hits:
        if not _match_repo(case, hit):
            continue
        if not _match_doc_type(case, hit):
            continue
        if not _match_source_path(case, hit):
            continue
        if not _match_terms(case, hit):
            continue
        return hit
    return None


def _match_repo(case: RagEvaluationCase, hit: SearchHit) -> bool:
    """判断命中仓库是否满足要求。"""
    if not case.expected_repo_id:
        return True
    return hit.document.repo_id == case.expected_repo_id


def _match_doc_type(case: RagEvaluationCase, hit: SearchHit) -> bool:
    """判断文档类型是否满足要求。"""
    if not case.expected_doc_types:
        return True
    return hit.document.doc_type in case.expected_doc_types


def _match_source_path(case: RagEvaluationCase, hit: SearchHit) -> bool:
    """判断源文件路径是否满足要求。"""
    if not case.expected_source_paths:
        return True
    return (hit.document.source_path or '') in case.expected_source_paths


def _match_terms(case: RagEvaluationCase, hit: SearchHit) -> bool:
    """判断命中文档是否包含期望关键词。"""
    if not case.expected_terms_any:
        return True

    search_space = ' '.join(
        [
            hit.document.title,
            hit.document.content,
            hit.snippet,
        ]
    ).lower()
    return any(term.lower() in search_space for term in case.expected_terms_any)


def _build_pass_reason(case: RagEvaluationCase, hit: SearchHit) -> str:
    """生成通过时的说明文字。"""
    path = hit.document.source_path or '仓库级文档'
    return (
        f'问题 `{case.case_id}` 命中了 {hit.document.doc_type} 文档，'
        f'来源为 {path}。'
    )


def _build_fail_reason(case: RagEvaluationCase, result: SearchResult) -> str:
    """生成失败时的说明文字。"""
    if not result.hits:
        return f'问题 `{case.case_id}` 没有命中任何文档。'

    top_hit = result.hits[0]
    path = top_hit.document.source_path or '仓库级文档'
    return (
        f'问题 `{case.case_id}` 未命中预期文档；'
        f'当前首条结果是 {top_hit.document.doc_type} / {path}。'
    )
