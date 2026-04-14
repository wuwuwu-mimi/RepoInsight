import math
import re
from collections import Counter

from repoinsight.models.rag_model import KnowledgeDocument, SearchHit, SearchResult
from repoinsight.storage.chroma_store import search_documents_in_chroma
from repoinsight.storage.local_knowledge_store import (
    DEFAULT_KNOWLEDGE_DIR,
    load_all_documents,
    load_repo_documents,
)


# 过滤掉过短 token，避免大量单字符噪音影响排序。
MIN_TOKEN_LENGTH = 2

# 不同查询意图对应的关键词，用于做轻量级检索路由。
QUERY_INTENT_KEYWORDS = {
    'overview': ('做什么', '是什么', '介绍', '概述', 'overview', 'purpose', '用途', '功能'),
    'startup': ('启动', '运行', 'run', 'start', 'dev', 'server', '怎么跑', '如何启动'),
    'entrypoint': ('入口', 'main', 'entrypoint', '从哪里开始', '启动文件', 'server.py', 'app.py'),
    'env': ('环境变量', 'env', '配置项', 'token', 'secret', 'api key', 'apikey'),
    'config': ('配置', '构建', '打包', '脚本', '依赖', 'package', 'install', '编译'),
    'tech': ('技术栈', '框架', '语言', 'runtime', 'tech stack', '用了什么', '使用了什么'),
    'architecture': ('架构', '模块', '依赖关系', '调用链', '子项目', 'monorepo', 'workspace', 'service', '包结构'),
    'api': ('接口', '路由', 'api', 'endpoint', 'http', 'rest', 'post /', 'get /', 'put /', 'delete /', 'patch /'),
    'implementation': ('实现', '实现逻辑', '怎么实现', '如何实现', '代码逻辑', '源码', '函数', '方法', '类', 'method', 'function', '代码'),
}

# 查询意图与文档类型的加权关系。
INTENT_DOC_TYPE_MULTIPLIERS = {
    'overview': {
        'readme_summary': 2.60,
        'repo_summary': 2.20,
        'repo_fact': 1.45,
        'entrypoint_summary': 0.82,
        'config_summary': 0.78,
        'subproject_summary': 0.62,
        'key_file_summary': 0.68,
        'function_summary': 0.55,
        'class_summary': 0.55,
        'api_route_summary': 0.60,
    },
    'startup': {
        'entrypoint_summary': 1.65,
        'config_summary': 1.30,
        'key_file_summary': 1.12,
        'subproject_summary': 1.08,
        'repo_summary': 0.92,
        'readme_summary': 0.98,
    },
    'entrypoint': {
        'entrypoint_summary': 1.75,
        'key_file_summary': 1.18,
        'subproject_summary': 1.12,
        'repo_summary': 0.90,
    },
    'env': {
        'config_summary': 1.70,
        'key_file_summary': 1.18,
        'entrypoint_summary': 1.08,
        'repo_summary': 0.88,
    },
    'config': {
        'config_summary': 1.55,
        'repo_fact': 1.18,
        'key_file_summary': 1.08,
        'entrypoint_summary': 1.04,
    },
    'tech': {
        'repo_fact': 1.35,
        'repo_summary': 1.22,
        'config_summary': 1.18,
        'subproject_summary': 1.10,
    },
    'architecture': {
        'subproject_summary': 1.55,
        'entrypoint_summary': 1.24,
        'key_file_summary': 1.22,
        'config_summary': 1.12,
        'repo_summary': 1.05,
    },
    'api': {
        'api_route_summary': 1.95,
        'function_summary': 1.18,
        'entrypoint_summary': 1.12,
        'key_file_summary': 1.10,
        'subproject_summary': 1.04,
    },
    'implementation': {
        'function_summary': 1.85,
        'class_summary': 1.55,
        'key_file_summary': 1.18,
        'entrypoint_summary': 1.08,
        'subproject_summary': 1.05,
    },
}

# 不同查询意图关注的结构化 metadata 字段。
INTENT_METADATA_WEIGHTS = {
    'overview': {
        'project_type': 4.5,
        'frameworks': 2.6,
        'languages': 2.4,
        'runtimes': 2.2,
        'entrypoints': 1.8,
        'project_markers': 1.8,
    },
    'startup': {
        'entrypoint_startup_commands': 4.2,
        'entrypoint_exposed_interfaces': 3.4,
        'entrypoints': 3.2,
        'config_scripts_or_commands': 3.0,
    },
    'entrypoint': {
        'entrypoints': 4.5,
        'source_path': 3.6,
        'entrypoint_exposed_interfaces': 3.6,
        'code_symbol_names': 3.1,
    },
    'env': {
        'config_env_vars': 5.0,
        'config_key_points': 2.4,
        'module_relation_targets': 1.2,
    },
    'config': {
        'config_key_points': 3.8,
        'config_scripts_or_commands': 3.2,
        'config_related_paths': 3.0,
        'package_managers': 2.8,
        'build_tools': 2.8,
        'deploy_tools': 2.4,
    },
    'tech': {
        'frameworks': 4.0,
        'runtimes': 3.8,
        'languages': 3.6,
        'build_tools': 3.4,
        'package_managers': 3.4,
        'test_tools': 3.0,
        'deploy_tools': 2.8,
        'module_relation_targets': 1.8,
    },
    'architecture': {
        'subproject_roots': 4.2,
        'subproject_markers': 3.2,
        'code_symbol_names': 4.1,
        'module_relation_targets': 4.1,
        'code_entity_names': 4.0,
        'code_entity_refs': 4.2,
        'code_relation_targets': 3.8,
        'code_relation_sources': 3.2,
        'code_relation_types': 2.4,
        'code_symbols': 2.6,
        'module_relations': 2.6,
        'source_path': 2.2,
    },
    'api': {
        'route_path': 5.0,
        'http_methods': 4.6,
        'handler_name': 4.4,
        'handler_qualified_name': 4.4,
        'api_route_paths': 4.2,
        'api_handler_names': 4.0,
        'code_entity_names': 3.8,
        'code_entity_refs': 4.2,
        'code_relation_targets': 3.0,
        'code_relation_sources': 2.8,
        'code_relation_types': 2.4,
        'called_symbols': 3.0,
        'source_path': 2.8,
    },
    'implementation': {
        'symbol_name': 5.0,
        'qualified_name': 4.6,
        'owner_class': 3.6,
        'called_symbols': 4.0,
        'class_methods': 3.8,
        'function_names': 3.6,
        'class_names': 3.4,
        'code_entity_names': 4.0,
        'code_entity_refs': 4.4,
        'code_relation_targets': 3.4,
        'code_relation_sources': 3.0,
        'code_relation_types': 2.2,
        'source_path': 2.6,
    },
}

# 无论查询意图如何，技术栈相关字段都适合做一层显式匹配加权。
GENERIC_STRUCTURED_FIELDS = {
    'frameworks': 2.2,
    'runtimes': 2.0,
    'languages': 2.0,
    'build_tools': 1.8,
    'package_managers': 1.8,
    'code_symbol_names': 1.6,
    'module_relation_targets': 1.6,
    'code_entity_names': 1.8,
    'code_entity_refs': 1.8,
    'code_relation_targets': 1.6,
    'code_relation_sources': 1.4,
    'code_relation_types': 1.2,
    'api_route_paths': 1.8,
    'api_handler_names': 1.8,
    'function_names': 1.8,
    'class_names': 1.8,
    'symbol_name': 1.8,
    'qualified_name': 1.8,
    'route_path': 1.8,
    'handler_qualified_name': 1.8,
    'called_symbols': 1.6,
}


# 单仓库问答时，文档量通常不大，直接走本地重排会比先探测 Chroma 更稳定。
LOCAL_ONLY_REPO_DOC_THRESHOLD = 200

# 全局检索在知识库规模较小时，本地排序也足够快，且能避开损坏向量库带来的启动开销。
LOCAL_ONLY_GLOBAL_DOC_THRESHOLD = 500


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
    repo_id: str | None = None,
) -> SearchResult:
    """优先执行向量检索；若未启用则回退到本地轻量检索。"""
    documents = _load_search_documents(target_dir=target_dir, repo_id=repo_id)
    query_tokens = _tokenize(query)
    query_intent = _infer_query_intent(query)
    repo_count = len({item.repo_id for item in documents})
    document_count = len(documents)

    if not documents or not query_tokens:
        return SearchResult(
            query=query,
            hits=[],
            backend='local',
            repo_count=repo_count,
            document_count=document_count,
        )

    if _should_try_chroma(
        repo_id=repo_id,
        document_count=document_count,
        repo_count=repo_count,
    ):
        try:
            vector_hits = search_documents_in_chroma(
                query=query,
                top_k=top_k,
                repo_id=repo_id,
            )
            if vector_hits:
                reranked_hits = _rerank_existing_hits(vector_hits, query, query_tokens, query_intent)
                return SearchResult(
                    query=query,
                    hits=reranked_hits[:top_k],
                    backend='chroma',
                    repo_count=repo_count,
                    document_count=document_count,
                )
        except Exception:
            # Chroma 不可用或索引异常时，自动退回本地检索，避免问答直接失败。
            pass

    document_scores: list[SearchHit] = []
    idf = _build_idf(documents)
    for document in documents:
        score = _score_document(query, query_tokens, query_intent, document, idf)
        if score <= 0:
            continue

        snippet = _build_snippet(document.content, query_tokens)
        document_scores.append(
            SearchHit(
                document=document,
                score=score,
                snippet=snippet,
            )
        )

    document_scores.sort(key=lambda item: (-item.score, item.document.repo_id, item.document.doc_id))
    return SearchResult(
        query=query,
        hits=document_scores[:top_k],
        backend='local',
        repo_count=repo_count,
        document_count=document_count,
    )


def _load_search_documents(target_dir: str, repo_id: str | None) -> list[KnowledgeDocument]:
    """按检索范围加载文档，单仓库问答时避免无意义地扫描整库 JSON。"""
    if repo_id:
        return load_repo_documents(repo_id=repo_id, target_dir=target_dir)
    return load_all_documents(target_dir=target_dir)



def _should_try_chroma(repo_id: str | None, document_count: int, repo_count: int) -> bool:
    """知识库规模较小时优先本地检索，减少 Chroma 初始化失败带来的额外等待。"""
    if repo_id and document_count <= LOCAL_ONLY_REPO_DOC_THRESHOLD:
        return False
    if repo_id is None and repo_count > 0 and document_count <= LOCAL_ONLY_GLOBAL_DOC_THRESHOLD:
        return False
    return True


def _score_document(
    query: str,
    query_tokens: list[str],
    query_intent: str,
    document: KnowledgeDocument,
    idf: dict[str, float],
) -> float:
    """根据查询词、文档类型与结构化 metadata 计算混合得分。"""
    title_tokens = _tokenize(document.title)
    content_tokens = _tokenize(document.content)
    metadata_tokens = _tokenize(_flatten_metadata(document.metadata))

    title_counts = Counter(title_tokens)
    content_counts = Counter(content_tokens)
    metadata_counts = Counter(metadata_tokens)

    score = 0.0
    for token in query_tokens:
        token_idf = idf.get(token, 0.0)
        score += title_counts[token] * (token_idf + 3.0)
        score += content_counts[token] * (token_idf + 1.0)
        score += metadata_counts[token] * (token_idf + 2.0)

    score += _score_metadata_fields(query_tokens, query_intent, document.metadata)
    score += _score_explicit_file_path_matches(query_tokens, document)
    score += _score_precise_query_matches(query, document)
    score += _score_intent_specific_precision(query_intent, query, document)
    score += _score_query_coverage(query_tokens, title_tokens, content_tokens, metadata_tokens)
    score *= _doc_type_multiplier(document.doc_type, query_intent)
    score *= _default_doc_type_multiplier(document.doc_type)
    return round(score, 4)


def _rerank_existing_hits(
    hits: list[SearchHit],
    query: str,
    query_tokens: list[str],
    query_intent: str,
) -> list[SearchHit]:
    """对向量检索命中结果再做一次结构化重排。"""
    reranked: list[SearchHit] = []
    for hit in hits:
        adjusted_score = hit.score
        adjusted_score += _score_metadata_fields(query_tokens, query_intent, hit.document.metadata)
        adjusted_score += _score_explicit_file_path_matches(query_tokens, hit.document)
        adjusted_score += _score_precise_query_matches(query, hit.document)
        adjusted_score += _score_intent_specific_precision(query_intent, query, hit.document)
        adjusted_score += _score_query_coverage(
            query_tokens,
            _tokenize(hit.document.title),
            _tokenize(hit.document.content),
            _tokenize(_flatten_metadata(hit.document.metadata)),
        )
        adjusted_score *= _doc_type_multiplier(hit.document.doc_type, query_intent)
        adjusted_score *= _default_doc_type_multiplier(hit.document.doc_type)
        reranked.append(
            SearchHit(
                document=hit.document,
                score=round(adjusted_score, 4),
                snippet=hit.snippet,
            )
        )

    reranked.sort(key=lambda item: (-item.score, item.document.repo_id, item.document.doc_id))
    return reranked


def _score_precise_query_matches(query: str, document: KnowledgeDocument) -> float:
    """为显式符号、路由和文件路径命中提供更强加权。"""
    exact_terms = _extract_precise_query_terms(query)
    if not exact_terms:
        return 0.0

    source_path = (document.source_path or '').lower()
    title = document.title.lower()
    metadata_values = {
        item.lower()
        for item in _collect_precise_document_values(document)
    }

    score = 0.0
    for term in exact_terms:
        if not term:
            continue
        if term == source_path or term in metadata_values:
            score += 6.0
            continue
        if source_path and term in source_path:
            score += 3.0
        if term in title:
            score += 2.4
        if any(term in value for value in metadata_values):
            score += 2.8
    return score


def _extract_precise_query_terms(query: str) -> list[str]:
    """提取查询里可直接用于精确定位的代码锚点。"""
    lowered = query.lower()
    terms = re.findall(r'[a-z0-9_./:-]+\.[a-z0-9_./:-]+', lowered)
    terms.extend(re.findall(r'/[a-z0-9_\-/{}/:]+', lowered))
    terms.extend(re.findall(r'[a-z_][a-z0-9_]{2,}', lowered))

    exact_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip('.:')
        if len(normalized) < 3:
            continue
        if not any(char in normalized for char in ('/', '.', '_')):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        exact_terms.append(normalized)
    return exact_terms


def _collect_precise_document_values(document: KnowledgeDocument) -> list[str]:
    """收集文档里适合精确匹配的结构化值。"""
    precise_fields = (
        'source_path',
        'symbol_name',
        'qualified_name',
        'handler_name',
        'handler_qualified_name',
        'route_path',
        'api_route_paths',
        'api_handler_names',
        'function_names',
        'class_names',
        'code_entity_names',
        'code_entity_refs',
        'code_entity_kinds',
        'code_symbol_names',
        'called_symbols',
        'code_relation_targets',
        'code_relation_sources',
        'code_relation_types',
        'module_relation_targets',
        'entrypoints',
        'config_related_paths',
    )
    values: list[str] = []
    if document.source_path:
        values.append(document.source_path)
    for field_name in precise_fields:
        values.extend(_metadata_values(document.metadata, field_name))
    return values


def _score_query_coverage(
    query_tokens: list[str],
    title_tokens: list[str],
    content_tokens: list[str],
    metadata_tokens: list[str],
) -> float:
    """奖励多个查询 token 在同一文档中被覆盖。"""
    if not query_tokens:
        return 0.0

    token_pool = set(title_tokens) | set(content_tokens) | set(metadata_tokens)
    covered = [token for token in query_tokens if token in token_pool]
    if not covered:
        return 0.0

    score = min(len(covered), 4) * 0.9
    if len(covered) >= min(len(query_tokens), 2):
        score += 1.2
    return score


def _score_intent_specific_precision(
    query_intent: str,
    query: str,
    document: KnowledgeDocument,
) -> float:
    """针对实现类与 API 类问题，按文档结构做更细的精确匹配加权。"""
    exact_terms = _extract_precise_query_terms(query)
    if not exact_terms:
        return 0.0

    if query_intent == 'implementation':
        field_weights = {
            'qualified_name': 5.0,
            'symbol_name': 4.4,
            'function_names': 3.8,
            'class_names': 3.6,
            'called_symbols': 2.8,
            'source_path': 2.6,
        }
        preferred_doc_types = {'function_summary': 1.5, 'class_summary': 1.2, 'key_file_summary': 0.8}
    elif query_intent == 'api':
        field_weights = {
            'route_path': 5.2,
            'handler_qualified_name': 4.6,
            'handler_name': 4.2,
            'api_route_paths': 4.0,
            'api_handler_names': 3.8,
            'called_symbols': 2.6,
            'source_path': 2.4,
        }
        preferred_doc_types = {'api_route_summary': 1.6, 'function_summary': 0.9, 'entrypoint_summary': 0.8}
    else:
        return 0.0

    score = 0.0
    for field_name, weight in field_weights.items():
        values = [value.lower() for value in _metadata_values(document.metadata, field_name)]
        if not values:
            continue
        for term in exact_terms:
            if any(term == value or term in value for value in values):
                score += weight
                break

    score += preferred_doc_types.get(document.doc_type, 0.0)
    return score


def _score_metadata_fields(
    query_tokens: list[str],
    query_intent: str,
    metadata: dict[str, str | int | float | bool | list[str]],
) -> float:
    """针对结构化字段做额外加权，提升精确检索与意图命中率。"""
    score = 0.0
    field_weights = dict(GENERIC_STRUCTURED_FIELDS)
    field_weights.update(INTENT_METADATA_WEIGHTS.get(query_intent, {}))

    for field_name, weight in field_weights.items():
        values = _metadata_values(metadata, field_name)
        if not values:
            continue
        lowered_values = [value.lower() for value in values]
        match_count = 0
        for token in query_tokens:
            if any(token == value or token in value for value in lowered_values):
                match_count += 1
        if match_count:
            score += min(match_count, 3) * weight

    return score


def _score_explicit_file_path_matches(query_tokens: list[str], document: KnowledgeDocument) -> float:
    """若查询中直接提到了文件名或路径，则给对应文档明显加权。"""
    if not document.source_path:
        return 0.0

    lowered_path = document.source_path.lower()
    file_name = lowered_path.rsplit('/', maxsplit=1)[-1]
    score = 0.0
    for token in query_tokens:
        if token == lowered_path or token == file_name:
            score += 4.0
        elif token in lowered_path:
            score += 1.5
    return score


def _default_doc_type_multiplier(doc_type: str) -> float:
    """保留一层基础文档类型偏好。"""
    if doc_type == 'repo_summary':
        return 1.15
    if doc_type == 'repo_fact':
        return 1.10
    if doc_type == 'readme_summary':
        return 1.12
    if doc_type == 'config_summary':
        return 1.16
    if doc_type == 'entrypoint_summary':
        return 1.16
    if doc_type == 'subproject_summary':
        return 1.10
    if doc_type == 'function_summary':
        return 1.18
    if doc_type == 'api_route_summary':
        return 1.20
    if doc_type == 'class_summary':
        return 1.12
    return 1.0


def _doc_type_multiplier(doc_type: str, query_intent: str) -> float:
    """根据查询意图对不同文档类型进行动态加权。"""
    return INTENT_DOC_TYPE_MULTIPLIERS.get(query_intent, {}).get(doc_type, 1.0)


def _infer_query_intent(query: str) -> str:
    """根据问题文本推断当前更接近哪一类检索意图。"""
    lowered = query.lower()
    if re.search(r'\b(get|post|put|delete|patch|options|head)\b\s+/[a-z0-9_\-/{}/:]+', lowered):
        return 'api'
    if re.search(r'/[a-z0-9_\-/{}/:]+', lowered) and any(token in lowered for token in ('接口', '路由', 'api', 'endpoint')):
        return 'api'
    for intent, keywords in QUERY_INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent
    return 'generic'


def _metadata_values(
    metadata: dict[str, str | int | float | bool | list[str]],
    field_name: str,
) -> list[str]:
    """把 metadata 中指定字段统一展开成字符串列表。"""
    value = metadata.get(field_name)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _build_idf(documents: list[KnowledgeDocument]) -> dict[str, float]:
    """计算一个轻量的逆文档频率表，提升区分度。"""
    token_document_count: Counter[str] = Counter()
    for document in documents:
        tokens = set(
            _tokenize(document.title)
            + _tokenize(document.content)
            + _tokenize(_flatten_metadata(document.metadata))
        )
        for token in tokens:
            token_document_count[token] += 1

    total_documents = max(len(documents), 1)
    idf: dict[str, float] = {}
    for token, count in token_document_count.items():
        idf[token] = math.log((1 + total_documents) / (1 + count)) + 1.0

    return idf


def _build_snippet(content: str, query_tokens: list[str], max_chars: int = 180) -> str:
    """根据查询词从文档内容中截取一个适合展示的命中片段。"""
    normalized_content = content.replace('\n', ' ')
    lowered = normalized_content.lower()

    hit_index = -1
    for token in query_tokens:
        hit_index = lowered.find(token.lower())
        if hit_index >= 0:
            break

    if hit_index < 0:
        return normalized_content[:max_chars] + ('...' if len(normalized_content) > max_chars else '')

    start = max(0, hit_index - 50)
    end = min(len(normalized_content), hit_index + max_chars - 50)
    snippet = normalized_content[start:end].strip()
    if start > 0:
        snippet = '...' + snippet
    if end < len(normalized_content):
        snippet = snippet + '...'
    return snippet


def _tokenize(text: str) -> list[str]:
    """对中英文混合文本做一个足够简单的 token 切分。"""
    lowered = text.lower()
    latin_tokens = re.findall(r'[a-z0-9_\-\.]+', lowered)
    route_tokens = re.findall(r'/[a-z0-9_\-/{}/:]+', lowered)
    chinese_sequences = re.findall(r'[一-鿿]{2,}', lowered)

    tokens = list(latin_tokens) + list(route_tokens)
    for sequence in chinese_sequences:
        tokens.append(sequence)
        tokens.extend(
            sequence[index:index + 2]
            for index in range(len(sequence) - 1)
        )

    deduplicated: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if token in seen:
            continue
        seen.add(token)
        deduplicated.append(token)

    return deduplicated
def _flatten_metadata(metadata: dict[str, str | int | float | bool | list[str]]) -> str:
    """把元数据展开成字符串，便于一起参与检索。"""
    values: list[str] = []
    for value in metadata.values():
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return ' '.join(values)
