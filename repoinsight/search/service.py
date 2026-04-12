import math
import re
from collections import Counter

from repoinsight.models.rag_model import KnowledgeDocument, SearchHit, SearchResult
from repoinsight.storage.chroma_store import search_documents_in_chroma
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR, load_all_documents


# 过滤掉过短 token，避免大量单字符噪音影响排序。
MIN_TOKEN_LENGTH = 2

# 不同查询意图对应的关键词，用于做轻量级检索路由。
QUERY_INTENT_KEYWORDS = {
    'startup': ('启动', '运行', 'run', 'start', 'dev', 'server', '怎么跑', '如何启动'),
    'entrypoint': ('入口', 'main', 'entrypoint', '从哪里开始', '启动文件', 'server.py', 'app.py'),
    'env': ('环境变量', 'env', '配置项', 'token', 'secret', 'api key', 'apikey'),
    'config': ('配置', '构建', '打包', '脚本', '依赖', 'package', 'install', '编译'),
    'tech': ('技术栈', '框架', '语言', 'runtime', 'tech stack', '用了什么', '使用了什么'),
    'architecture': ('架构', '模块', '依赖关系', '调用链', '子项目', 'monorepo', 'workspace', 'service', '包结构'),
}

# 查询意图与文档类型的加权关系。
INTENT_DOC_TYPE_MULTIPLIERS = {
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
}

# 不同查询意图关注的结构化 metadata 字段。
INTENT_METADATA_WEIGHTS = {
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
        'code_symbols': 2.6,
        'module_relations': 2.6,
        'source_path': 2.2,
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
}


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
    repo_id: str | None = None,
) -> SearchResult:
    """优先执行向量检索；若未启用则回退到本地轻量检索。"""
    documents = load_all_documents(target_dir=target_dir)
    if repo_id:
        documents = [item for item in documents if item.repo_id == repo_id]
    query_tokens = _tokenize(query)
    query_intent = _infer_query_intent(query)

    if not documents or not query_tokens:
        return SearchResult(
            query=query,
            hits=[],
            backend='local',
            repo_count=len({item.repo_id for item in documents}),
            document_count=len(documents),
        )

    if repo_id is None:
        try:
            vector_hits = search_documents_in_chroma(query=query, top_k=top_k)
            if vector_hits:
                reranked_hits = _rerank_existing_hits(vector_hits, query_tokens, query_intent)
                return SearchResult(
                    query=query,
                    hits=reranked_hits[:top_k],
                    backend='chroma',
                    repo_count=len({item.repo_id for item in documents}),
                    document_count=len(documents),
                )
        except Exception:
            # 若未安装 Chroma 或 embedding 模型，自动退回到轻量本地检索。
            pass

    document_scores: list[SearchHit] = []
    idf = _build_idf(documents)
    for document in documents:
        score = _score_document(query_tokens, query_intent, document, idf)
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
        repo_count=len({item.repo_id for item in documents}),
        document_count=len(documents),
    )


def _score_document(
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
    score *= _doc_type_multiplier(document.doc_type, query_intent)
    score *= _default_doc_type_multiplier(document.doc_type)
    return round(score, 4)


def _rerank_existing_hits(
    hits: list[SearchHit],
    query_tokens: list[str],
    query_intent: str,
) -> list[SearchHit]:
    """对向量检索命中结果再做一次结构化重排。"""
    reranked: list[SearchHit] = []
    for hit in hits:
        adjusted_score = hit.score
        adjusted_score += _score_metadata_fields(query_tokens, query_intent, hit.document.metadata)
        adjusted_score += _score_explicit_file_path_matches(query_tokens, hit.document)
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
    if doc_type == 'config_summary':
        return 1.16
    if doc_type == 'entrypoint_summary':
        return 1.16
    if doc_type == 'subproject_summary':
        return 1.10
    return 1.0


def _doc_type_multiplier(doc_type: str, query_intent: str) -> float:
    """根据查询意图对不同文档类型进行动态加权。"""
    return INTENT_DOC_TYPE_MULTIPLIERS.get(query_intent, {}).get(doc_type, 1.0)


def _infer_query_intent(query: str) -> str:
    """从问题文本中推断当前更接近哪一类检索意图。"""
    lowered = query.lower()
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
    chinese_sequences = re.findall(r'[一-鿿]{2,}', lowered)

    tokens = list(latin_tokens)
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
