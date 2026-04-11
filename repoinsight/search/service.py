import math
import re
from collections import Counter

from repoinsight.models.rag_model import KnowledgeDocument, SearchHit, SearchResult
from repoinsight.storage.local_knowledge_store import DEFAULT_KNOWLEDGE_DIR, load_all_documents


# 过滤掉过短 token，避免大量单字符噪音影响排序。
MIN_TOKEN_LENGTH = 2


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    target_dir: str = DEFAULT_KNOWLEDGE_DIR,
) -> SearchResult:
    """在本地知识库中执行一个轻量级检索。"""
    documents = load_all_documents(target_dir=target_dir)
    query_tokens = _tokenize(query)

    if not documents or not query_tokens:
        return SearchResult(
            query=query,
            hits=[],
            repo_count=len({item.repo_id for item in documents}),
            document_count=len(documents),
        )

    document_scores: list[SearchHit] = []
    idf = _build_idf(documents)
    for document in documents:
        score = _score_document(query_tokens, document, idf)
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
        repo_count=len({item.repo_id for item in documents}),
        document_count=len(documents),
    )


def _score_document(
    query_tokens: list[str],
    document: KnowledgeDocument,
    idf: dict[str, float],
) -> float:
    """根据查询词与文档词频做一个简单的混合打分。"""
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

    # 仓库摘要通常更适合做第一跳召回，因此略微提高其权重。
    if document.doc_type == 'repo_summary':
        score *= 1.2
    elif document.doc_type == 'repo_fact':
        score *= 1.1

    return round(score, 4)


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
    tokens = re.findall(r'[a-z0-9_\-\.]+|[\u4e00-\u9fff]{2,}', lowered)
    return [token for token in tokens if len(token) >= MIN_TOKEN_LENGTH]


def _flatten_metadata(metadata: dict[str, str | int | float | bool | list[str]]) -> str:
    """把元数据展开成字符串，便于一起参与检索。"""
    values: list[str] = []
    for value in metadata.values():
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return ' '.join(values)
