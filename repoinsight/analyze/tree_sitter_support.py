from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class TreeSitterDocument:
    """表示一次成功的 tree-sitter 解析结果。"""

    language_name: str
    source_bytes: bytes
    root_node: object


def parse_source_with_tree_sitter(file_path: str, content: str) -> TreeSitterDocument | None:
    """根据文件扩展名尝试用 tree-sitter 解析源码。"""
    language_name = _infer_language_name(file_path)
    if language_name is None:
        return None

    parser = _get_parser(language_name)
    if parser is None:
        return None

    source_bytes = content.encode('utf-8')
    tree = parser.parse(source_bytes)
    return TreeSitterDocument(
        language_name=language_name,
        source_bytes=source_bytes,
        root_node=tree.root_node,
    )


def get_node_text(node: object, source_bytes: bytes) -> str:
    """读取节点在原始源码中的文本。"""
    start_byte = getattr(node, 'start_byte')
    end_byte = getattr(node, 'end_byte')
    return source_bytes[start_byte:end_byte].decode('utf-8', errors='ignore')


@lru_cache(maxsize=8)
def _get_parser(language_name: str):
    """按语言懒加载 parser；未安装对应语法包时返回 None。"""
    try:
        from tree_sitter import Language, Parser
    except ModuleNotFoundError:
        return None

    try:
        if language_name == 'python':
            import tree_sitter_python

            return Parser(Language(tree_sitter_python.language()))
        if language_name == 'javascript':
            import tree_sitter_javascript

            return Parser(Language(tree_sitter_javascript.language()))
        if language_name == 'typescript':
            import tree_sitter_typescript

            return Parser(Language(tree_sitter_typescript.language_typescript()))
        if language_name == 'tsx':
            import tree_sitter_typescript

            return Parser(Language(tree_sitter_typescript.language_tsx()))
    except ModuleNotFoundError:
        return None
    return None


def _infer_language_name(file_path: str) -> str | None:
    """把文件路径映射到当前支持的 tree-sitter 语言。"""
    lower_path = file_path.lower()
    if lower_path.endswith('.py'):
        return 'python'
    if lower_path.endswith('.js') or lower_path.endswith('.jsx'):
        return 'javascript'
    if lower_path.endswith('.ts'):
        return 'typescript'
    if lower_path.endswith('.tsx'):
        return 'tsx'
    return None
