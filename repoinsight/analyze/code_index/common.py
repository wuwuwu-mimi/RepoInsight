from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from repoinsight.models.analysis_model import ApiRouteSummary, ClassSummary, CodeSymbol, FunctionSummary, KeyFileContent, ModuleRelation

@dataclass
class _CodeIndexAccumulator:
    """收集中间结果，便于不同语言提取器复用。"""

    code_symbols: list[CodeSymbol] = field(default_factory=list)
    module_relations: list[ModuleRelation] = field(default_factory=list)
    function_summaries: list[FunctionSummary] = field(default_factory=list)
    class_summaries: list[ClassSummary] = field(default_factory=list)
    api_route_summaries: list[ApiRouteSummary] = field(default_factory=list)

@dataclass
class _PythonRouteMatch:
    """表示一次 Python 路由装饰器命中。"""

    route_path: str
    http_methods: list[str]
    framework: str | None

@dataclass
class _JsRouteMatch:
    """表示一次 JS/TS 路由注册命中。"""

    http_methods: list[str]
    route_path: str
    handler_expression: str
    line_number: int


def _python_expr_to_text(node: ast.AST) -> str:
    """尽量把 AST 表达式转换成可读文本。"""
    try:
        return ast.unparse(node).strip()
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = _python_expr_to_text(node.value)
            return f'{left}.{node.attr}' if left else node.attr
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return ''

def _build_api_route_summary_text(
    route_path: str,
    http_methods: list[str],
    handler_name: str,
    framework: str | None,
    called_symbols: list[str],
) -> str:
    """生成接口/路由级摘要文本。"""
    methods_text = '/'.join(http_methods) if http_methods else 'HTTP'
    parts = [f'接口 {methods_text} {route_path} 由 {handler_name} 处理']
    if framework:
        parts.append(f'框架线索为 {framework}')
    if called_symbols:
        parts.append(f"处理过程中会调用 {', '.join(called_symbols[:4])}")
    else:
        parts.append('当前尚未提取到明显的下游调用线索')
    return '，'.join(parts) + '。'

def _parse_python_route_decorator(node: ast.AST) -> _PythonRouteMatch | None:
    """解析 Python Web 框架常见的路由装饰器。"""
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None

    attr_name = node.func.attr.lower()
    if attr_name not in {'get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'route', 'api_route', 'websocket'}:
        return None

    route_path = _extract_python_route_path(node)
    if not route_path:
        return None

    return _PythonRouteMatch(
        route_path=route_path,
        http_methods=_extract_python_http_methods(node, attr_name),
        framework=_infer_python_route_framework(_python_expr_to_text(node.func), attr_name),
    )

def _extract_python_route_path(node: ast.Call) -> str | None:
    """提取 Python 路由装饰器中的路径。"""
    if node.args:
        literal = _extract_string_literal(node.args[0])
        if literal:
            return literal
    for keyword in node.keywords:
        if keyword.arg in {'path', 'rule'}:
            literal = _extract_string_literal(keyword.value)
            if literal:
                return literal
    return None

def _extract_python_http_methods(node: ast.Call, attr_name: str) -> list[str]:
    """提取 Python 路由装饰器中的 HTTP 方法。"""
    if attr_name == 'websocket':
        return ['WEBSOCKET']
    if attr_name in {'get', 'post', 'put', 'delete', 'patch', 'options', 'head'}:
        return [attr_name.upper()]

    for keyword in node.keywords:
        if keyword.arg != 'methods':
            continue
        methods = _extract_python_string_list(keyword.value)
        if methods:
            return [item.upper() for item in methods]
    return ['GET']

def _infer_python_route_framework(func_text: str, attr_name: str) -> str | None:
    """根据装饰器文本推断 Python 路由框架线索。"""
    lowered = func_text.lower()
    if attr_name in {'api_route', 'websocket'}:
        return 'fastapi'
    if 'router' in lowered:
        return 'fastapi'
    if attr_name == 'route':
        return 'flask'
    return None

def _extract_string_literal(node: ast.AST) -> str | None:
    """尽量提取 AST 中的字符串字面量。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

def _extract_python_string_list(node: ast.AST) -> list[str]:
    """提取字符串列表字面量。"""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    values: list[str] = []
    for item in node.elts:
        literal = _extract_string_literal(item)
        if literal:
            values.append(literal)
    return values

def _split_js_arguments(text: str) -> list[str]:
    """按顶层逗号拆分 JS 参数，尽量避开括号和字符串内部。"""
    result: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False

    for char in text:
        if quote is not None:
            current.append(char)
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == quote:
                quote = None
            continue

        if char in {'"', "'", '`'}:
            quote = char
            current.append(char)
            continue

        if char in '([{':
            depth += 1
            current.append(char)
            continue
        if char in ')]}':
            depth = max(depth - 1, 0)
            current.append(char)
            continue

        if char == ',' and depth == 0:
            item = ''.join(current).strip()
            if item:
                result.append(item)
            current = []
            continue

        current.append(char)

    tail = ''.join(current).strip()
    if tail:
        result.append(tail)
    return result

def _normalize_javascript_signature(signature_text: str, name: str) -> str:
    """清洗 JS/TS 原始声明片段，便于在报告中展示。"""
    cleaned = ' '.join(signature_text.replace('{', '').split())
    if cleaned:
        return cleaned.rstrip()
    return name

def _split_parameter_text(text: str) -> list[str]:
    """把形参文本拆成便于检索的参数列表。"""
    if not text.strip():
        return []
    parameters: list[str] = []
    for raw_item in text.split(','):
        item = raw_item.strip()
        if not item:
            continue
        item = re.split(r'[:=]', item, maxsplit=1)[0].strip()
        if item:
            parameters.append(item)
    return parameters

def _line_number_from_offset(text: str, offset: int) -> int:
    """把字符偏移量转换为 1-based 行号。"""
    return text.count('\n', 0, offset) + 1

def _deduplicate_keep_order(items: list[str]) -> list[str]:
    """稳定去重并过滤空字符串。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result

def _build_python_function_summary(
    name: str,
    owner_class: str | None,
    decorators: list[str],
    called_symbols: list[str],
    return_signals: list[str],
    is_async: bool,
) -> str:
    """生成适合 RAG 直接使用的 Python 函数摘要。"""
    subject = f'方法 {owner_class}.{name}' if owner_class else f'函数 {name}'
    parts = [f'{subject}']
    if is_async:
        parts.append('以异步方式执行')

    role = _infer_python_function_role(name, decorators, called_symbols)
    if role:
        parts.append(role)

    if called_symbols:
        parts.append(f"会调用 {', '.join(called_symbols[:4])}")
    if return_signals:
        parts.append(f"并返回 {', '.join(return_signals[:3])}")
    if len(parts) == 1:
        parts.append('当前已识别到声明信息，但尚未提炼出更明确的职责线索')
    return '，'.join(parts) + '。'

def _infer_python_function_role(name: str, decorators: list[str], called_symbols: list[str]) -> str:
    """根据命名和调用线索推断 Python 函数职责。"""
    lowered_name = name.lower()
    lowered_calls = [item.lower() for item in called_symbols]
    lowered_decorators = [item.lower() for item in decorators]

    if lowered_name in {'main', 'run', 'start', 'bootstrap'}:
        return '可能承担启动或入口职责'
    if any(item.endswith(('.get', '.post', '.put', '.delete')) or item in {'app.get', 'app.post'} for item in lowered_decorators):
        return '可能负责处理某个 HTTP 路由'
    if any('include_router' in item or 'add_api_route' in item for item in lowered_calls):
        return '负责注册路由或把接口挂载到应用上'
    if any(item in {'fastapi', 'flask', 'typer.typer'} for item in lowered_calls):
        return '负责创建应用或命令行对象'
    if any('open' in item or 'load' in item or 'read' in item for item in lowered_calls):
        return '包含读取或加载数据的逻辑'
    if any('save' in item or 'write' in item or 'commit' in item for item in lowered_calls):
        return '包含写入、保存或提交逻辑'
    if lowered_name.startswith(('get_', 'list_', 'fetch_')):
        return '更偏向查询或读取逻辑'
    if lowered_name.startswith(('create_', 'build_', 'make_')):
        return '更偏向构建或创建对象'
    if lowered_name.startswith(('handle_', 'process_', 'run_')):
        return '更偏向业务处理流程'
    return ''

def _build_python_class_summary(name: str, methods: list[str]) -> str:
    """生成 Python 类摘要。"""
    if methods:
        return f'类 {name} 定义了方法 {", ".join(methods[:5])}，可能用于封装相关状态与行为。'
    return f'类 {name} 已被识别，当前尚未提炼出明确的方法协作线索。'

def _build_javascript_function_summary(
    name: str,
    owner_class: str | None,
    called_symbols: list[str],
    is_async: bool,
) -> str:
    """生成 JS/TS 函数摘要。"""
    subject = f'方法 {owner_class}.{name}' if owner_class else f'函数 {name}'
    parts = [subject]
    if is_async:
        parts.append('包含异步处理逻辑')

    lowered_name = name.lower()
    lowered_calls = [item.lower() for item in called_symbols]
    if lowered_name in {'main', 'bootstrap', 'start', 'setup'}:
        parts.append('可能承担前端挂载或启动职责')
    elif any(item in {'createroot', 'reactdom.createroot', 'createapp'} for item in lowered_calls):
        parts.append('负责应用挂载或初始化')
    elif lowered_name.startswith(('handle', 'on', 'submit')):
        parts.append('更偏向事件处理逻辑')
    elif lowered_name.startswith(('fetch', 'load', 'query')):
        parts.append('更偏向数据请求或读取逻辑')
    elif lowered_name.startswith(('create', 'build')):
        parts.append('更偏向构建或创建对象')

    if called_symbols:
        parts.append(f"会调用 {', '.join(called_symbols[:4])}")
    if len(parts) == 1:
        parts.append('当前已识别到声明信息，但尚未提炼出更明确的职责线索')
    return '，'.join(parts) + '。'

def _build_javascript_class_summary(name: str, methods: list[str]) -> str:
    """生成 JS/TS 类摘要。"""
    if methods:
        return f'类 {name} 定义了方法 {", ".join(methods[:5])}，可能用于组织相关业务或组件行为。'
    return f'类 {name} 已被识别，当前尚未提炼出明确的方法协作线索。'

def _extract_function_body(text: str, match: re.Match[str]) -> tuple[int, str]:
    """根据函数声明提取函数体和结束行号。"""
    block_start = text.find('{', match.end() - 1)
    if block_start == -1:
        line_start = _line_number_from_offset(text, match.start())
        line_end = _line_number_from_offset(text, match.end())
        return line_end, text[match.start():match.end()]
    return _extract_block_from_match(text, match)

def _extract_block_from_match(text: str, match: re.Match[str]) -> tuple[int, str]:
    """从匹配位置开始找到配对的大括号块。"""
    block_start = text.find('{', match.end() - 1)
    line_start = _line_number_from_offset(text, match.start())
    if block_start == -1:
        return line_start, ''

    depth = 0
    for index in range(block_start, len(text)):
        char = text[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                line_end = _line_number_from_offset(text, index)
                return line_end, text[block_start + 1:index]
    return line_start, text[block_start + 1:]

