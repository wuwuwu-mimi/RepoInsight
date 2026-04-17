from __future__ import annotations

import re

from repoinsight.analyze.code_semantics import normalize_javascript_call_target
from repoinsight.analyze.code_index.common import (
    _CodeIndexAccumulator,
    _build_api_route_summary_text,
    _build_javascript_class_summary,
    _build_javascript_function_summary,
    _deduplicate_keep_order,
    _extract_block_from_match,
    _extract_function_body,
    _JsRouteMatch,
    _line_number_from_offset,
    _normalize_javascript_signature,
    _split_js_arguments,
    _split_parameter_text,
)
from repoinsight.analyze.tree_sitter_support import get_node_text, parse_source_with_tree_sitter
from repoinsight.models.analysis_model import ApiRouteSummary, ClassSummary, CodeSymbol, FunctionSummary, KeyFileContent, ModuleRelation

def _extract_javascript_index(file_content: KeyFileContent, accumulator: _CodeIndexAccumulator) -> None:
    """优先使用 tree-sitter 提取 JS/TS 结构，失败时回退到轻量规则。"""
    if _extract_javascript_index_with_tree_sitter(file_content, accumulator):
        return
    _extract_javascript_index_with_regex(file_content, accumulator)

def _extract_javascript_index_with_regex(file_content: KeyFileContent, accumulator: _CodeIndexAccumulator) -> None:
    """使用轻量规则提取 JS/TS 的函数、类和依赖。"""
    text = file_content.content
    lower_path = file_content.path.lower()
    language_scope = 'nodejs'

    for match in re.finditer(r'import\s+[^;\n]*?from\s+[\'\"]([^\'\"]+)[\'\"]', text):
        accumulator.module_relations.append(
            ModuleRelation(
                source_path=file_content.path,
                target=match.group(1),
                relation_type='import',
                line_number=_line_number_from_offset(text, match.start()),
            )
        )
    for match in re.finditer(r'require\(\s*[\'\"]([^\'\"]+)[\'\"]\s*\)', text):
        accumulator.module_relations.append(
            ModuleRelation(
                source_path=file_content.path,
                target=match.group(1),
                relation_type='require',
                line_number=_line_number_from_offset(text, match.start()),
            )
        )

    class_ranges: list[tuple[int, int, str]] = []
    class_pattern = re.compile(
        r'(?:export\s+default\s+|export\s+)?class\s+([A-Za-z_$][\w$]*)\s*(?:extends\s+([A-Za-z0-9_.$]+))?\s*\{'
    )
    for match in class_pattern.finditer(text):
        class_name = match.group(1)
        extends_name = match.group(2)
        line_start = _line_number_from_offset(text, match.start())
        line_end, body = _extract_block_from_match(text, match)
        class_ranges.append((match.start(), match.end(), class_name))
        methods = _extract_js_class_methods(body, base_line=line_start)
        accumulator.code_symbols.append(
            CodeSymbol(
                name=class_name,
                symbol_type='class',
                source_path=file_content.path,
                line_number=line_start,
            )
        )
        accumulator.class_summaries.append(
            ClassSummary(
                name=class_name,
                qualified_name=class_name,
                source_path=file_content.path,
                language_scope=language_scope,
                line_start=line_start,
                line_end=line_end,
                bases=[extends_name] if extends_name else [],
                decorators=[],
                methods=[item['name'] for item in methods[:12]],
                summary=_build_javascript_class_summary(class_name, [item['name'] for item in methods]),
            )
        )
        for method in methods:
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=method['name'],
                    symbol_type='method',
                    source_path=file_content.path,
                    line_number=method['line_start'],
                )
            )
            accumulator.function_summaries.append(
                FunctionSummary(
                    name=method['name'],
                    qualified_name=f"{class_name}.{method['name']}",
                    source_path=file_content.path,
                    language_scope=language_scope,
                    line_start=method['line_start'],
                    line_end=method['line_end'],
                    signature=method['signature'],
                    owner_class=class_name,
                    is_async=method['is_async'],
                    decorators=[],
                    parameters=method['parameters'],
                    called_symbols=method['called_symbols'],
                    return_signals=[],
                    summary=_build_javascript_function_summary(
                        name=method['name'],
                        owner_class=class_name,
                        called_symbols=method['called_symbols'],
                        is_async=method['is_async'],
                    ),
                )
            )

    function_patterns = [
        re.compile(r'(?:export\s+default\s+|export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{'),
        re.compile(r'(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>'),
        re.compile(r'(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?function\s*\(([^)]*)\)'),
    ]

    seen_function_starts: set[int] = set()
    for pattern in function_patterns:
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end, _ in class_ranges):
                continue
            if match.start() in seen_function_starts:
                continue
            seen_function_starts.add(match.start())
            name = match.group(1)
            parameters = _split_parameter_text(match.group(2))
            line_start = _line_number_from_offset(text, match.start())
            line_end, body = _extract_function_body(text, match)
            signature = _normalize_javascript_signature(match.group(0), name)
            is_async = 'async' in match.group(0)
            called_symbols = _extract_javascript_called_symbols(body)
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=name,
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=line_start,
                )
            )
            accumulator.function_summaries.append(
                FunctionSummary(
                    name=name,
                    qualified_name=name,
                    source_path=file_content.path,
                    language_scope=language_scope,
                    line_start=line_start,
                    line_end=line_end,
                    signature=signature,
                    owner_class=None,
                    is_async=is_async,
                    decorators=[],
                    parameters=parameters,
                    called_symbols=called_symbols,
                    return_signals=[],
                    summary=_build_javascript_function_summary(
                        name=name,
                        owner_class=None,
                        called_symbols=called_symbols,
                        is_async=is_async,
                    ),
                )
            )

    if lower_path.endswith(('.ts', '.tsx')):
        for match in re.finditer(r'interface\s+([A-Za-z_$][\w$]*)', text):
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    symbol_type='interface',
                    source_path=file_content.path,
                    line_number=_line_number_from_offset(text, match.start()),
                )
            )

    _extract_javascript_routes(
        file_content=file_content,
        accumulator=accumulator,
        language_scope=language_scope,
    )

def _extract_javascript_index_with_tree_sitter(
    file_content: KeyFileContent,
    accumulator: _CodeIndexAccumulator,
) -> bool:
    """使用 tree-sitter 提取 JS/TS 的导入、函数、类和路由信息。"""
    document = parse_source_with_tree_sitter(file_content.path, file_content.content)
    if document is None or document.language_name not in {'javascript', 'typescript', 'tsx'}:
        return False

    language_scope = _get_javascript_language_scope(file_content.path)
    for node in _iter_javascript_top_level_nodes(document.root_node):
        if node.type == 'import_statement':
            source_node = node.child_by_field_name('source')
            source_text = _extract_tree_sitter_string(source_node, document.source_bytes)
            if source_text:
                accumulator.module_relations.append(
                    ModuleRelation(
                        source_path=file_content.path,
                        target=source_text,
                        relation_type='import',
                        line_number=node.start_point.row + 1,
                    )
                )
            continue

        if node.type == 'class_declaration':
            _collect_javascript_tree_sitter_class(
                node=node,
                file_content=file_content,
                accumulator=accumulator,
                source_bytes=document.source_bytes,
                language_scope=language_scope,
            )
            continue

        if node.type == 'function_declaration':
            _collect_javascript_tree_sitter_function(
                node=node,
                function_node=node,
                function_name=get_node_text(node.child_by_field_name('name'), document.source_bytes),
                owner_class=None,
                source_path=file_content.path,
                accumulator=accumulator,
                source_bytes=document.source_bytes,
                language_scope=language_scope,
            )
            continue

        if node.type == 'lexical_declaration':
            for child in node.named_children:
                if child.type != 'variable_declarator':
                    continue
                value_node = child.child_by_field_name('value')
                if value_node is None or value_node.type not in {'arrow_function', 'function_expression'}:
                    continue
                name_node = child.child_by_field_name('name')
                function_name = get_node_text(name_node, document.source_bytes).strip() if name_node else ''
                if not function_name:
                    continue
                _collect_javascript_tree_sitter_function(
                    node=child,
                    function_node=value_node,
                    function_name=function_name,
                    owner_class=None,
                    source_path=file_content.path,
                    accumulator=accumulator,
                    source_bytes=document.source_bytes,
                    language_scope=language_scope,
                )
            continue

        if node.type == 'interface_declaration':
            name_node = node.child_by_field_name('name')
            if name_node is None:
                continue
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=get_node_text(name_node, document.source_bytes),
                    symbol_type='interface',
                    source_path=file_content.path,
                    line_number=node.start_point.row + 1,
                )
            )

    for node in _iter_tree_sitter_named_nodes(document.root_node):
        if node.type != 'call_expression':
            continue
        function_node = node.child_by_field_name('function')
        if function_node is None:
            continue

        callee_text = normalize_javascript_call_target(get_node_text(function_node, document.source_bytes))
        if callee_text == 'require':
            arguments_node = node.child_by_field_name('arguments')
            if arguments_node is None or not arguments_node.named_children:
                continue
            source_text = _extract_tree_sitter_string(arguments_node.named_children[0], document.source_bytes)
            if not source_text:
                continue
            accumulator.module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=source_text,
                    relation_type='require',
                    line_number=node.start_point.row + 1,
                )
            )

    _collect_javascript_tree_sitter_routes(
        file_content=file_content,
        accumulator=accumulator,
        source_bytes=document.source_bytes,
        language_scope=language_scope,
        root_node=document.root_node,
    )
    return True

def _collect_javascript_tree_sitter_class(
    node,
    file_content: KeyFileContent,
    accumulator: _CodeIndexAccumulator,
    source_bytes: bytes,
    language_scope: str,
) -> None:
    """提取 class 声明及其方法。"""
    name_node = node.child_by_field_name('name')
    body_node = node.child_by_field_name('body')
    if name_node is None or body_node is None:
        return

    class_name = get_node_text(name_node, source_bytes).strip()
    method_names: list[str] = []
    for child in body_node.named_children:
        if child.type != 'method_definition':
            continue
        method_name_node = child.child_by_field_name('name')
        if method_name_node is None:
            continue
        method_name = get_node_text(method_name_node, source_bytes).strip()
        if not method_name:
            continue
        method_names.append(method_name)
        _collect_javascript_tree_sitter_function(
            node=child,
            function_node=child,
            function_name=method_name,
            owner_class=class_name,
            source_path=file_content.path,
            accumulator=accumulator,
            source_bytes=source_bytes,
            language_scope=language_scope,
        )

    accumulator.code_symbols.append(
        CodeSymbol(
            name=class_name,
            symbol_type='class',
            source_path=file_content.path,
            line_number=node.start_point.row + 1,
        )
    )
    accumulator.class_summaries.append(
        ClassSummary(
            name=class_name,
            qualified_name=class_name,
            source_path=file_content.path,
            language_scope=language_scope,
            line_start=node.start_point.row + 1,
            line_end=node.end_point.row + 1,
            bases=_get_javascript_class_bases(node=node, source_bytes=source_bytes),
            decorators=[],
            methods=method_names[:12],
            summary=_build_javascript_class_summary(class_name, method_names),
        )
    )

def _collect_javascript_tree_sitter_function(
    node,
    function_node,
    function_name: str,
    owner_class: str | None,
    source_path: str,
    accumulator: _CodeIndexAccumulator,
    source_bytes: bytes,
    language_scope: str,
) -> None:
    """提取单个 JS/TS 函数或方法。"""
    body_node = function_node.child_by_field_name('body')
    if body_node is None:
        return

    parameters_node = function_node.child_by_field_name('parameters')
    signature_source = get_node_text(node, source_bytes).strip()
    signature_text = signature_source.split('{', maxsplit=1)[0].strip() if '{' in signature_source else signature_source
    called_symbols = _collect_javascript_call_targets_from_node(body_node, source_bytes)
    qualified_name = f'{owner_class}.{function_name}' if owner_class else function_name

    accumulator.code_symbols.append(
        CodeSymbol(
            name=function_name,
            symbol_type='method' if owner_class else 'function',
            source_path=source_path,
            line_number=node.start_point.row + 1,
        )
    )
    accumulator.function_summaries.append(
        FunctionSummary(
            name=function_name,
            qualified_name=qualified_name,
            source_path=source_path,
            language_scope=language_scope,
            line_start=node.start_point.row + 1,
            line_end=node.end_point.row + 1,
            signature=_normalize_javascript_signature(signature_text, function_name),
            owner_class=owner_class,
            is_async=_is_javascript_async_node(function_node, source_bytes),
            decorators=[],
            parameters=_extract_tree_sitter_parameters(parameters_node, source_bytes),
            called_symbols=called_symbols,
            return_signals=[],
            summary=_build_javascript_function_summary(
                name=function_name,
                owner_class=owner_class,
                called_symbols=called_symbols,
                is_async=_is_javascript_async_node(function_node, source_bytes),
            ),
        )
    )

def _collect_javascript_tree_sitter_routes(
    file_content: KeyFileContent,
    accumulator: _CodeIndexAccumulator,
    source_bytes: bytes,
    language_scope: str,
    root_node,
) -> None:
    """提取 Express 风格的 app/router 路由注册。"""
    function_lookup = _build_function_lookup(accumulator.function_summaries, file_content.path)
    for node in _iter_tree_sitter_named_nodes(root_node):
        if node.type != 'call_expression':
            continue

        function_node = node.child_by_field_name('function')
        arguments_node = node.child_by_field_name('arguments')
        if function_node is None or arguments_node is None or function_node.type != 'member_expression':
            continue

        object_node = function_node.child_by_field_name('object')
        property_node = function_node.child_by_field_name('property')
        if object_node is None or property_node is None:
            continue

        object_name = get_node_text(object_node, source_bytes).replace('?.', '.').strip().split('.')[-1]
        method_name = get_node_text(property_node, source_bytes).strip().lower()
        if object_name not in {'app', 'router'}:
            continue
        if method_name not in {'get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'all'}:
            continue

        arguments = list(arguments_node.named_children)
        if len(arguments) < 2:
            continue

        route_path = _extract_tree_sitter_string(arguments[0], source_bytes)
        if not route_path:
            continue

        handler_expression = get_node_text(arguments[-1], source_bytes).strip()
        handler_name, handler_qualified_name, called_symbols = _resolve_js_route_handler(
            handler_expression=handler_expression,
            line_number=node.start_point.row + 1,
            function_lookup=function_lookup,
        )
        http_methods = (
            ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']
            if method_name == 'all'
            else [method_name.upper()]
        )
        accumulator.code_symbols.append(
            CodeSymbol(
                name=f"{'/'.join(http_methods)} {route_path}",
                symbol_type='route',
                source_path=file_content.path,
                line_number=node.start_point.row + 1,
            )
        )
        accumulator.api_route_summaries.append(
            ApiRouteSummary(
                route_path=route_path,
                http_methods=http_methods,
                source_path=file_content.path,
                language_scope=language_scope,
                framework='express',
                handler_name=handler_name,
                handler_qualified_name=handler_qualified_name,
                owner_class=None,
                line_number=node.start_point.row + 1,
                decorators=[handler_expression],
                called_symbols=called_symbols,
                summary=_build_api_route_summary_text(
                    route_path=route_path,
                    http_methods=http_methods,
                    handler_name=handler_qualified_name,
                    framework='express',
                    called_symbols=called_symbols,
                ),
            )
        )

def _iter_javascript_top_level_nodes(root_node) -> list[object]:
    """返回顶层定义节点，并展开一层 export 包裹。"""
    result: list[object] = []
    for child in getattr(root_node, 'named_children', []):
        if child.type == 'export_statement':
            result.extend(list(getattr(child, 'named_children', [])))
            continue
        result.append(child)
    return result

def _iter_tree_sitter_named_nodes(node):
    """深度优先遍历 named 节点。"""
    yield node
    for child in getattr(node, 'named_children', []):
        yield from _iter_tree_sitter_named_nodes(child)

def _collect_javascript_call_targets_from_node(node, source_bytes: bytes) -> list[str]:
    """提取函数体中的调用目标，并跳过内部嵌套函数。"""
    call_targets: list[str] = []

    def _visit(current, *, allow_root: bool = False) -> None:
        if not allow_root and current.type in {'function_declaration', 'function_expression', 'arrow_function', 'method_definition', 'class_declaration'}:
            return
        if current.type == 'call_expression':
            function_node = current.child_by_field_name('function')
            if function_node is not None:
                normalized_target = normalize_javascript_call_target(get_node_text(function_node, source_bytes))
                if normalized_target:
                    call_targets.append(normalized_target)
        for child in getattr(current, 'named_children', []):
            _visit(child)

    _visit(node, allow_root=True)
    return _deduplicate_keep_order(call_targets)[:12]

def _extract_tree_sitter_string(node, source_bytes: bytes) -> str | None:
    """尽量从 tree-sitter 字符串节点中提取纯文本。"""
    if node is None:
        return None

    text = get_node_text(node, source_bytes).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", '`'}:
        return text[1:-1]
    return None

def _extract_tree_sitter_parameters(node, source_bytes: bytes) -> list[str]:
    """把参数节点转换成便于检索的参数列表。"""
    if node is None:
        return []
    text = get_node_text(node, source_bytes).strip()
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    return _split_parameter_text(text)

def _get_javascript_language_scope(file_path: str) -> str:
    """根据路径推断 JS/TS 的语言范围标签。"""
    lower_path = file_path.lower()
    if lower_path.endswith(('.ts', '.tsx')):
        return 'typescript'
    return 'nodejs'

def _get_javascript_class_bases(node, source_bytes: bytes) -> list[str]:
    """提取 class extends 的父类信息。"""
    for child in getattr(node, 'named_children', []):
        if child.type != 'class_heritage':
            continue
        text = get_node_text(child, source_bytes).strip()
        if text.lower().startswith('extends '):
            return [text[8:].strip()]
    return []

def _is_javascript_async_node(node, source_bytes: bytes) -> bool:
    """判断函数节点是否带有 async。"""
    text = get_node_text(node, source_bytes).lstrip()
    return text.startswith('async ')

def _extract_javascript_routes(
    file_content: KeyFileContent,
    accumulator: _CodeIndexAccumulator,
    language_scope: str,
) -> None:
    """提取 Express 风格的 app/router 路由注册。"""
    function_lookup = _build_function_lookup(accumulator.function_summaries, file_content.path)
    for route_match in _iter_js_route_matches(file_content.content):
        handler_name, handler_qualified_name, called_symbols = _resolve_js_route_handler(
            route_match.handler_expression,
            route_match.line_number,
            function_lookup,
        )
        accumulator.code_symbols.append(
            CodeSymbol(
                name=f"{'/'.join(route_match.http_methods)} {route_match.route_path}",
                symbol_type='route',
                source_path=file_content.path,
                line_number=route_match.line_number,
            )
        )
        accumulator.api_route_summaries.append(
            ApiRouteSummary(
                route_path=route_match.route_path,
                http_methods=route_match.http_methods,
                source_path=file_content.path,
                language_scope=language_scope,
                framework='express',
                handler_name=handler_name,
                handler_qualified_name=handler_qualified_name,
                owner_class=None,
                line_number=route_match.line_number,
                decorators=[route_match.handler_expression],
                called_symbols=called_symbols,
                summary=_build_api_route_summary_text(
                    route_path=route_match.route_path,
                    http_methods=route_match.http_methods,
                    handler_name=handler_qualified_name,
                    framework='express',
                    called_symbols=called_symbols,
                ),
            )
        )

def _build_function_lookup(
    function_summaries: list[FunctionSummary],
    source_path: str,
) -> dict[str, FunctionSummary]:
    """为 JS 路由处理函数构建简单查找表。"""
    lookup: dict[str, FunctionSummary] = {}
    for item in function_summaries:
        if item.source_path != source_path:
            continue
        lookup.setdefault(item.qualified_name, item)
        lookup.setdefault(item.name, item)
    return lookup

def _iter_js_route_matches(text: str) -> list[_JsRouteMatch]:
    """遍历 Express 风格的单行路由定义。"""
    pattern = re.compile(
        r'(?P<prefix>\b(?:app|router)\.(?P<method>get|post|put|delete|patch|options|head|all)\s*\(\s*)'
        r'(?P<quote>["\'`])(?P<path>[^"\'`]+)(?P=quote)\s*,\s*(?P<handler>[^\n;]+)',
        re.IGNORECASE,
    )
    matches: list[_JsRouteMatch] = []
    for match in pattern.finditer(text):
        method = match.group('method').upper()
        matches.append(
            _JsRouteMatch(
                http_methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']
                if method == 'ALL'
                else [method],
                route_path=match.group('path'),
                handler_expression=match.group('handler').rstrip().rstrip(')'),
                line_number=_line_number_from_offset(text, match.start()),
            )
        )
    return matches

def _resolve_js_route_handler(
    handler_expression: str,
    line_number: int,
    function_lookup: dict[str, FunctionSummary],
) -> tuple[str, str, list[str]]:
    """把 JS 路由处理器表达式转换成结构化字段。"""
    candidate = _pick_js_handler_candidate(handler_expression)
    if not candidate or '=>' in candidate or candidate.startswith('function') or candidate.startswith('async '):
        inline_name = f'inline_handler@L{line_number}'
        return inline_name, inline_name, _extract_javascript_called_symbols(handler_expression)

    normalized = normalize_javascript_call_target(candidate.rstrip(')').strip())
    handler_name = normalized.split('.')[-1]
    summary = function_lookup.get(normalized) or function_lookup.get(handler_name)
    if summary is not None:
        return handler_name, normalized, summary.called_symbols[:12]
    return handler_name, normalized, []

def _pick_js_handler_candidate(handler_expression: str) -> str:
    """从路由参数列表中挑出最可能的最终处理函数。"""
    candidates = _split_js_arguments(handler_expression)
    if not candidates:
        return handler_expression.strip()
    return candidates[-1].strip()

def _extract_js_class_methods(body: str, *, base_line: int) -> list[dict[str, object]]:
    """从类体中抽取方法定义。"""
    methods: list[dict[str, object]] = []
    method_pattern = re.compile(r'(?m)^\s*(async\s+)?([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{')
    for match in method_pattern.finditer(body):
        method_name = match.group(2)
        if method_name in {'if', 'for', 'while', 'switch', 'catch'}:
            continue
        relative_line_start = _line_number_from_offset(body, match.start())
        relative_line_end, method_body = _extract_block_from_match(body, match)
        line_start = base_line + relative_line_start - 1
        line_end = base_line + relative_line_end - 1
        methods.append(
            {
                'name': method_name,
                'signature': _normalize_javascript_signature(match.group(0), method_name),
                'line_start': line_start,
                'line_end': line_end,
                'is_async': bool(match.group(1)),
                'parameters': _split_parameter_text(match.group(3)),
                'called_symbols': _extract_javascript_called_symbols(method_body),
            }
        )
    return methods

def _extract_javascript_called_symbols(body: str) -> list[str]:
    """用轻量规则提取 JS/TS 函数体中的调用目标。"""
    keywords = {'if', 'for', 'while', 'switch', 'catch', 'return', 'new'}
    calls: list[str] = []
    pattern = re.compile(r'([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s*\(')
    for match in pattern.finditer(body):
        callee = normalize_javascript_call_target(match.group(1))
        if callee in keywords:
            continue
        if callee:
            calls.append(callee)
    return _deduplicate_keep_order(calls)[:12]

