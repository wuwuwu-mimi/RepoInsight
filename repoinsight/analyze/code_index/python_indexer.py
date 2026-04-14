from __future__ import annotations

import ast
import re

from repoinsight.analyze.code_index.common import (
    _CodeIndexAccumulator,
    _build_api_route_summary_text,
    _build_python_class_summary,
    _build_python_function_summary,
    _deduplicate_keep_order,
    _extract_python_http_methods,
    _extract_python_route_path,
    _extract_python_string_list,
    _extract_string_literal,
    _infer_python_route_framework,
    _line_number_from_offset,
    _parse_python_route_decorator,
    _PythonRouteMatch,
    _split_js_arguments,
)
from repoinsight.analyze.tree_sitter_support import get_node_text, parse_source_with_tree_sitter
from repoinsight.models.analysis_model import ApiRouteSummary, ClassSummary, CodeSymbol, FunctionSummary, KeyFileContent, ModuleRelation

class _PythonFunctionInspector(ast.NodeVisitor):
    """提取 Python 函数体中的调用和返回线索，忽略内部嵌套定义。"""

    def __init__(self) -> None:
        self.called_symbols: list[str] = []
        self.return_signals: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        target = _python_expr_to_text(node.func)
        if target:
            self.called_symbols.append(target)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            value_text = _python_expr_to_text(node.value)
            if value_text:
                self.return_signals.append(value_text)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

class _PythonModuleIndexer(ast.NodeVisitor):
    """使用 AST 提取 Python 的符号、依赖和函数级摘要。"""

    def __init__(self, file_content: KeyFileContent, accumulator: _CodeIndexAccumulator) -> None:
        self.file_content = file_content
        self.accumulator = accumulator

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.accumulator.module_relations.append(
                ModuleRelation(
                    source_path=self.file_content.path,
                    target=alias.name,
                    relation_type='import',
                    line_number=node.lineno,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ''
        if node.level:
            module_name = '.' * node.level + module_name
        if module_name:
            self.accumulator.module_relations.append(
                ModuleRelation(
                    source_path=self.file_content.path,
                    target=module_name,
                    relation_type='import',
                    line_number=node.lineno,
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.accumulator.code_symbols.append(
            CodeSymbol(
                name=node.name,
                symbol_type='class',
                source_path=self.file_content.path,
                line_number=node.lineno,
            )
        )

        method_names: list[str] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_names.append(child.name)
                self._add_function_summary(child, owner_class=node.name)

        self.accumulator.class_summaries.append(
            ClassSummary(
                name=node.name,
                qualified_name=node.name,
                source_path=self.file_content.path,
                language_scope='python',
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno),
                bases=_deduplicate_keep_order(
                    [_python_expr_to_text(base) for base in node.bases if _python_expr_to_text(base)]
                )[:8],
                decorators=_deduplicate_keep_order(
                    [_python_expr_to_text(item) for item in node.decorator_list if _python_expr_to_text(item)]
                )[:8],
                methods=method_names[:12],
                summary=_build_python_class_summary(node.name, method_names),
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function_summary(node, owner_class=None)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function_summary(node, owner_class=None)

    def _add_function_summary(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        owner_class: str | None,
    ) -> None:
        self.accumulator.code_symbols.append(
            CodeSymbol(
                name=node.name,
                symbol_type='method' if owner_class else 'function',
                source_path=self.file_content.path,
                line_number=node.lineno,
            )
        )

        inspector = _PythonFunctionInspector()
        for child in node.body:
            inspector.visit(child)

        parameters = _extract_python_parameters(node)
        decorators = _deduplicate_keep_order(
            [_python_expr_to_text(item) for item in node.decorator_list if _python_expr_to_text(item)]
        )[:8]
        called_symbols = _deduplicate_keep_order(inspector.called_symbols)[:12]
        return_signals = _deduplicate_keep_order(inspector.return_signals)[:6]
        qualified_name = f'{owner_class}.{node.name}' if owner_class else node.name

        self.accumulator.function_summaries.append(
            FunctionSummary(
                name=node.name,
                qualified_name=qualified_name,
                source_path=self.file_content.path,
                language_scope='python',
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno),
                signature=_build_python_signature(node, owner_class),
                owner_class=owner_class,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                decorators=decorators,
                parameters=parameters,
                called_symbols=called_symbols,
                return_signals=return_signals,
                summary=_build_python_function_summary(
                    name=node.name,
                    owner_class=owner_class,
                    decorators=decorators,
                    called_symbols=called_symbols,
                    return_signals=return_signals,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                ),
            )
        )

        self._add_python_route_summaries(
            node=node,
            owner_class=owner_class,
            qualified_name=qualified_name,
            decorators=decorators,
            called_symbols=called_symbols,
        )

    def _add_python_route_summaries(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        owner_class: str | None,
        qualified_name: str,
        decorators: list[str],
        called_symbols: list[str],
    ) -> None:
        for decorator in node.decorator_list:
            route_match = _parse_python_route_decorator(decorator)
            if route_match is None:
                continue

            self.accumulator.code_symbols.append(
                CodeSymbol(
                    name=f"{'/'.join(route_match.http_methods)} {route_match.route_path}",
                    symbol_type='route',
                    source_path=self.file_content.path,
                    line_number=node.lineno,
                )
            )
            self.accumulator.api_route_summaries.append(
                ApiRouteSummary(
                    route_path=route_match.route_path,
                    http_methods=route_match.http_methods,
                    source_path=self.file_content.path,
                    language_scope='python',
                    framework=route_match.framework,
                    handler_name=node.name,
                    handler_qualified_name=qualified_name,
                    owner_class=owner_class,
                    line_number=node.lineno,
                    decorators=decorators,
                    called_symbols=called_symbols,
                    summary=_build_api_route_summary_text(
                        route_path=route_match.route_path,
                        http_methods=route_match.http_methods,
                        handler_name=qualified_name,
                        framework=route_match.framework,
                        called_symbols=called_symbols,
                    ),
                )
            )

def _extract_python_index(file_content: KeyFileContent, accumulator: _CodeIndexAccumulator) -> None:
    """优先使用 tree-sitter 提取 Python 结构；失败时退回 AST/正则。"""
    if _extract_python_index_with_tree_sitter(file_content, accumulator):
        return

    try:
        tree = ast.parse(file_content.content)
    except SyntaxError:
        _extract_python_index_with_regex(file_content, accumulator)
        return

    _PythonModuleIndexer(file_content, accumulator).visit(tree)

def _extract_python_index_with_tree_sitter(
    file_content: KeyFileContent,
    accumulator: _CodeIndexAccumulator,
) -> bool:
    """使用 tree-sitter 提取 Python 的导入、类、函数和路由信息。"""
    document = parse_source_with_tree_sitter(file_content.path, file_content.content)
    if document is None or document.language_name != 'python':
        return False

    for node in getattr(document.root_node, 'named_children', []):
        if node.type in {'import_statement', 'import_from_statement'}:
            _collect_python_tree_sitter_imports(
                node=node,
                source_path=file_content.path,
                accumulator=accumulator,
                source_bytes=document.source_bytes,
            )
            continue

        resolved_definition, decorators, line_node = _resolve_python_tree_sitter_definition(
            node,
            document.source_bytes,
        )
        if resolved_definition is None:
            continue

        if resolved_definition.type == 'class_definition':
            _collect_python_tree_sitter_class(
                class_node=resolved_definition,
                decorators=decorators,
                line_node=line_node,
                source_path=file_content.path,
                accumulator=accumulator,
                source_bytes=document.source_bytes,
            )
            continue

        if resolved_definition.type == 'function_definition':
            _collect_python_tree_sitter_function(
                function_node=resolved_definition,
                decorators=decorators,
                line_node=line_node,
                owner_class=None,
                source_path=file_content.path,
                accumulator=accumulator,
                source_bytes=document.source_bytes,
            )

    return True

def _collect_python_tree_sitter_imports(
    node,
    source_path: str,
    accumulator: _CodeIndexAccumulator,
    source_bytes: bytes,
) -> None:
    """提取 tree-sitter Python 导入关系。"""
    line_number = node.start_point.row + 1
    text = get_node_text(node, source_bytes).strip()
    if node.type == 'import_statement':
        import_text = text[len('import ') :] if text.startswith('import ') else text
        for raw_item in import_text.split(','):
            item = raw_item.strip()
            if not item:
                continue
            target = item.split(' as ', maxsplit=1)[0].strip()
            if target:
                accumulator.module_relations.append(
                    ModuleRelation(
                        source_path=source_path,
                        target=target,
                        relation_type='import',
                        line_number=line_number,
                    )
                )
        return

    module_match = re.match(r'from\s+([A-Za-z0-9_\.]+|\.+[A-Za-z0-9_\.]*)\s+import\b', text)
    if module_match is None:
        return
    accumulator.module_relations.append(
        ModuleRelation(
            source_path=source_path,
            target=module_match.group(1),
            relation_type='import',
            line_number=line_number,
        )
    )

def _collect_python_tree_sitter_class(
    class_node,
    decorators: list[str],
    line_node,
    source_path: str,
    accumulator: _CodeIndexAccumulator,
    source_bytes: bytes,
) -> None:
    """提取 tree-sitter Python 类及其方法。"""
    name_node = class_node.child_by_field_name('name')
    body_node = class_node.child_by_field_name('body')
    if name_node is None or body_node is None:
        return

    class_name = get_node_text(name_node, source_bytes).strip()
    line_number = line_node.start_point.row + 1
    method_names: list[str] = []

    for child in getattr(body_node, 'named_children', []):
        resolved_definition, child_decorators, child_line_node = _resolve_python_tree_sitter_definition(
            child,
            source_bytes,
        )
        if resolved_definition is None or resolved_definition.type != 'function_definition':
            continue
        method_names.append(get_node_text(resolved_definition.child_by_field_name('name'), source_bytes).strip())
        _collect_python_tree_sitter_function(
            function_node=resolved_definition,
            decorators=child_decorators,
            line_node=child_line_node,
            owner_class=class_name,
            source_path=source_path,
            accumulator=accumulator,
            source_bytes=source_bytes,
        )

    accumulator.code_symbols.append(
        CodeSymbol(
            name=class_name,
            symbol_type='class',
            source_path=source_path,
            line_number=line_number,
        )
    )
    accumulator.class_summaries.append(
        ClassSummary(
            name=class_name,
            qualified_name=class_name,
            source_path=source_path,
            language_scope='python',
            line_start=line_number,
            line_end=class_node.end_point.row + 1,
            bases=_extract_python_tree_sitter_bases(class_node, source_bytes),
            decorators=decorators[:8],
            methods=method_names[:12],
            summary=_build_python_class_summary(class_name, method_names),
        )
    )

def _collect_python_tree_sitter_function(
    function_node,
    decorators: list[str],
    line_node,
    owner_class: str | None,
    source_path: str,
    accumulator: _CodeIndexAccumulator,
    source_bytes: bytes,
) -> None:
    """提取 tree-sitter Python 函数或方法。"""
    name_node = function_node.child_by_field_name('name')
    body_node = function_node.child_by_field_name('body')
    if name_node is None or body_node is None:
        return

    name = get_node_text(name_node, source_bytes).strip()
    line_number = line_node.start_point.row + 1
    called_symbols = _collect_python_tree_sitter_call_targets(body_node, source_bytes)
    return_signals = _collect_python_tree_sitter_return_signals(body_node, source_bytes)
    qualified_name = f'{owner_class}.{name}' if owner_class else name
    is_async = _is_python_tree_sitter_async(function_node, source_bytes)
    parameters = _extract_python_tree_sitter_parameters(function_node.child_by_field_name('parameters'), source_bytes)

    accumulator.code_symbols.append(
        CodeSymbol(
            name=name,
            symbol_type='method' if owner_class else 'function',
            source_path=source_path,
            line_number=line_number,
        )
    )
    accumulator.function_summaries.append(
        FunctionSummary(
            name=name,
            qualified_name=qualified_name,
            source_path=source_path,
            language_scope='python',
            line_start=line_number,
            line_end=function_node.end_point.row + 1,
            signature=_build_python_tree_sitter_signature(
                function_node=function_node,
                owner_class=owner_class,
                source_bytes=source_bytes,
            ),
            owner_class=owner_class,
            is_async=is_async,
            decorators=decorators[:8],
            parameters=parameters,
            called_symbols=called_symbols,
            return_signals=return_signals,
            summary=_build_python_function_summary(
                name=name,
                owner_class=owner_class,
                decorators=decorators,
                called_symbols=called_symbols,
                return_signals=return_signals,
                is_async=is_async,
            ),
        )
    )
    _add_python_tree_sitter_route_summaries(
        function_node=function_node,
        decorators=decorators,
        owner_class=owner_class,
        qualified_name=qualified_name,
        source_path=source_path,
        line_number=line_number,
        called_symbols=called_symbols,
        accumulator=accumulator,
        source_bytes=source_bytes,
    )

def _resolve_python_tree_sitter_definition(
    node,
    source_bytes: bytes,
) -> tuple[object | None, list[str], object | None]:
    """展开 decorated_definition，只返回真正的定义节点与装饰器文本。"""
    if node.type == 'decorated_definition':
        definition_node = node.child_by_field_name('definition')
        decorators: list[str] = []
        for child in getattr(node, 'named_children', []):
            if child.type == 'decorator':
                decorators.append(get_node_text(child, source_bytes).strip())
        return definition_node, decorators, node
    if node.type in {'function_definition', 'class_definition'}:
        return node, [], node
    return None, [], None

def _extract_python_tree_sitter_bases(class_node, source_bytes: bytes) -> list[str]:
    """提取 Python 类定义中的继承基类。"""
    for child in getattr(class_node, 'named_children', []):
        if child.type != 'argument_list':
            continue
        return _deduplicate_keep_order(
            [
                get_node_text(item, source_bytes).strip()
                for item in getattr(child, 'named_children', [])
                if get_node_text(item, source_bytes).strip()
            ]
        )[:8]
    return []

def _collect_python_tree_sitter_call_targets(node, source_bytes: bytes) -> list[str]:
    """提取函数体中的调用目标，并跳过内部嵌套定义。"""
    calls: list[str] = []

    def _visit(current, *, allow_root: bool = False) -> None:
        if not allow_root and current.type in {'function_definition', 'class_definition', 'lambda'}:
            return
        if not allow_root and current.type == 'decorated_definition':
            return

        if current.type == 'call':
            function_node = current.child_by_field_name('function')
            if function_node is not None:
                target = get_node_text(function_node, source_bytes).strip()
                if target:
                    calls.append(target)
        for child in getattr(current, 'named_children', []):
            _visit(child)

    _visit(node, allow_root=True)
    return _deduplicate_keep_order(calls)[:12]

def _collect_python_tree_sitter_return_signals(node, source_bytes: bytes) -> list[str]:
    """提取函数体中的 return 线索，并跳过内部嵌套定义。"""
    signals: list[str] = []

    def _visit(current, *, allow_root: bool = False) -> None:
        if not allow_root and current.type in {'function_definition', 'class_definition', 'lambda'}:
            return
        if not allow_root and current.type == 'decorated_definition':
            return

        if current.type == 'return_statement':
            named_children = list(getattr(current, 'named_children', []))
            if named_children:
                signal = get_node_text(named_children[-1], source_bytes).strip()
                if signal:
                    signals.append(signal)
        for child in getattr(current, 'named_children', []):
            _visit(child)

    _visit(node, allow_root=True)
    return _deduplicate_keep_order(signals)[:6]

def _extract_python_tree_sitter_parameters(node, source_bytes: bytes) -> list[str]:
    """提取 Python 参数名列表。"""
    if node is None:
        return []
    text = get_node_text(node, source_bytes).strip()
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    return _split_python_parameter_text(text)

def _split_python_parameter_text(text: str) -> list[str]:
    """把 Python 形参文本拆成适合检索的参数列表。"""
    if not text.strip():
        return []

    parameters: list[str] = []
    for raw_item in _split_js_arguments(text):
        item = raw_item.strip()
        if not item or item in {'*', '/'}:
            continue
        item = re.split(r'(?<!\*)[:=]', item, maxsplit=1)[0].strip()
        if item and item not in {'*', '/'}:
            parameters.append(item)
    return parameters

def _build_python_tree_sitter_signature(
    function_node,
    owner_class: str | None,
    source_bytes: bytes,
) -> str:
    """构造适合展示的 Python 函数签名。"""
    name_node = function_node.child_by_field_name('name')
    parameters_node = function_node.child_by_field_name('parameters')
    name = get_node_text(name_node, source_bytes).strip() if name_node is not None else 'unknown'
    qualified_name = f'{owner_class}.{name}' if owner_class else name
    parameters_text = get_node_text(parameters_node, source_bytes).strip() if parameters_node is not None else '()'
    prefix = 'async def' if _is_python_tree_sitter_async(function_node, source_bytes) else 'def'

    return_annotation = ''
    for child in getattr(function_node, 'named_children', []):
        if child.type == 'type':
            annotation_text = get_node_text(child, source_bytes).strip()
            if annotation_text:
                return_annotation = f' -> {annotation_text}'
            break

    return f'{prefix} {qualified_name}{parameters_text}{return_annotation}'

def _is_python_tree_sitter_async(function_node, source_bytes: bytes) -> bool:
    """判断 tree-sitter Python 函数节点是否带 async。"""
    text = get_node_text(function_node, source_bytes).lstrip()
    return text.startswith('async def ')

def _add_python_tree_sitter_route_summaries(
    function_node,
    decorators: list[str],
    owner_class: str | None,
    qualified_name: str,
    source_path: str,
    line_number: int,
    called_symbols: list[str],
    accumulator: _CodeIndexAccumulator,
    source_bytes: bytes,
) -> None:
    """根据装饰器文本提取 Python 路由摘要。"""
    name_node = function_node.child_by_field_name('name')
    if name_node is None:
        return
    handler_name = get_node_text(name_node, source_bytes).strip() or qualified_name.split('.')[-1]

    for decorator_text in decorators:
        route_match = _parse_python_route_decorator_text(decorator_text)
        if route_match is None:
            continue

        accumulator.code_symbols.append(
            CodeSymbol(
                name=f"{'/'.join(route_match.http_methods)} {route_match.route_path}",
                symbol_type='route',
                source_path=source_path,
                line_number=line_number,
            )
        )
        accumulator.api_route_summaries.append(
            ApiRouteSummary(
                route_path=route_match.route_path,
                http_methods=route_match.http_methods,
                source_path=source_path,
                language_scope='python',
                framework=route_match.framework,
                handler_name=handler_name,
                handler_qualified_name=qualified_name,
                owner_class=owner_class,
                line_number=line_number,
                decorators=decorators[:8],
                called_symbols=called_symbols,
                summary=_build_api_route_summary_text(
                    route_path=route_match.route_path,
                    http_methods=route_match.http_methods,
                    handler_name=qualified_name,
                    framework=route_match.framework,
                    called_symbols=called_symbols,
                ),
            )
        )

def _parse_python_route_decorator_text(text: str) -> _PythonRouteMatch | None:
    """把装饰器文本转换成统一的 Python 路由匹配结果。"""
    normalized = text.strip()
    if normalized.startswith('@'):
        normalized = normalized[1:]

    try:
        expression = ast.parse(normalized, mode='eval').body
    except SyntaxError:
        return None
    return _parse_python_route_decorator(expression)

def _extract_python_index_with_regex(file_content: KeyFileContent, accumulator: _CodeIndexAccumulator) -> None:
    """当 Python 源码不完整时，用轻量正则保底。"""
    for index, raw_line in enumerate(file_content.content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        function_match = re.match(r'(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)', line)
        if function_match:
            name = function_match.group(1)
            signature = line.rstrip(':')
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=name,
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=index,
                )
            )
            accumulator.function_summaries.append(
                FunctionSummary(
                    name=name,
                    qualified_name=name,
                    source_path=file_content.path,
                    language_scope='python',
                    line_start=index,
                    line_end=index,
                    signature=signature,
                    owner_class=None,
                    is_async=line.startswith('async def '),
                    decorators=[],
                    parameters=_split_parameter_text(function_match.group(2)),
                    called_symbols=[],
                    return_signals=[],
                    summary='该函数已被识别，但由于源码无法完整解析，当前仅保留基础声明信息。',
                )
            )
            continue

        class_match = re.match(r'class\s+([A-Za-z_][A-Za-z0-9_]*)', line)
        if class_match:
            name = class_match.group(1)
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=name,
                    symbol_type='class',
                    source_path=file_content.path,
                    line_number=index,
                )
            )
            accumulator.class_summaries.append(
                ClassSummary(
                    name=name,
                    qualified_name=name,
                    source_path=file_content.path,
                    language_scope='python',
                    line_start=index,
                    line_end=index,
                    bases=[],
                    decorators=[],
                    methods=[],
                    summary='该类已被识别，但由于源码无法完整解析，当前仅保留基础声明信息。',
                )
            )
            continue

        import_match = re.match(r'import\s+([A-Za-z0-9_\.]+)', line)
        if import_match:
            accumulator.module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=import_match.group(1),
                    relation_type='import',
                    line_number=index,
                )
            )
            continue

        from_match = re.match(r'from\s+([A-Za-z0-9_\.]+)\s+import', line)
        if from_match:
            accumulator.module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=from_match.group(1),
                    relation_type='import',
                    line_number=index,
                )
            )

def _extract_python_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """提取 Python 函数参数名。"""
    parameters: list[str] = []
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    parameters.extend(arg.arg for arg in positional)
    if args.vararg is not None:
        parameters.append(f'*{args.vararg.arg}')
    parameters.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        parameters.append(f'**{args.kwarg.arg}')
    return parameters

def _build_python_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    owner_class: str | None,
) -> str:
    """构造适合展示的 Python 函数签名。"""
    parameters = ', '.join(_extract_python_parameters(node))
    prefix = 'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'
    name = f'{owner_class}.{node.name}' if owner_class else node.name
    return_annotation = ''
    if node.returns is not None:
        annotation_text = _python_expr_to_text(node.returns)
        if annotation_text:
            return_annotation = f' -> {annotation_text}'
    return f'{prefix} {name}({parameters}){return_annotation}'

def _python_expr_to_text(node: ast.AST) -> str:
    """尽量把 AST 表达式转成可读文本。"""
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

