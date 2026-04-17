from __future__ import annotations

import re

from repoinsight.analyze.code_index.common import _CodeIndexAccumulator
from repoinsight.models.analysis_model import CodeSymbol, KeyFileContent, ModuleRelation

def _extract_go_index(file_content: KeyFileContent, accumulator: _CodeIndexAccumulator) -> None:
    """保留 Go 的基础函数与 import 抽取。"""
    for index, raw_line in enumerate(file_content.content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        function_match = re.match(r'func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if function_match:
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=function_match.group(1),
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=index,
                )
            )
        import_match = re.match(r'"([^"]+)"', line)
        if import_match:
            accumulator.module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=import_match.group(1),
                    relation_type='import',
                    line_number=index,
                )
            )

def _extract_rust_index(file_content: KeyFileContent, accumulator: _CodeIndexAccumulator) -> None:
    """保留 Rust 的基础函数与 use 抽取。"""
    for index, raw_line in enumerate(file_content.content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        function_match = re.match(r'fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if function_match:
            accumulator.code_symbols.append(
                CodeSymbol(
                    name=function_match.group(1),
                    symbol_type='function',
                    source_path=file_content.path,
                    line_number=index,
                )
            )
        use_match = re.match(r'use\s+([^;]+);', line)
        if use_match:
            accumulator.module_relations.append(
                ModuleRelation(
                    source_path=file_content.path,
                    target=use_match.group(1),
                    relation_type='use',
                    line_number=index,
                )
            )

