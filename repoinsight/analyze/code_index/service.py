from __future__ import annotations

from repoinsight.analyze.code_index.common import _CodeIndexAccumulator
from repoinsight.analyze.code_index.javascript_indexer import _extract_javascript_index
from repoinsight.analyze.code_index.other_indexers import _extract_go_index, _extract_rust_index
from repoinsight.analyze.code_index.python_indexer import _extract_python_index
from repoinsight.models.analysis_model import ApiRouteSummary, ClassSummary, CodeSymbol, FunctionSummary, KeyFileContent, ModuleRelation

def extract_code_index(
    key_file_contents: list[KeyFileContent],
) -> tuple[
    list[CodeSymbol],
    list[ModuleRelation],
    list[FunctionSummary],
    list[ClassSummary],
    list[ApiRouteSummary],
]:
    """从关键文件中抽取代码级索引。"""
    accumulator = _CodeIndexAccumulator()

    for file_content in key_file_contents:
        lower_path = file_content.path.lower()
        if lower_path.endswith('.py'):
            _extract_python_index(file_content, accumulator)
        elif lower_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
            _extract_javascript_index(file_content, accumulator)
        elif lower_path.endswith('.go'):
            _extract_go_index(file_content, accumulator)
        elif lower_path.endswith('.rs'):
            _extract_rust_index(file_content, accumulator)

    return (
        accumulator.code_symbols,
        accumulator.module_relations,
        accumulator.function_summaries,
        accumulator.class_summaries,
        accumulator.api_route_summaries,
    )

