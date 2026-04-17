from repoinsight.models.analysis_model import CodeEntity, CodeRelationEdge, KeyFileContent
from repoinsight.report.json_report import generate_json_report_payload
from repoinsight.report.markdown_report import generate_markdown_report
from tests.test_summary_builders import _build_result


def test_generate_markdown_report_includes_unified_code_structure_sections() -> None:
    result = _build_result(
        [KeyFileContent(path='app.py', size_bytes=128, content='def create_app():\n    return None\n')],
        ['app.py'],
    )
    result.project_profile.code_entities = [
        CodeEntity(
            entity_kind='function',
            name='create_app',
            qualified_name='create_app',
            source_path='app.py',
            language_scope='python',
            location='app.py:L1-L2',
            tags=['entrypoint'],
        )
    ]
    result.project_profile.code_relation_edges = [
        CodeRelationEdge(
            source_ref='route:GET /health',
            target_ref='function:create_app',
            relation_type='handle_route',
            source_path='app.py',
            line_number=1,
        )
    ]

    report = generate_markdown_report(result)

    assert '### 统一代码实体' in report
    assert '`app.py` · function `create_app`' in report
    assert '### 统一代码关系边' in report
    assert '`route:GET /health` -[handle_route]-> `function:create_app`' in report


def test_generate_json_report_payload_includes_unified_code_structure_fields() -> None:
    result = _build_result(
        [KeyFileContent(path='app.py', size_bytes=128, content='def create_app():\n    return None\n')],
        ['app.py'],
    )
    result.project_profile.code_entities = [
        CodeEntity(
            entity_kind='function',
            name='create_app',
            qualified_name='create_app',
            source_path='app.py',
            language_scope='python',
            location='app.py:L1-L2',
            tags=['entrypoint'],
        )
    ]
    result.project_profile.code_relation_edges = [
        CodeRelationEdge(
            source_ref='route:GET /health',
            target_ref='function:create_app',
            relation_type='handle_route',
            source_path='app.py',
            line_number=1,
        )
    ]

    payload = generate_json_report_payload(result)
    profile = payload['project_profile']

    assert profile['code_entities'] == [
        {
            'entity_kind': 'function',
            'name': 'create_app',
            'qualified_name': 'create_app',
            'source_path': 'app.py',
            'language_scope': 'python',
            'location': 'app.py:L1-L2',
            'tags': ['entrypoint'],
        }
    ]
    assert profile['code_relation_edges'] == [
        {
            'source_ref': 'route:GET /health',
            'target_ref': 'function:create_app',
            'relation_type': 'handle_route',
            'source_path': 'app.py',
            'line_number': 1,
        }
    ]
