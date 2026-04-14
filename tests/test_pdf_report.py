import shutil
from pathlib import Path

import pytest
import io
from contextlib import redirect_stdout
from unittest.mock import patch

pytest.importorskip('reportlab')

import repoinsight.cli.main as cli_main
from repoinsight.report.pdf_report import (
    export_repo_report_to_pdf,
    get_pdf_report_path,
    remove_pdf_report,
)


def test_export_repo_report_to_pdf_creates_pdf_file() -> None:
    output_dir = 'data/test_pdf_export/reports'
    project_root = Path(__file__).resolve().parents[1]
    report_root = project_root / output_dir
    shutil.rmtree(report_root.parent, ignore_errors=True)

    try:
        report_root.mkdir(parents=True, exist_ok=True)
        markdown_path = report_root / 'demo__sample.md'
        markdown_path.write_text(
            '# RepoInsight Report: demo/sample\n\n'
            '## 项目概览\n'
            '- 仓库：`demo/sample`\n'
            '- 描述：用于验证 PDF 导出\n\n'
            '## 关键文件内容预览\n'
            '```text\n'
            'def main():\n'
            '    return \"ok\"\n'
            '```\n',
            encoding='utf-8',
        )

        pdf_path = export_repo_report_to_pdf(repo_id='demo/sample', output_dir=output_dir)

        assert pdf_path == get_pdf_report_path('demo/sample', output_dir=output_dir)
        assert pdf_path.exists() is True
        assert pdf_path.read_bytes().startswith(b'%PDF')
    finally:
        shutil.rmtree(report_root.parent, ignore_errors=True)


def test_remove_pdf_report_deletes_exported_file() -> None:
    output_dir = 'data/test_pdf_export_remove/reports'
    project_root = Path(__file__).resolve().parents[1]
    report_root = project_root / output_dir
    shutil.rmtree(report_root.parent, ignore_errors=True)

    try:
        report_root.mkdir(parents=True, exist_ok=True)
        markdown_path = report_root / 'demo__sample.md'
        markdown_path.write_text('# demo\n', encoding='utf-8')

        pdf_path = export_repo_report_to_pdf(repo_id='demo/sample', output_dir=output_dir)
        assert pdf_path.exists() is True

        with patch.object(Path, 'unlink', autospec=True, return_value=None) as unlink_mock:
            assert remove_pdf_report('demo/sample', output_dir=output_dir) is True
            unlink_mock.assert_called_once_with(pdf_path)
    finally:
        shutil.rmtree(report_root.parent, ignore_errors=True)


def test_export_command_prints_success_message() -> None:
    original_export = cli_main.export_repo_report_to_pdf
    try:
        cli_main.export_repo_report_to_pdf = lambda repo_id, output_dir='reports': Path(
            f'E:/PythonProject/RepoInsight/{output_dir}/demo__sample.pdf'
        )
        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.export_report('demo/sample')

        rendered = output.getvalue()
        assert 'PDF' in rendered
        assert 'demo__sample.pdf' in rendered
    finally:
        cli_main.export_repo_report_to_pdf = original_export
