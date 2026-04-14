from __future__ import annotations

import gc
import io
import os
import re
import stat
import time
from html import escape
from pathlib import Path

from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.report.markdown_report import generate_markdown_report, get_report_path


def save_pdf_report(
    result: AnalysisRunResult,
    output_dir: str = 'reports',
) -> Path:
    """把分析结果导出为 PDF 报告。"""
    repo_id = result.repo_info.repo_model.full_name
    pdf_path = get_pdf_report_path(repo_id=repo_id, output_dir=output_dir)
    markdown_text = generate_markdown_report(result)
    export_markdown_text_to_pdf(
        markdown_text=markdown_text,
        target_path=pdf_path,
        title=f'RepoInsight Report: {repo_id}',
    )
    return pdf_path


def export_repo_report_to_pdf(
    repo_id: str,
    output_dir: str = 'reports',
) -> Path:
    """把已保存的 Markdown 报告再次导出为 PDF。"""
    markdown_path = get_report_path(repo_id=repo_id, output_dir=output_dir)
    if not markdown_path.exists():
        raise FileNotFoundError(f'未找到 Markdown 报告：{markdown_path}')

    markdown_text = markdown_path.read_text(encoding='utf-8')
    pdf_path = get_pdf_report_path(repo_id=repo_id, output_dir=output_dir)
    export_markdown_text_to_pdf(
        markdown_text=markdown_text,
        target_path=pdf_path,
        title=f'RepoInsight Report: {repo_id}',
    )
    return pdf_path


def export_markdown_text_to_pdf(
    markdown_text: str,
    target_path: Path,
    title: str = 'RepoInsight Report',
) -> Path:
    """将 Markdown 文本转换成一个可阅读的 PDF 文件。"""
    SimpleDocTemplate, Paragraph, Preformatted, Spacer, PageBreak, getSampleStyleSheet, ParagraphStyle = (
        _load_reportlab_components()
    )
    pdfmetrics, UnicodeCIDFont, colors, A4 = _load_reportlab_support()

    target_path.parent.mkdir(parents=True, exist_ok=True)

    base_font_name = 'STSong-Light'
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(base_font_name))
    except (KeyError, ValueError):
        # 重复注册或运行时已存在时直接复用。
        pass

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        name='RepoInsightBody',
        parent=styles['BodyText'],
        fontName=base_font_name,
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        name='RepoInsightBullet',
        parent=body_style,
        leftIndent=16,
        firstLineIndent=-10,
    )
    code_style = ParagraphStyle(
        name='RepoInsightCode',
        parent=body_style,
        fontName=base_font_name,
        fontSize=9.5,
        leading=14,
        leftIndent=10,
        rightIndent=10,
        backColor=colors.HexColor('#F3F4F6'),
        borderPadding=8,
        borderWidth=0.5,
        borderColor=colors.HexColor('#E5E7EB'),
        spaceBefore=4,
        spaceAfter=10,
    )
    heading_styles = {
        1: ParagraphStyle(
            name='RepoInsightHeading1',
            parent=body_style,
            fontSize=18,
            leading=24,
            textColor=colors.HexColor('#0F172A'),
            spaceBefore=8,
            spaceAfter=10,
        ),
        2: ParagraphStyle(
            name='RepoInsightHeading2',
            parent=body_style,
            fontSize=14,
            leading=20,
            textColor=colors.HexColor('#111827'),
            spaceBefore=8,
            spaceAfter=8,
        ),
        3: ParagraphStyle(
            name='RepoInsightHeading3',
            parent=body_style,
            fontSize=12,
            leading=18,
            textColor=colors.HexColor('#1F2937'),
            spaceBefore=6,
            spaceAfter=6,
        ),
    }

    elements: list[object] = []
    paragraph_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code_block = False

    def flush_paragraph_buffer() -> None:
        if not paragraph_buffer:
            return
        paragraph_text = ' '.join(item.strip() for item in paragraph_buffer if item.strip())
        paragraph_buffer.clear()
        if paragraph_text:
            elements.append(Paragraph(_render_inline_markdown(paragraph_text), body_style))

    def flush_code_buffer() -> None:
        if not code_buffer:
            return
        code_text = '\n'.join(code_buffer).rstrip()
        code_buffer.clear()
        if code_text:
            elements.append(Preformatted(code_text, code_style))

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip('\n')
        stripped = line.strip()

        if stripped.startswith('```'):
            flush_paragraph_buffer()
            if in_code_block:
                flush_code_buffer()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buffer.append(line)
            continue

        if not stripped:
            flush_paragraph_buffer()
            if elements and not isinstance(elements[-1], Spacer):
                elements.append(Spacer(1, 4))
            continue

        if stripped == '---':
            flush_paragraph_buffer()
            elements.append(PageBreak())
            continue

        heading_level = _detect_heading_level(stripped)
        if heading_level is not None:
            flush_paragraph_buffer()
            heading_text = stripped[heading_level + 1 :].strip()
            elements.append(Paragraph(_render_inline_markdown(heading_text), heading_styles[heading_level]))
            continue

        if stripped.startswith('- '):
            flush_paragraph_buffer()
            bullet_text = stripped[2:].strip()
            elements.append(Paragraph(f'• {_render_inline_markdown(bullet_text)}', bullet_style))
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph_buffer()
    flush_code_buffer()

    if not elements:
        elements.append(Paragraph('报告内容为空', body_style))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=title,
        author='RepoInsight',
        leftMargin=42,
        rightMargin=42,
        topMargin=48,
        bottomMargin=42,
    )
    doc.build(
        elements,
        onFirstPage=lambda canvas, document: _draw_page_footer(canvas, document, title, base_font_name),
        onLaterPages=lambda canvas, document: _draw_page_footer(canvas, document, title, base_font_name),
    )
    pdf_bytes = buffer.getvalue()
    buffer.close()
    target_path.write_bytes(pdf_bytes)
    del doc
    gc.collect()
    return target_path


def get_pdf_report_path(repo_id: str, output_dir: str = 'reports') -> Path:
    """根据仓库标识返回 PDF 报告路径。"""
    owner, repo = _parse_repo_id(repo_id)
    project_root = Path(__file__).resolve().parents[2]
    report_root = project_root / output_dir
    return report_root / f'{owner}__{repo}.pdf'


def remove_pdf_report(repo_id: str, output_dir: str = 'reports') -> bool:
    """删除指定仓库的 PDF 报告。"""
    pdf_path = get_pdf_report_path(repo_id=repo_id, output_dir=output_dir)
    if not pdf_path.exists():
        return False

    last_error: PermissionError | None = None
    for _ in range(3):
        try:
            pdf_path.unlink()
            return True
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            try:
                os.chmod(pdf_path, stat.S_IWRITE)
            except OSError:
                pass
            time.sleep(0.1)

    if last_error is not None:
        raise last_error
    return False


def _load_reportlab_components():
    """延迟加载 reportlab，避免未安装时影响其他命令。"""
    try:
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            '未安装 reportlab，无法导出 PDF。请先安装依赖：pip install reportlab'
        ) from exc

    return SimpleDocTemplate, Paragraph, Preformatted, Spacer, PageBreak, getSampleStyleSheet, ParagraphStyle


def _load_reportlab_support():
    """加载 PDF 样式和中文字体支持。"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            '未安装 reportlab，无法导出 PDF。请先安装依赖：pip install reportlab'
        ) from exc

    return pdfmetrics, UnicodeCIDFont, colors, A4


def _draw_page_footer(canvas, document, title: str, font_name: str) -> None:
    """在每一页底部绘制标题和页码。"""
    canvas.saveState()
    canvas.setFont(font_name, 9)
    canvas.setFillColorRGB(0.39, 0.43, 0.48)
    canvas.drawString(document.leftMargin, 22, title)
    canvas.drawRightString(document.pagesize[0] - document.rightMargin, 22, f'第 {canvas.getPageNumber()} 页')
    canvas.restoreState()


def _detect_heading_level(line: str) -> int | None:
    """识别当前行是否为一级到三级标题。"""
    for level in (3, 2, 1):
        prefix = '#' * level + ' '
        if line.startswith(prefix):
            return level
    return None


def _render_inline_markdown(text: str) -> str:
    """把常见的 Markdown 行内语法转成 reportlab 可识别的简单标签。"""
    escaped = escape(text)
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    escaped = re.sub(r'`(.+?)`', r'<font color="#0F766E">\1</font>', escaped)
    return escaped


def _parse_repo_id(repo_id: str) -> tuple[str, str]:
    """把 owner/repo 形式的仓库标识拆成 owner 和 repo。"""
    normalized = repo_id.strip().strip('/')
    parts = [part for part in normalized.split('/') if part]
    if len(parts) != 2:
        raise ValueError('仓库标识格式应为 owner/repo')

    owner, repo = parts
    if owner in {'.', '..'} or repo in {'.', '..'}:
        raise ValueError('仓库标识不合法')

    return owner, repo
