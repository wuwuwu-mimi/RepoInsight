from pathlib import Path

from repoinsight.models.analysis_model import KeyFileContent
from repoinsight.models.file_model import FileEntry


# 关键文件默认最多读取 20 KB，避免后续分析输入过大。
DEFAULT_MAX_READ_BYTES = 20 * 1024



def read_key_files(
    root_path: str,
    key_files: list[FileEntry],
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
) -> list[KeyFileContent]:
    """读取关键文件内容，并在必要时截断。"""
    root = Path(root_path).resolve()
    contents: list[KeyFileContent] = []

    for entry in key_files:
        file_path = root / entry.path
        if not file_path.exists() or not file_path.is_file():
            continue

        try:
            raw_bytes = file_path.read_bytes()
        except OSError:
            continue

        truncated = len(raw_bytes) > max_read_bytes
        sliced_bytes = raw_bytes[:max_read_bytes]
        text = sliced_bytes.decode('utf-8', errors='replace').strip()

        contents.append(
            KeyFileContent(
                path=entry.path,
                size_bytes=entry.size_bytes,
                content=text,
                truncated=truncated,
            )
        )

    return contents
