import os
from pathlib import Path

from repoinsight.models.file_model import FileEntry, IgnoredEntry, ScanResult, ScanStats

IGNORED_DIR_NAMES = {
    '.git',
    '.idea',
    '.vscode',
    '.venv',
    'venv',
    'env',
    '__pycache__',
    'node_modules',
    'dist',
    'build',
    'coverage',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.tox',
    'target',
    'bin',
    'obj',
}

IGNORED_FILE_EXTENSIONS = {
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.svg',
    '.webp',
    '.ico',
    '.mp4',
    '.mov',
    '.avi',
    '.mp3',
    '.wav',
    '.zip',
    '.tar',
    '.gz',
    '.rar',
    '.7z',
    '.pdf',
    '.doc',
    '.docx',
    '.ppt',
    '.pptx',
    '.exe',
    '.dll',
    '.so',
    '.dylib',
    '.class',
    '.db',
    '.sqlite',
    '.sqlite3',
    '.parquet',
    '.npy',
    '.npz',
    '.log',
}

KEY_FILE_NAMES = {
    'readme.md',
    'README.md',
    'README',
    'pyproject.toml',
    'poetry.lock',
    'requirements.txt',
    'setup.py',
    'setup.cfg',
    'package.json',
    'package-lock.json',
    'pnpm-lock.yaml',
    'yarn.lock',
    'dockerfile',
    'docker-compose.yml',
    'docker-compose.yaml',
    '.env.example',
    'makefile',
    'main.py',
    'app.py',
    'manage.py',
    'cli.py',
    '__main__.py',
    'tsconfig.json',
    'vite.config.ts',
    'vite.config.js',
    'vite.config.mjs',
    'next.config.js',
    'next.config.mjs',
    'next.config.ts',
    'pom.xml',
    'build.gradle',
    'build.gradle.kts',
    'settings.gradle',
    'go.mod',
    'go.sum',
    'main.go',
    'cargo.toml',
    'cargo.lock',
    'main.rs',
    'lib.rs',
    'composer.json',
    'gemfile',
}

KEY_FILE_SUFFIXES = {
    '.csproj',
    '.sln',
}


def scan_repo(root_path: str, max_file_size: int = 1_000_000) -> ScanResult:
    """扫描仓库目录，返回候选文件、关键文件和目录树预览。"""
    root = Path(root_path).resolve()
    if not root.exists():
        raise ValueError(f'仓库路径不存在：{root_path}')
    if not root.is_dir():
        raise ValueError(f'仓库路径不是目录：{root_path}')

    all_files: list[FileEntry] = []
    key_files: list[FileEntry] = []
    ignored_entries: list[IgnoredEntry] = []
    total_seen = 0

    for current_root, dirs, files in os.walk(root):
        current_path = Path(current_root)

        kept_dirs: list[str] = []
        for dir_name in dirs:
            if should_ignore_dir(dir_name):
                ignored_dir_path = (current_path / dir_name).relative_to(root).as_posix()
                ignored_entries.append(
                    IgnoredEntry(
                        path=ignored_dir_path,
                        entry_type='directory',
                        reason='ignored_dir',
                    )
                )
                continue
            kept_dirs.append(dir_name)
        dirs[:] = kept_dirs

        for file_name in files:
            total_seen += 1
            file_path = current_path / file_name
            reason = get_ignore_reason(file_path, max_file_size)
            relative_path = file_path.relative_to(root).as_posix()
            if reason is not None:
                ignored_entries.append(
                    IgnoredEntry(
                        path=relative_path,
                        entry_type='file',
                        reason=reason,
                    )
                )
                continue

            entry = build_file_entry(root, file_path)
            all_files.append(entry)
            if entry.is_key_file:
                key_files.append(entry)

    stats = ScanStats(
        total_seen=total_seen,
        kept_count=len(all_files),
        ignored_count=len(ignored_entries),
        key_file_count=len(key_files),
    )
    tree_preview = build_tree_preview(all_files)

    return ScanResult(
        root_path=str(root),
        all_files=all_files,
        key_files=key_files,
        tree_preview=tree_preview,
        ignored_entries=ignored_entries,
        stats=stats,
    )


def should_ignore_dir(dir_name: str) -> bool:
    """判断目录名是否应被忽略。"""
    return dir_name.lower() in IGNORED_DIR_NAMES


def get_ignore_reason(file_path: Path, max_file_size: int) -> str | None:
    """判断文件是否应该被忽略；若应忽略则返回原因。"""
    if not file_path.is_file():
        return 'not_file'

    if file_path.suffix.lower() in IGNORED_FILE_EXTENSIONS:
        return 'ignored_extension'

    if file_path.stat().st_size > max_file_size:
        return 'file_too_large'

    if is_binary_file(file_path):
        return 'binary_file'

    return None


def is_binary_file(file_path: Path, chunk_size: int = 1024) -> bool:
    """用一个轻量规则判断文件是否像二进制文件。"""
    try:
        data = file_path.read_bytes()[:chunk_size]
    except OSError:
        return True

    if not data:
        return False

    if b'\x00' in data:
        return True

    return False


def is_key_file(path: str) -> bool:
    """按文件名判断是否为关键文件。"""
    normalized = Path(path).as_posix().lower()
    file_name = Path(path).name.lower()

    if file_name in KEY_FILE_NAMES:
        return True

    if Path(path).suffix.lower() in KEY_FILE_SUFFIXES:
        return True

    if normalized.startswith('.github/workflows/'):
        return True

    return False


def build_file_entry(root_path: Path, file_path: Path) -> FileEntry:
    """把文件路径转换为统一的 FileEntry 模型。"""
    relative_path = file_path.relative_to(root_path).as_posix()
    parent_dir = file_path.relative_to(root_path).parent.as_posix()
    if parent_dir == '.':
        parent_dir = ''

    extension = file_path.suffix.lower() or None
    return FileEntry(
        path=relative_path,
        name=file_path.name,
        extension=extension,
        size_bytes=file_path.stat().st_size,
        parent_dir=parent_dir,
        is_key_file=is_key_file(relative_path),
    )


def build_tree_preview(all_files: list[FileEntry], max_depth: int = 2) -> list[str]:
    """根据候选文件生成一个简化版目录树预览。"""
    preview_set: set[str] = set()

    for entry in all_files:
        parts = entry.path.split('/')
        upper = min(len(parts), max_depth)
        for depth in range(1, upper + 1):
            item = '/'.join(parts[:depth])
            if depth < len(parts):
                item = f'{item}/'
            preview_set.add(item)

    return sorted(preview_set)
