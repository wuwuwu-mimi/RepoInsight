import os
import shutil
import stat
from datetime import datetime
from pathlib import Path

from repoinsight.models.cache_model import CachedRepoEntry, CachedRepoListResult
from repoinsight.report.markdown_report import get_report_path


# clone 缓存固定存放在项目根目录下的 clone 目录。
DEFAULT_CLONE_DIR = 'clone'



def get_project_root() -> Path:
    """返回项目根目录路径。"""
    return Path(__file__).resolve().parents[2]



def get_clone_root(target_dir: str = DEFAULT_CLONE_DIR) -> Path:
    """返回本地 clone 根目录路径。"""
    return get_project_root() / target_dir



def parse_repo_id(repo_id: str) -> tuple[str, str]:
    """把 owner/repo 形式的仓库标识拆分成 owner 和 repo。"""
    normalized = repo_id.strip().strip('/')
    parts = [part for part in normalized.split('/') if part]
    if len(parts) != 2:
        raise ValueError('仓库标识格式应为 owner/repo')

    owner, repo = parts
    if owner in {'.', '..'} or repo in {'.', '..'}:
        raise ValueError('仓库标识不合法')

    return owner, repo



def get_clone_path(repo_id: str, target_dir: str = DEFAULT_CLONE_DIR) -> Path:
    """根据仓库标识返回本地克隆目录路径。"""
    owner, repo = parse_repo_id(repo_id)
    clone_root = get_clone_root(target_dir).resolve()
    clone_path = (clone_root / owner / repo).resolve()

    # 只允许操作 clone 根目录下的路径，避免路径穿越造成误删。
    if clone_root not in clone_path.parents:
        raise ValueError('仓库路径不在允许的 clone 目录下')

    return clone_path



def list_cloned_repos(target_dir: str = DEFAULT_CLONE_DIR) -> CachedRepoListResult:
    """列出 clone 目录下所有有效的本地仓库。"""
    clone_root = get_clone_root(target_dir).resolve()
    if not clone_root.exists() or not clone_root.is_dir():
        return CachedRepoListResult(clone_root=str(clone_root), repos=[], total_count=0)

    repos: list[CachedRepoEntry] = []
    for owner_dir in clone_root.iterdir():
        if not owner_dir.is_dir():
            continue

        for repo_dir in owner_dir.iterdir():
            if not repo_dir.is_dir():
                continue

            git_dir = repo_dir / '.git'
            is_git_repo = git_dir.exists() and git_dir.is_dir()
            if not is_git_repo:
                continue

            repo_id = f'{owner_dir.name}/{repo_dir.name}'
            report_path = get_report_path(repo_id)
            has_report = report_path.exists()

            repos.append(
                CachedRepoEntry(
                    repo_id=repo_id,
                    owner=owner_dir.name,
                    repo=repo_dir.name,
                    local_path=str(repo_dir.resolve()),
                    is_git_repo=True,
                    last_modified=datetime.fromtimestamp(repo_dir.stat().st_mtime),
                    size_bytes=_get_directory_size(repo_dir),
                    has_report=has_report,
                    report_path=str(report_path) if has_report else None,
                )
            )

    repos.sort(key=lambda item: item.repo_id.lower())
    return CachedRepoListResult(
        clone_root=str(clone_root),
        repos=repos,
        total_count=len(repos),
    )



def remove_cloned_repo(repo_id: str, target_dir: str = DEFAULT_CLONE_DIR) -> bool:
    """删除本地已克隆仓库；成功返回 True，不存在返回 False。"""
    clone_path = get_clone_path(repo_id, target_dir)

    if not clone_path.exists():
        return False

    if not clone_path.is_dir():
        raise ValueError(f'目标路径不是目录：{clone_path}')

    try:
        shutil.rmtree(clone_path, onexc=_handle_remove_readonly)
    except TypeError:
        # 兼容旧版本 Python，退回到 onerror 回调签名。
        shutil.rmtree(clone_path, onerror=_handle_remove_readonly_legacy)
    except PermissionError as exc:
        raise PermissionError(
            f'删除失败，目录可能正在被占用或权限不足：{clone_path}'
        ) from exc
    except OSError as exc:
        raise OSError(f'删除失败：{clone_path}，原始错误：{exc}') from exc

    _cleanup_empty_parents(clone_path.parent, get_clone_root(target_dir).resolve())
    return True



def _get_directory_size(root_path: Path) -> int:
    """统计目录总大小，单位为字节。"""
    total_size = 0
    for item in root_path.rglob('*'):
        if item.is_file():
            try:
                total_size += item.stat().st_size
            except OSError:
                continue
    return total_size



def _handle_remove_readonly(func, path, exc_info) -> None:
    """处理只读文件删除失败的情况，去掉只读属性后重试。"""
    target = path if isinstance(path, Path) else Path(path)

    try:
        os.chmod(target, stat.S_IWRITE)
        func(target)
    except PermissionError as exc:
        raise PermissionError(
            f'删除失败，文件可能正在被占用：{target}'
        ) from exc



def _handle_remove_readonly_legacy(func, path, exc_info) -> None:
    """兼容 shutil.rmtree(onerror=...) 的旧版回调签名。"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except PermissionError as exc:
        raise PermissionError(
            f'删除失败，文件可能正在被占用：{path}'
        ) from exc



def _cleanup_empty_parents(start_path: Path, stop_path: Path) -> None:
    """删除仓库目录后，顺手清理空的 owner 目录，但不会删除 clone 根目录。"""
    current = start_path
    while current != stop_path:
        if not current.exists():
            current = current.parent
            continue

        try:
            next(current.iterdir())
            break
        except StopIteration:
            current.rmdir()
            current = current.parent
