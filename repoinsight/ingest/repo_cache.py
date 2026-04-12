import os
import shutil
import stat
import json
from datetime import datetime
from pathlib import Path

from repoinsight.models.cache_model import CachedRepoEntry, CachedRepoListResult
from repoinsight.report.markdown_report import get_report_path


# clone 缓存固定存放在项目根目录下的 clone 目录。
DEFAULT_CLONE_DIR = 'clone'

# 本地知识库默认目录。
DEFAULT_KNOWLEDGE_DIR = 'data/knowledge'



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
    """列出当前项目中可追踪到的仓库及其资产状态。"""
    clone_root = get_clone_root(target_dir).resolve()
    repo_assets = _collect_repo_assets(clone_root=clone_root)
    if not repo_assets:
        return CachedRepoListResult(clone_root=str(clone_root), repos=[], total_count=0)

    repos: list[CachedRepoEntry] = []
    for repo_id in sorted(repo_assets.keys(), key=str.lower):
        owner, repo = parse_repo_id(repo_id)
        assets = repo_assets[repo_id]
        report_path = get_report_path(repo_id)
        has_markdown_report = assets['has_markdown_report']
        repos.append(
            CachedRepoEntry(
                repo_id=repo_id,
                owner=owner,
                repo=repo,
                local_path=assets['local_path'],
                has_clone=assets['has_clone'],
                is_git_repo=assets['is_git_repo'],
                last_modified=assets['last_modified'],
                size_bytes=assets['size_bytes'],
                has_report=has_markdown_report,
                report_path=str(report_path) if has_markdown_report else None,
                has_markdown_report=has_markdown_report,
                has_json_report=assets['has_json_report'],
                has_llm_context=assets['has_llm_context'],
                has_knowledge=assets['has_knowledge'],
                has_vector_index=assets['has_vector_index'],
                asset_status=_build_asset_status(assets),
            )
        )

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


def _collect_repo_assets(clone_root: Path) -> dict[str, dict[str, object]]:
    """收集仓库在 clone、报告、知识库和向量库中的存在状态。"""
    repo_assets: dict[str, dict[str, object]] = {}

    if clone_root.exists() and clone_root.is_dir():
        for owner_dir in clone_root.iterdir():
            if not owner_dir.is_dir():
                continue

            for repo_dir in owner_dir.iterdir():
                if not repo_dir.is_dir():
                    continue

                repo_id = f'{owner_dir.name}/{repo_dir.name}'
                git_dir = repo_dir / '.git'
                is_git_repo = git_dir.exists() and git_dir.is_dir()
                if not is_git_repo:
                    continue

                asset = _get_or_create_repo_asset(repo_assets, repo_id)
                asset['has_clone'] = True
                asset['is_git_repo'] = True
                asset['local_path'] = str(repo_dir.resolve())
                asset['last_modified'] = datetime.fromtimestamp(repo_dir.stat().st_mtime)
                asset['size_bytes'] = _get_directory_size(repo_dir)

    report_dir = get_report_path('placeholder/repo').parent
    if report_dir.exists():
        for file_path in report_dir.iterdir():
            if not file_path.is_file():
                continue
            repo_id = _parse_repo_id_from_report_name(file_path.name)
            if repo_id is None:
                continue

            asset = _get_or_create_repo_asset(repo_assets, repo_id)
            suffixes = file_path.suffixes
            if file_path.suffix == '.md':
                asset['has_markdown_report'] = True
            elif file_path.suffix == '.json':
                asset['has_json_report'] = True
            elif suffixes[-2:] == ['.llm', '.txt'] or file_path.name.endswith('.llm.txt'):
                asset['has_llm_context'] = True

    knowledge_root = get_project_root() / DEFAULT_KNOWLEDGE_DIR
    if knowledge_root.exists():
        for file_path in knowledge_root.rglob('*.json'):
            repo_id = _read_repo_id_from_knowledge_file(file_path)
            if repo_id is None:
                continue
            asset = _get_or_create_repo_asset(repo_assets, repo_id)
            asset['has_knowledge'] = True

    try:
        from repoinsight.storage.chroma_store import list_repo_ids_in_chroma

        vector_repo_ids = list_repo_ids_in_chroma()
    except Exception:
        vector_repo_ids = set()

    for repo_id in vector_repo_ids:
        asset = _get_or_create_repo_asset(repo_assets, repo_id)
        asset['has_vector_index'] = True

    return repo_assets


def _get_or_create_repo_asset(
    repo_assets: dict[str, dict[str, object]],
    repo_id: str,
) -> dict[str, object]:
    """获取或初始化某个仓库的资产状态。"""
    if repo_id not in repo_assets:
        repo_assets[repo_id] = {
            'has_clone': False,
            'is_git_repo': False,
            'local_path': '',
            'last_modified': None,
            'size_bytes': None,
            'has_markdown_report': False,
            'has_json_report': False,
            'has_llm_context': False,
            'has_knowledge': False,
            'has_vector_index': False,
        }
    return repo_assets[repo_id]


def _parse_repo_id_from_report_name(file_name: str) -> str | None:
    """从报告文件名中恢复 owner/repo。"""
    base_name = file_name
    for suffix in ('.llm.txt', '.json', '.md'):
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break

    if '__' not in base_name:
        return None

    owner, repo = base_name.split('__', maxsplit=1)
    if not owner or not repo:
        return None
    return f'{owner}/{repo}'


def _read_repo_id_from_knowledge_file(file_path: Path) -> str | None:
    """从本地知识库文件中读取 repo_id。"""
    try:
        payload = json.loads(file_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None

    repo_id = payload.get('repo_id')
    if not isinstance(repo_id, str) or not repo_id.strip():
        return None
    return repo_id.strip()


def _build_asset_status(assets: dict[str, object]) -> str:
    """根据资产布尔状态生成适合列表展示的摘要状态。"""
    has_clone = bool(assets.get('has_clone'))
    has_reports = bool(
        assets.get('has_markdown_report')
        or assets.get('has_json_report')
        or assets.get('has_llm_context')
    )
    has_knowledge = bool(assets.get('has_knowledge'))
    has_vector_index = bool(assets.get('has_vector_index'))

    if has_clone and (has_reports or has_knowledge or has_vector_index):
        return '完整'
    if has_clone:
        return '仅 Clone'
    if has_vector_index and (has_reports or has_knowledge):
        return '孤儿索引+资产'
    if has_vector_index:
        return '仅索引残留'
    if has_knowledge and has_reports:
        return '孤儿知识+报告'
    if has_knowledge:
        return '仅知识残留'
    if has_reports:
        return '仅报告残留'
    return '空'



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
