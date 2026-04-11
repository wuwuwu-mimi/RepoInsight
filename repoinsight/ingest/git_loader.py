import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

from rich import print

from repoinsight.utils.check_util import check_url

# 判定为可重试的 Git 网络错误关键字。
RETRYABLE_GIT_ERROR_KEYWORDS = (
    'timed out',
    'failed to connect',
    'could not resolve host',
    'connection reset',
    'network is unreachable',
    'operation timed out',
    'early eof',
    'connection was reset',
    'unable to access',
)


async def clone_repo(
    url: str,
    target_dir: str = 'clone',
    max_retries: int = 3,
) -> str | None:
    """克隆 GitHub 仓库，返回本地仓库目录绝对路径。"""
    if not check_url(url):
        print('[red]克隆失败：不是合法的 GitHub 仓库地址[/red]')
        return None

    try:
        normalized_url, owner, repo = _normalize_repo_url(url)
    except ValueError as exc:
        print(f'[red]克隆失败：{exc}[/red]')
        return None

    clone_url = _build_clone_url(normalized_url)
    final_clone_path = _build_clone_path(target_dir=target_dir, owner=owner, repo=repo)
    final_clone_path.parent.mkdir(parents=True, exist_ok=True)

    # 已存在可复用仓库时直接返回，避免重复克隆失败。
    if (final_clone_path / '.git').exists():
        print(f'[yellow]仓库已存在，复用本地目录[/yellow]：{final_clone_path}')
        return str(final_clone_path)

    # 目标目录已存在但不是 git 仓库时，避免直接覆盖用户文件。
    if final_clone_path.exists() and any(final_clone_path.iterdir()):
        print(f'[red]克隆失败：目标目录已存在且不是 Git 仓库：{final_clone_path}[/red]')
        return None

    cmd = [
        'git',
        'clone',
        '--depth',
        '1',
        '--single-branch',  # 首版只拉默认分支，减少网络传输量。
        clone_url,
        str(final_clone_path),
    ]

    for attempt in range(1, max_retries + 1):
        success, error_message = await _run_clone_command(cmd)
        if success:
            print(f'[green]克隆成功[/green]：{final_clone_path}')
            return str(final_clone_path)

        error_type = _classify_git_error(error_message)
        is_last_attempt = attempt == max_retries

        if error_type == 'network' and not is_last_attempt:
            wait_seconds = attempt
            print(
                f'[yellow]克隆失败，检测到网络问题，{wait_seconds} 秒后进行第 {attempt + 1} 次重试[/yellow]'
            )
            await asyncio.sleep(wait_seconds)
            continue

        if error_type == 'network':
            print(f'[red]克隆失败：网络连接异常，已重试 {max_retries} 次。原始错误：{error_message}[/red]')
        elif error_type == 'permission':
            print(f'[red]克隆失败：仓库可能为私有仓库，或当前没有访问权限。原始错误：{error_message}[/red]')
        elif error_type == 'not_found':
            print(f'[red]克隆失败：仓库不存在或地址错误。原始错误：{error_message}[/red]')
        elif error_type == 'git_missing':
            print('[red]克隆失败：本机未安装 git，或 git 未加入 PATH。[/red]')
        else:
            print(f'[red]克隆失败：{error_message}[/red]')
        return None

    return None


async def _run_clone_command(cmd: list[str]) -> tuple[bool, str]:
    """执行一次 git clone 命令，返回是否成功及错误信息。"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode == 0:
            return True, ''

        error_message = stderr.decode('utf-8', errors='replace').strip()
        return False, error_message or 'git clone 执行失败'
    except FileNotFoundError:
        return False, 'git command not found'
    except Exception as exc:
        return False, str(exc)



def _build_clone_path(target_dir: str, owner: str, repo: str) -> Path:
    """构造仓库本地克隆路径，统一落在项目根目录下。"""
    project_root = Path(__file__).resolve().parents[2]
    clone_root = project_root / target_dir
    return clone_root / owner / repo



def _build_clone_url(normalized_url: str) -> str:
    """根据环境变量构造最终克隆地址，便于接代理或镜像。"""
    clone_base_url = os.getenv('GIT_CLONE_BASE_URL')
    if not clone_base_url:
        return normalized_url

    parsed = urlparse(normalized_url)
    clone_base_url = clone_base_url.rstrip('/')
    return f'{clone_base_url}{parsed.path}'



def _classify_git_error(error_message: str) -> str:
    """对 git clone 错误做一个轻量分类，便于给出更友好的提示。"""
    text = error_message.lower()

    if 'git command not found' in text or 'is not recognized' in text:
        return 'git_missing'

    if 'repository not found' in text or 'not found' in text:
        return 'not_found'

    if 'authentication failed' in text or 'permission denied' in text:
        return 'permission'

    if any(keyword in text for keyword in RETRYABLE_GIT_ERROR_KEYWORDS):
        return 'network'

    return 'unknown'



def _normalize_repo_url(url: str) -> tuple[str, str, str]:
    """把 GitHub 链接规范化为仓库根地址，并提取 owner / repo。"""
    parsed = urlparse(url.strip())
    if parsed.netloc not in {'github.com', 'www.github.com'}:
        raise ValueError('当前仅支持 github.com 仓库地址')

    parts = [part for part in parsed.path.split('/') if part]
    if len(parts) < 2:
        raise ValueError('仓库地址格式应为 https://github.com/<owner>/<repo>')

    owner, repo = parts[0], parts[1]
    if repo.endswith('.git'):
        repo = repo[:-4]

    normalized_url = f'https://github.com/{owner}/{repo}.git'
    return normalized_url, owner, repo
