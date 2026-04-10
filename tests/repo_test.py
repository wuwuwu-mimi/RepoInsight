"""clone repo"""
import asyncio
import os
from rich import print

from repoinsight.utils.check_util import check_url

from dotenv import load_dotenv

load_dotenv()


async def clone_repo(url: str, target_dir: str) -> str | None:
    """
    clone repo
    :param url: repo url
    :param target_dir: clone target dir
    :return: path of cloned repo
    """
    # ========== 修复 1：只在这里取一次环境变量 ==========
    clone_target = os.getenv("CLONE_TARGET_PATH", target_dir)

    # 检查 URL
    if not check_url(url):
        print("[red]❌ URL 不合法[/red]")
        return None
    url = to_proxy(url)
    # ========== 修复 2：正确拼接路径 ==========
    base_dir = os.path.dirname(os.path.abspath(__file__))
    final_clone_path = os.path.abspath(os.path.join(base_dir, clone_target))

    # ========== 修复 3：git 克隆命令 ==========
    cmd = [
        "git", "clone",
        "--depth", "1",
        url,
        final_clone_path
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            err = stderr.decode("utf-8", errors="ignore").strip()
            print(f"[red]❌ 克隆失败：{err}[/red]")
            return None

        print(f"[green]✅ 克隆成功：[/green]{final_clone_path}")
        return final_clone_path  # 成功一定返回路径

    except Exception as e:
        print(f"[red]❌ 克隆异常：{str(e)}[/red]")
        return None


async def main():
    path = await clone_repo(
        url="https://github.com/octocat/Hello-World.git",
        target_dir="./clone"
    )
    print("最终路径：", path)


def to_proxy(url: str) -> str | None:
    """
        自动给 GitHub URL 加上 xget 代理

        """
    if "github.com" in url:
        # xget 官方加速格式
        return url.replace(
            "https://github.com/",
            "https://xget.xi-xu.me/gh/"
        )
    return url


if __name__ == "__main__":
    asyncio.run(main())
