"""clone repo"""
import asyncio
import os
from rich import print

from repoinsight.utils.check_util import check_url


async def clone_repo(url: str, target_dir: str) -> str | None:
    """
    clone repo
    :param url: repo url
    :param target_dir: clone target dir
    :return: path of cloned repo
    """
    clone_target = os.getenv("CLONE_TARGET_PATH") or target_dir
    """clone repo"""
    if not check_url(url):
        return None
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        final_clone_path = os.path.abspath(os.path.join(base_dir, clone_target))
        # create clone cmd
        cmd = [
            "git", "clone",
            "--depth", "1",  # TODO 以后可选深克隆
            url,
            final_clone_path
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, )

            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                err = stderr.decode("utf-8").strip()
                print(f"[red]克隆失败:{err} [/red]")
                return None
            
            print(f"[green]克隆成功[/green]：{final_clone_path}")
        except Exception as e:
            print(f"[red]克隆失败:未知异常[/red]")
            return None
    return final_clone_path


async def main():
    path = await clone_repo(
        url="https://github.com/octocat/Hello-World.git",
        target_dir="./clone"
    )
    print("最终路径：", path)


if __name__ == "__main__":
    asyncio.run(main())
