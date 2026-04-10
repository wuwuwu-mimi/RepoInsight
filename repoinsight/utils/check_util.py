import re

def check_url(url: str) -> bool:
    """检查是不是 GitHub URL"""
    if not isinstance(url, str) or len(url.strip()) == 0:
        return False


    github_pattern = re.compile(
        r'^https?://(www\.)?github\.com/[\w\-]+/[\w\-.]+(/.*)?$',
        re.IGNORECASE
    )
    return bool(github_pattern.match(url.strip()))