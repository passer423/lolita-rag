"""提交前检查的测试。

背景：一次 `git add -A` 把工作区里的个人简历（PDF + HTML 备份）提交进了公开仓库，
过了一段时间才发现。删除文件不等于删除历史——只能改写历史 + 强推，代价是所有
SHA 变化。所以这道闸门要能被信任：**该拦的必须拦住，不该拦的绝不能误伤**。

两个方向都测（每个拦截规则都配一个放行对照），因为"测试只验证拦得住"很容易写成
"拦一切"也能通过。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_staged.py"


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """一个最小的真实 git 仓库（脚本依赖 git 命令，不能只用假数据）。"""
    repo = tmp_path / "project"
    (repo / "config").mkdir(parents=True)
    (repo / "scripts").mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text("print('hi')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def stage(repo: Path, name: str, content: str = "x") -> None:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(repo, "add", name)


# ---------------------------------------------------------------- 该拦的

@pytest.mark.parametrize(
    "name",
    [
        "尹丹-前端开发工程师.pdf",      # 真实踩过的那次
        "resume.html",                  # 文件名命中 resume
        "个人简历.md",                   # 中文关键词
        "身份证扫描.pdf",
        ".env",                         # 凭据
        "cookie.har",                   # 抓包（含完整 Cookie）
        "server.pem",                   # 私钥
        "photo.jpg",                    # 图片默认拒绝
        "薪资流水.xlsx",
    ],
)
def test_blocks_personal_and_secret_files(repo, name):
    stage(repo, name)

    result = run(repo)

    assert result.returncode == 2, result.stdout + result.stderr
    assert name in result.stderr
    assert "拦下" in result.stderr


def test_blocks_large_file(repo):
    stage(repo, "big.txt", "a" * (3 * 1024 * 1024 + 10))

    result = run(repo)

    assert result.returncode == 2
    assert "体积异常" in result.stderr


# ---------------------------------------------------------------- 不该拦的

@pytest.mark.parametrize(
    "name",
    [
        "app.py",
        "README.md",
        "docs/guide.rst",
        ".env.example",          # 模板必须在白名单里放行
        "data/raw/cookie_sample.json",
        "tests/fixtures/saved_page.html",
    ],
)
def test_allows_normal_project_files(repo, name):
    stage(repo, name)

    result = run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK]" in result.stdout


def test_output_uses_gbk_safe_characters_only(repo):
    """钩子在 Windows 上拿到的 stdout 是 cp936：输出里出现 emoji 会抛
    UnicodeEncodeError，**把"通过"变成"拦住提交"**（实测踩过，每次提交都被拦）。
    这里直接断言输出能被 GBK 编码。
    """
    stage(repo, "app2.py")
    passed = run(repo)
    stage(repo, "简历.pdf")
    blocked = run(repo)

    for result in (passed, blocked):
        for text in (result.stdout, result.stderr):
            text.encode("gbk")          # 编不出来就直接失败


# ---------------------------------------------------------------- 白名单

def test_allowlist_releases_an_intentional_binary(repo):
    stage(repo, "docs/screenshot.png")
    assert run(repo).returncode == 2, "默认必须拦住图片"

    (repo / "config" / "commit_allowlist.txt").write_text("docs/**/*.png\n", encoding="utf-8")
    git(repo, "add", "config/commit_allowlist.txt")

    result = run(repo)

    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------- CI 兜底

def test_tracked_mode_catches_a_file_that_bypassed_the_hook(repo):
    """`git commit --no-verify` 能绕过钩子——所以 CI 还要能查已经进 HEAD 的文件。"""
    stage(repo, "简历.pdf")
    git(repo, "commit", "-q", "-m", "oops")     # 相当于绕过了钩子

    staged = run(repo)
    tracked = run(repo, "--tracked")

    assert staged.returncode == 0, "暂存区已空，钩子这次确实没东西可拦"
    assert tracked.returncode == 2, "但 --tracked 必须把它揪出来"
    assert "简历.pdf" in tracked.stderr


def test_clean_repo_passes(repo):
    result = run(repo, "--tracked")

    assert result.returncode == 0
    assert "未发现" in result.stdout


def test_non_git_directory_is_not_a_failure(tmp_path):
    """不在 git 仓库里只警告、不拦——否则在某些 CI 环境下会把构建搞挂。"""
    result = run(tmp_path, "--tracked")

    assert result.returncode == 0
    assert "不是 git 仓库" in result.stderr
