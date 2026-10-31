"""提交前检查：拦住个人文件与密钥进入仓库。

## 为什么需要这个

一次 `git add -A` 把工作区里的**个人简历**（PDF 与 HTML 备份）一起提交进了公开仓库，
而且过了一段时间才发现。删除文件**不等于**删除历史：它仍然能从旧提交里取到，
唯一办法是改写历史 + 强推——代价是所有 SHA 变化、别人的克隆全部失效。

所以与其事后补救，不如**提交前就拦住**。这个脚本有两道用法：

    python scripts/check_staged.py             # 检查暂存区（pre-commit 钩子调用）
    python scripts/check_staged.py --tracked   # 检查 HEAD 里的全部文件（CI 调用）

第二道是为了兜住"有人用 `git commit --no-verify` 绕过钩子"的情况。

## 放行方式

* 确有必要提交的二进制（如文档截图）写进 ``config/commit_allowlist.txt``（支持通配符）；
* 单次应急：``git commit --no-verify``——但 CI 里的 ``--tracked`` 检查仍会拦下来。

退出码：``0`` 通过 / ``2`` 拦下 / ``1`` 环境问题（不是 git 仓库等，只警告不拦）。
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_NAME = Path("config") / "commit_allowlist.txt"

# 个人文件/办公文档：几乎不可能属于代码项目
PERSONAL_EXT = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pages", ".key", ".numbers", ".odt", ".ods",
    ".heic", ".mp4", ".mov", ".avi", ".zip", ".rar", ".7z",
}
# 图片：文档截图是合理的，但要显式进白名单——不然手机相册会被 add 进来
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".psd", ".svg"}
# 密钥/证书
SECRET_EXT = {".pem", ".p12", ".pfx", ".keystore", ".jks", ".ppk"}
# 文件名里出现这些词，基本就是个人材料
PERSONAL_NAME = (
    "简历", "求职", "resume", "curriculum", "个人简历",
    "身份证", "户口", "护照", "学位证", "毕业证", "成绩单",
    "薪资", "工资单", "劳动合同", "离职证明", "体检",
)
# 密钥/凭据文件名（.env.example 是模板，必须放行）
SECRET_NAME = (".env", ".har", "id_rsa", "id_ed25519", "credentials", "secret", ".npmrc", ".pypirc")
SECRET_SAFE = (".env.example", ".env.sample", ".env.template")
# 单个文件大于该体积就提示（本仓库全部文件加起来不到 1 MB）
LARGE_FILE_BYTES = 3 * 1024 * 1024


def repo_root() -> Path:
    """定位**运行时所在的**仓库根，而不是脚本文件所在的仓库根。

    这个区别是实测踩出来的：最初用 ``Path(__file__).parents[1]``，于是把钩子装到
    别的仓库、或在临时仓库里跑测试时，它仍然去查自己那个仓库——检查结果完全无效。
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=False,
    )
    text = (result.stdout or "").strip()
    if result.returncode == 0 and text:
        return Path(text)
    return Path.cwd()


def _git(root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        # 中文文件名要按 UTF-8 解；surrogateescape 保证非 UTF-8 名字也不炸
        encoding="utf-8",
        errors="surrogateescape",
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def list_staged(root: Path) -> list[str]:
    """暂存区里新增/修改/重命名/复制的文件（-z 避免空格与特殊字符被转义）。"""
    raw = _git(root, ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"])
    return [path for path in raw.split("\0") if path]


def list_tracked(root: Path) -> list[str]:
    """HEAD 里的全部文件（CI 用）。"""
    raw = _git(root, ["ls-tree", "-r", "-z", "--name-only", "HEAD"])
    return [path for path in raw.split("\0") if path]


def allowed_patterns(root: Path) -> list[str]:
    allowlist = root / ALLOWLIST_NAME
    if not allowlist.exists():
        return []
    lines = allowlist.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def pattern_variants(pattern: str) -> list[str]:
    """把 ``docs/**/*.png`` 这类写法展开成 fnmatch 能匹配的形式。

    ``fnmatch`` 里的 ``*`` 会跨越 ``/``，但 ``**/`` 这种"零个或多个目录"的语义它不懂：
    实测 ``fnmatch("docs/screenshot.png", "docs/**/*.png")`` 是 **False**（因为模式
    要求 ``**`` 后面还有一个 ``/``）。白名单是写给人看的，必须支持这种直觉写法。
    """
    variants = {pattern}
    for candidate in (pattern.replace("**/", ""), pattern.replace("**", "*")):
        variants.add(candidate)
    return sorted(variants)


def is_allowed(path: str, patterns: list[str]) -> bool:
    lowered = path.lower()
    for pattern in patterns:
        for variant in pattern_variants(pattern):
            if fnmatch.fnmatch(lowered, variant.lower()):
                return True
    return False


def file_size(root: Path, path: str) -> int:
    try:
        return (root / path).stat().st_size
    except OSError:
        return 0


def suspicious_reason(root: Path, path: str) -> str | None:
    """返回拦截原因；``None`` 表示放行。"""
    name = Path(path).name
    lowered = name.lower()
    suffix = Path(name).suffix.lower()

    if lowered in SECRET_SAFE:
        return None
    # .env / .env.local / .env.production 等都算，但模板除外（上面已放行）
    if lowered == ".env" or lowered.startswith(".env.") or lowered.endswith(".har"):
        return "凭据/抓包文件（.env、.har 里通常是 Cookie 与密钥）"
    if any(marker in lowered for marker in SECRET_NAME):
        return "疑似凭据文件"
    if suffix in SECRET_EXT:
        return "私钥/证书"
    if suffix in PERSONAL_EXT:
        return f"个人文件或办公文档（{suffix}）"
    if any(word in name for word in PERSONAL_NAME):
        return "文件名像个人材料（简历/证件/薪资等）"
    if suffix in IMAGE_EXT:
        return f"图片（{suffix}）：文档截图请写进 config/commit_allowlist.txt"
    size = file_size(root, path)
    if size > LARGE_FILE_BYTES:
        return f"体积异常（{size / 1024 / 1024:.1f} MB）"
    return None


def main(argv: list[str] | None = None) -> int:
    # 这个脚本会被 git 钩子调用，而 Windows 上钩子的 stdout 是 cp936(GBK)：
    # 一个 emoji 就能让它抛 UnicodeEncodeError，**从"通过"变成"拦住提交"**。
    # 实测踩过：脚本本身写得对，却因为打印 "✅" 导致每次提交都被拦。
    # 所以：① 输出只用 GBK 能编码的字符；② 这里再加一层，任何不可编码字符都降级而非抛错。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):       # pragma: no cover - 非 TTY/被重定向
            pass

    parser = argparse.ArgumentParser(description="提交前检查：拦住个人文件与密钥")
    parser.add_argument(
        "--tracked", action="store_true", help="检查 HEAD 里全部文件（CI 用，兜住 --no-verify）"
    )
    parser.add_argument("--quiet", action="store_true", help="通过时不输出")
    args = parser.parse_args(argv)

    root = repo_root()
    if not _git(root, ["rev-parse", "--git-dir"]).strip():
        print("[check_staged] 不是 git 仓库，跳过检查", file=sys.stderr)
        return 0

    files = list_tracked(root) if args.tracked else list_staged(root)
    patterns = allowed_patterns(root)
    offenders: list[tuple[str, str]] = []
    for path in files:
        if is_allowed(path, patterns):
            continue
        reason = suspicious_reason(root, path)
        if reason:
            offenders.append((path, reason))

    scope = "HEAD 里的文件" if args.tracked else "暂存区"
    if not offenders:
        if not args.quiet:
            print(f"[check_staged] {scope}共 {len(files)} 个文件，未发现个人文件/密钥 [OK]")
        return 0

    print(f"\n[check_staged] 拦下 {len(offenders)} 个不该进仓库的文件（{scope}）：\n", file=sys.stderr)
    for path, reason in offenders:
        print(f"  [X] {path}\n      原因：{reason}", file=sys.stderr)
    print(
        "\n处理方式：\n"
        "  · 大多数情况：`git rm --cached <文件>` 把它移出暂存区（文件仍留在本地）\n"
        "  · 确实需要提交（如文档截图）：把它写进 config/commit_allowlist.txt\n"
        "  · 应急放行：`git commit --no-verify`（但 CI 的 --tracked 检查仍会拦）\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
