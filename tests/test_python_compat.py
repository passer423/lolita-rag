"""跨 Python 版本的语法兼容性检查。

## 为什么需要这个文件

本机开发环境是 **Python 3.13**，而 CI 跑 **3.11**。3.12 通过 PEP 701 放开了 f-string
的书写限制，于是存在一类**在本机永远发现不了、在 CI 上直接语法错误**的写法。

实测踩过（这个仓库真实发生过）：

    f"[{index}] {re.sub(r'\\n+', ' ', hit.chunk.text.strip())}"

表达式里带反斜杠，3.11 及以前是 ``SyntaxError: f-string expression part cannot
include a backslash``。后果不只是「一个测试挂了」——它是 ``src/generator/llm.py`` 的
导入期错误，于是**所有测试模块收集失败**，pytest 报的是**退出码 2（收集失败）**，
看起来完全不像语法问题。而本机 3.13 跑得好好的。

CI 是唯一能发现它的地方，但等 CI 反馈一轮要几分钟。所以这里把它变成**本机就能跑的
静态检查**：用 tokenizer 扫描所有源码里的 f-string，找出 3.11 不接受的两类写法。

静态检查一定是**启发式**的（真正的权威是解释器本身），所以下面既有正向对照
（必须能抓出真实踩过的写法），也有反向对照（合法写法不能误报）。
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

SOURCE_DIRS = ("src", "scripts", "tests")

# Python 3.12 以下没有 FSTRING_* 系列 token，本模块的静态检查无从实现
# （那些版本上解释器自己就会拒绝这类源码）。所以整组测试要**跳过**而不是失败——
# 实测踩过：没加这个 skip 时，CI（3.11）上两条正向对照反而挂了。
requires_fstring_tokens = pytest.mark.skipif(
    not hasattr(tokenize, "FSTRING_START"),
    reason="Python 3.12 以下没有 FSTRING_* token，无法做这项静态检查",
)


def fstring_hazards(source: str) -> list[tuple[int, str]]:
    """返回 ``(行号, 原因)``：在 Python 3.11 及以前会语法错误的 f-string 写法。

    Python 3.12 以下没有 ``FSTRING_START`` 这些 token 常量，此时直接返回空
    ——在那些版本上解释器自己就会拒绝加载源码，不需要（也无法）做这项静态检查。
    """
    if not hasattr(tokenize, "FSTRING_START"):
        return []

    hazards: list[tuple[int, str]] = []
    quotes: list[str] = []          # 每层 f-string 的引号字符
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []

    for token in tokens:
        if token.type == tokenize.FSTRING_START:
            quote = token.string[-1]
            if quotes and quote == quotes[-1]:
                hazards.append(
                    (token.start[0], "嵌套 f-string 复用了相同的引号（PEP 701，Python 3.12+）")
                )
            quotes.append(quote)
            continue
        if token.type == tokenize.FSTRING_END:
            if quotes:
                quotes.pop()
            continue
        if not quotes:
            continue
        # 以下都在某个 f-string 的表达式区域里
        if token.type == tokenize.STRING:
            if "\\" in token.string:
                hazards.append(
                    (token.start[0], "f-string 表达式里的字符串含反斜杠（3.11 及以前是语法错误）")
                )
            if token.string and token.string[-1] == quotes[-1]:
                hazards.append(
                    (token.start[0], "表达式里复用了与外层 f-string 相同的引号（PEP 701，3.12+）")
                )
    return hazards


# ------------------------------------------------------------------ 检测器自检

@requires_fstring_tokens
def test_detects_the_exact_pattern_that_broke_ci():
    """正向对照：必须抓出真实踩过的那种写法，否则这个测试只是永远绿灯的摆设。"""
    bad = 'lines = [f"[{i}] {re.sub(r\'\\n+\', \' \', text)}" for i, text in xs]\n'
    hazards = fstring_hazards(bad)
    assert hazards, "检测器没能抓出「表达式里含反斜杠」"
    assert "反斜杠" in hazards[0][1]


@requires_fstring_tokens
def test_detects_nested_fstring_reusing_the_same_quote():
    bad = 'x = f"{f"{value}"}"\n'
    hazards = fstring_hazards(bad)
    assert hazards and "相同的引号" in hazards[0][1]


@requires_fstring_tokens
def test_accepts_valid_311_fstrings():
    """反向对照：合法写法一个都不能误报（误报会让人开始忽略这个检查）。"""
    good = "\n".join(
        [
            'a = f"{name}"',
            'b = f"{name}={value!r}"',
            'c = f"{x:>10.2f}"',
            'd = f"{a}{b}{c}"',
            'e = f"{price} 元，共 {count} 件"',
            "f = f'{name}'",              # 单引号 f-string
            'g = f"外层 {f\'内层\':>3}"',  # 嵌套但引号不同（3.11 合法）
            'h = "普通字符串里的 \\n 反斜杠"',   # 不在 f-string 表达式里
            'i = re.sub(r"\\n+", " ", text)',  # 普通调用
        ]
    )
    assert fstring_hazards(good) == []


# ------------------------------------------------------------------ 全仓扫描

@requires_fstring_tokens
def test_repository_has_no_fstring_syntax_hazards():
    problems: list[str] = []
    for folder in SOURCE_DIRS:
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            for line, reason in fstring_hazards(path.read_text(encoding="utf-8")):
                relative = path.relative_to(PROJECT_ROOT)
                problems.append(f"  {relative}:{line} {reason}")

    assert not problems, (
        "发现只在 Python 3.12+ 合法的 f-string 写法，CI（3.11）会直接语法错误：\n"
        + "\n".join(problems)
        + "\n改法：先把表达式算进变量，再插值。"
    )
