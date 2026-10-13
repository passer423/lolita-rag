"""前端（单页 HTML）的测试。

前端一直是「改完只能靠肉眼开浏览器看」的部分——这里尽量把它拉进自动化：

* **语法**：把内联 `<script>` 抽出来交给 node 做 `--check`（有 node 才跑，否则跳过）；
* **契约**：页面里必须有引用锚点、来源清单、运行时闸门提示这些结构；
* **安全**：答案里的 `[n]` 是**自己拼 HTML** 的，必须确认拼之前对模型输出做了转义——

最后一条最要紧：把模型输出直接塞进 `innerHTML` 就是 XSS，而这条路径正好是
「模型生成的内容 → 页面」，是典型的高风险位。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

INDEX_HTML = PROJECT_ROOT / "src" / "web" / "static" / "index.html"

# load_workspace_dependencies 给出的捆绑 node 路径（不在 PATH 上时用它）
BUNDLED_NODE = Path(
    r"C:\Users\50436\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"
)


def read_page() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def inline_script() -> str:
    blocks = re.findall(r"<script>(.*?)</script>", read_page(), re.S)
    assert blocks, "页面里应当有内联脚本"
    return blocks[-1]


def find_node() -> str | None:
    return shutil.which("node") or (str(BUNDLED_NODE) if BUNDLED_NODE.exists() else None)


# ------------------------------------------------------------------ 语法

def test_inline_script_is_syntactically_valid(tmp_path):
    node = find_node()
    if not node:
        pytest.skip("本机没有 node，跳过 JS 语法检查")

    script = tmp_path / "page.js"
    script.write_text(inline_script(), encoding="utf-8")
    proc = subprocess.run(
        [node, "--check", str(script)], capture_output=True, text=True, encoding="utf-8"
    )
    assert proc.returncode == 0, f"前端脚本语法错误：{proc.stderr}"


# ------------------------------------------------------------------ 契约

def test_answer_rendering_links_citations_to_source_cards():
    """「每条结论都能自己核对」要成立，[n] 必须真的能跳到来源卡片。"""
    page = read_page()
    assert "function linkCitations(" in page, "缺少引用锚点渲染"
    assert "function renderSources(" in page, "缺少来源清单渲染"
    assert 'class="cite"' in page
    assert 'id="hit-' in page, "来源卡片必须带锚点 id，否则 [n] 跳不过去"
    assert "a.cite" in page, "缺少引用样式"


def test_answer_shows_runtime_gate_verdict():
    """答案被降级过就必须显示出来，不能装作什么都没发生。"""
    page = read_page()
    assert "answer.degraded" in page
    assert "未通过引用校验" in page
    assert "answer.verified === false" in page, "要能区分「没检查」与「检查通过」"


def test_render_answer_receives_hits_for_source_labels():
    """来源清单要显示作者与时间，就必须把 hits 传进渲染函数。"""
    page = read_page()
    assert re.search(r"renderAnswer\(data\.answer,\s*generate,\s*data\.hits\)", page), (
        "调用处没有把 hits 传给 renderAnswer，来源清单只能显示裸链接"
    )


# ------------------------------------------------------------------ 安全

def test_model_output_is_escaped_before_being_placed_in_innerhtml():
    """拼 HTML 之前必须转义：模型输出直接进 innerHTML 就是 XSS。"""
    body = inline_script()
    start = body.index("function linkCitations(")
    end = body.index("function renderSources(")
    linker = body[start:end]

    assert "escapeHtml(" in linker, "引用渲染里必须转义模型输出"
    # 只允许插入自己生成的 <a>，不能把原文拼进去
    assert linker.count("'<a class=\"cite\"") == 1
    assert linker.count("source.slice(") >= 2, "应当按引用编号切分后逐段转义"


def test_no_external_resource_is_referenced():
    """离线可用：不能有 CDN 依赖（与 test_web.py 的接口测试配套）。"""
    page = read_page()
    assert "cdn." not in page.lower()
    assert "<script src=" not in page
