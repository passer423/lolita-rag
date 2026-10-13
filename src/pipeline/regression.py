"""回归对比的逻辑（脚本只负责参数与输出，逻辑放这里才可单测）。

为什么需要回归：这个项目的主要资产是「一套能发现问题的评测」，但此前每次都得人工跑
`pytest` 和 `eval.py`、再人眼看数字。**这个做法已经出过事**——第二轮优化里拒答从
1.000 掉到 0.750，是靠人肉比对才发现的。所以把「掉指标就报错」固化成代码。

一个必须说清的限制：**真实语料的回归只能在本地跑**。`data/`（语料与索引）不入库，
CI 上只有 `eval/qa_mock.jsonl` 能跑，所以 CI 覆盖的是「代码没坏」，
真正衡量检索质量的那条线得靠本机 `python scripts/regression.py`。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, Settings, settings as default_settings
from src.pipeline.query_log import settings_fingerprint
from src.pipeline.runtime import load_retriever

BASELINE_PATH = PROJECT_ROOT / "eval" / "baseline.json"

# 允许的抖动：指标是小数，浮点误差与语料微调会让它动一点，不该因此判回退。
TOLERANCE = 0.005

# 越小越好的指标（目前没有，保留扩展位）
LOWER_IS_BETTER: frozenset[str] = frozenset()

_PROGRESS_RE = re.compile(r"^([.FEsx]+)(?:\s*\[\s*\d+%\])?$")
_SUMMARY_RE = re.compile(r"(\d+)\s+passed")


def parse_counts(output: str) -> tuple[int, int]:
    """从 pytest 输出里数出 ``(通过, 失败)``。

    两个坑，实测都踩过：

    1. **这个环境里 pytest 的摘要行（``213 passed in 12.3s``）不会出现在捕获到的输出里**，
       所以不能只依赖它；
    2. 退化为「数所有 ``. F E s x`` 字符」是**错的**——警告文本里的路径与英文单词
       （``sentence_transformers`` 之类）会被算进去，实测把 213 报成了 **254**。

    所以：优先用摘要行；没有摘要行时，**只在严格匹配进度行**的行上数
    （进度行形如 ``....F...  [ 37%]``）。
    """
    for line in output.splitlines():
        match = _SUMMARY_RE.search(line)
        if match:
            failed = int(re.search(r"(\d+)\s+failed", line).group(1)) if " failed" in line else 0
            return int(match.group(1)), failed

    passed = failed = 0
    for line in output.splitlines():
        match = _PROGRESS_RE.match(line.strip())
        if not match:
            continue
        marks = match.group(1)
        passed += marks.count(".")
        failed += marks.count("F") + marks.count("E")
    return passed, failed


def run_tests(*, cwd: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    passed, failed = parse_counts((proc.stdout or "") + (proc.stderr or ""))
    return {"passed": passed, "failed": failed, "returncode": proc.returncode}


def run_eval(eval_path: str | Path, *, cwd: Path | None = None) -> dict[str, Any]:
    """子进程跑 ``scripts/eval.py --json``：隔离干净，也避免在本进程重复加载 BGE。"""
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval.py"),
            "--eval-path",
            str(eval_path),
            "--json",
        ],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"评测失败（退出码 {proc.returncode}）：{(proc.stderr or '')[-400:]}")
    return json.loads(proc.stdout)


def _portable(path: str | Path) -> str:
    """存进基线的路径必须与机器无关，否则这份文件一提交就绑死了某个人的目录。"""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def measure(
    eval_path: str | Path,
    *,
    settings: Settings | None = None,
    skip_tests: bool = False,
) -> dict[str, Any]:
    settings = settings or default_settings
    bundle = load_retriever(settings)          # 顺便拿到指纹（语料规模 / 模型）
    report = run_eval(eval_path)
    refusal = report.get("refusal_accuracy")
    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "eval_path": _portable(eval_path),
        "fingerprint": settings_fingerprint(settings, bundle),
        "metrics": {
            "recall_at_k": round(float(report["recall_at_k"]), 4),
            "mrr": round(float(report["mrr"]), 4),
            "hit_rate": round(float(report["hit_rate"]), 4),
            "refusal_accuracy": None if refusal is None else round(float(refusal), 4),
        },
        "counts": {
            "total": report["total"],
            "scored": report["scored"],
            "refusal_total": report["refusal_total"],
        },
        "tests": {"passed": 0, "failed": 0, "returncode": 0} if skip_tests else run_tests(),
    }


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> tuple[list[str], list[str]]:
    """返回 ``(回退项, 提示项)``。

    回退项 → 退出码 1；提示项（例如指纹变了、两组数字不可比）只提示，不算失败。
    """
    regressions: list[str] = []
    notes: list[str] = []

    failed = int((current.get("tests") or {}).get("failed") or 0)
    if failed:
        regressions.append(f"测试失败 {failed} 个")

    if current.get("fingerprint") != baseline.get("fingerprint"):
        notes.append(
            "参数/语料指纹与基线不同，下面的对比仅供参考："
            f"{baseline.get('fingerprint')} → {current.get('fingerprint')}"
        )

    for key, value in (current.get("metrics") or {}).items():
        base = (baseline.get("metrics") or {}).get(key)
        if value is None or base is None:
            continue
        delta = float(value) - float(base)
        worse = delta < -TOLERANCE if key not in LOWER_IS_BETTER else delta > TOLERANCE
        if worse:
            regressions.append(f"{key} 从 {base} 降到 {value}（{delta:+.4f}）")
    return regressions, notes


def render(current: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    fingerprint = current["fingerprint"]
    tests = current.get("tests") or {}
    lines = [
        f"评测集        : {current['eval_path']}",
        f"指纹          : {fingerprint['chunks']} 片段 / {fingerprint['embedder']} / "
        f"阈值 {fingerprint['min_vector_score']} / {fingerprint['fusion_mode']}",
        f"测试          : 通过 {tests.get('passed', 0)}，失败 {tests.get('failed', 0)}",
        "",
        f"{'指标':<18}{'当前':>9}{'基线':>9}{'变化':>10}",
    ]
    for key, value in (current.get("metrics") or {}).items():
        base = None if baseline is None else (baseline.get("metrics") or {}).get(key)
        base_text = "—" if base is None else f"{float(base):.3f}"
        if value is None:
            lines.append(f"{key:<18}{'—':>9}{base_text:>9}{'':>10}")
            continue
        delta = "" if base is None else f"{float(value) - float(base):+.4f}"
        lines.append(f"{key:<18}{float(value):>9.3f}{base_text:>9}{delta:>10}")
    return "\n".join(lines)


def load_baseline(path: Path | str = BASELINE_PATH) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_baseline(payload: dict[str, Any], path: Path | str = BASELINE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 显式 newline="\n"：Windows 上 Path.write_text 默认会把 \n 翻成 CRLF，
    # 而这个文件**要进版本库**——留着 CRLF 会让每次 git 操作都刷一条
    # 「CRLF will be replaced by LF」警告（实测如此）。
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path
