"""闸门消融：逐个关掉，看指标掉不掉——**掉才说明它承重**。

## 为什么要做这件事

检索层现在有 5 处判定（向量绝对下限、BM25 相对下限、生僻词锚定、关键词路的证据门槛、
核心概念缺失即拒答）。它们是我在真实数据上一条条加上去的，但**从来没有验证过哪一条是
真的承重**：如果某一条关掉后指标不动，那它就是在给系统增加复杂度而不产生价值。

这个工具就是回答「能不能删」的：在**同一份评测集**上跑 baseline 与各消融配置，比较
Recall / MRR / 拒答，并列出**具体哪些题的结果发生了变化**——只看总分容易得出
「关掉它也没事」的错觉，而真相往往是「某两道题翻车了，被别处的得分掩盖了」。

用法（评测集可以是 qa_real，也可以是 query_log 里积累出来的真实查询）：

    python scripts/ablate_gates.py
    python scripts/ablate_gates.py --eval-path eval/qa_real.jsonl --json

⚠️ 消融只改**运行期对象**（settings 字段 / 模块常量），跑完一定恢复，不落盘。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from src.config import Settings, settings as default_settings
from src.pipeline.eval import evaluate
from src.pipeline.runtime import RetrieverBundle, load_retriever

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AblationResult:
    name: str
    description: str
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    refusal_accuracy: float | None = None
    refused_cases: int = 0
    # 与 baseline 相比，结果发生变化的题目（这是最值得看的一列）
    changed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "refusal_accuracy": self.refusal_accuracy,
            "refused_cases": self.refused_cases,
            "changed": self.changed,
        }


@contextmanager
def _temporary(settings: Settings, bundle: RetrieverBundle, mutate: Callable[[], None]) -> Iterator[None]:
    """应用一处消融，结束后恢复。

    直接改运行期对象而不是重建检索器：重建要重新加载 BGE（每次约 5 秒），
    而 HybridRetriever 是在**每次 search 时**读这些字段的，改了就立即生效。
    """
    from src.retriever import hybrid as hybrid_module

    saved = {
        "min_vector_score": settings.min_vector_score,
        "min_bm25_ratio": settings.min_bm25_ratio,
        "distinctive_df_ratio": settings.distinctive_df_ratio,
        "bm25_min_ratio": getattr(bundle.retriever.bm25, "min_ratio", None),
        "bm25_only_min": hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES,
    }
    try:
        mutate()
        yield
    finally:
        settings.min_vector_score = saved["min_vector_score"]
        settings.min_bm25_ratio = saved["min_bm25_ratio"]
        settings.distinctive_df_ratio = saved["distinctive_df_ratio"]
        if saved["bm25_min_ratio"] is not None:
            bundle.retriever.bm25.min_ratio = saved["bm25_min_ratio"]
        hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES = saved["bm25_only_min"]


def _outcome(report) -> list[tuple[str, bool, bool]]:
    """把一次评测压成「问题 → (是否命中, 是否拒答)」，用于比较结果变化。"""
    return [
        (item.question, bool(item.hit), bool(item.refused))
        for item in report.results
    ]


def _describe_changes(before: list[tuple[str, bool, bool]], after: list[tuple[str, bool, bool]]) -> list[str]:
    changed: list[str] = []
    for (question, hit_before, refused_before), (_, hit_after, refused_after) in zip(before, after):
        if hit_before != hit_after or refused_before != refused_after:
            before_flag = "拒答" if refused_before else ("命中" if hit_before else "未命中")
            after_flag = "拒答" if refused_after else ("命中" if hit_after else "未命中")
            changed.append(f"{question[:28]}：{before_flag} → {after_flag}")
    return changed


def build_configs() -> list[tuple[str, str, Callable[[Settings, RetrieverBundle], None]]]:
    """每个消融配置：``(名字, 说明, 施加方式)``。"""
    from src.retriever import hybrid as hybrid_module

    def no_vector_floor(settings: Settings, bundle: RetrieverBundle) -> None:
        settings.min_vector_score = 0.0

    def no_bm25_ratio(settings: Settings, bundle: RetrieverBundle) -> None:
        settings.min_bm25_ratio = 0.0
        bundle.retriever.bm25.min_ratio = 0.0

    def no_distinctive(settings: Settings, bundle: RetrieverBundle) -> None:
        # 比例设成 1.0 → 所有查询词都算「有区分度」→ 退化为「命中任意内容词」
        settings.distinctive_df_ratio = 1.0

    def no_keyword_evidence(settings: Settings, bundle: RetrieverBundle) -> None:
        # 门槛降到 0 → 关键词路永远能独立送片段进结果
        hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES = 0

    def no_absent_refusal(settings: Settings, bundle: RetrieverBundle) -> None:
        # 「核心概念在语料里不存在就拒答」这条直接短路掉
        bundle.retriever._query_term_profile = lambda content: (set(content), set())  # type: ignore[method-assign]

    return [
        ("no_vector_floor", "关掉「向量绝对下限对每一路生效」", no_vector_floor),
        ("no_bm25_ratio", "关掉 BM25 相对下限", no_bm25_ratio),
        ("no_distinctive", "关掉「生僻词锚定」", no_distinctive),
        ("no_keyword_evidence", "关掉「关键词路要 ≥2 词或全覆盖」", no_keyword_evidence),
        ("no_absent_refusal", "关掉「核心概念缺失即拒答」", no_absent_refusal),
    ]


def run_ablation(
    eval_path: Path | str,
    settings: Settings | None = None,
    *,
    bundle: RetrieverBundle | None = None,
    top_k: int | None = None,
) -> list[AblationResult]:
    """跑 baseline + 各消融配置，返回结果列表（第一项是 baseline）。"""
    settings = settings or default_settings
    bundle = bundle or load_retriever(settings)

    report = evaluate(str(eval_path), settings, top_k=top_k, bundle=bundle)
    baseline_outcome = _outcome(report)
    results = [
        AblationResult(
            name="baseline",
            description="全部闸门开启（当前配置）",
            recall_at_k=report.recall_at_k,
            mrr=report.mrr,
            hit_rate=report.hit_rate,
            refusal_accuracy=report.refusal_accuracy,
            refused_cases=sum(1 for item in report.results if item.refused),
        )
    ]

    for name, description, mutate in build_configs():
        with _temporary(settings, bundle, lambda: mutate(settings, bundle)):
            ablated = evaluate(str(eval_path), settings, top_k=top_k, bundle=bundle)
            outcome = _outcome(ablated)
            results.append(
                AblationResult(
                    name=name,
                    description=description,
                    recall_at_k=ablated.recall_at_k,
                    mrr=ablated.mrr,
                    hit_rate=ablated.hit_rate,
                    refusal_accuracy=ablated.refusal_accuracy,
                    refused_cases=sum(1 for item in ablated.results if item.refused),
                    changed=_describe_changes(baseline_outcome, outcome),
                )
            )
    return results


def render(results: list[AblationResult]) -> str:
    baseline = results[0]
    lines = [
        f"baseline：Recall@5 {baseline.recall_at_k:.3f}｜MRR {baseline.mrr:.3f}｜"
        f"Hit@5 {baseline.hit_rate:.3f}｜拒答 "
        f"{'—' if baseline.refusal_accuracy is None else f'{baseline.refusal_accuracy:.3f}'}"
        f"（拒答 {baseline.refused_cases} 题）",
        "",
        f"{'关掉的闸门':<34}{'Recall':>8}{'MRR':>8}{'拒答':>8}{'变化的题':>9}",
    ]
    for item in results[1:]:
        refusal = "—" if item.refusal_accuracy is None else f"{item.refusal_accuracy:.3f}"
        lines.append(
            f"{item.description:<34}{item.recall_at_k:>8.3f}{item.mrr:>8.3f}"
            f"{refusal:>8}{len(item.changed):>9}"
        )
    lines.append("")
    lines.append("逐条变化（只看总分容易被互相抵消的得分掩盖）：")
    for item in results[1:]:
        lines.append(f"\n▸ 关掉「{item.description}」→ 变化 {len(item.changed)} 题")
        if not item.changed:
            lines.append("    （没有任何题的结果变化，说明这条闸门在**这份评测集上**不承重）")
        for change in item.changed[:8]:
            lines.append(f"    · {change}")
        if len(item.changed) > 8:
            lines.append(f"    …… 另有 {len(item.changed) - 8} 题")
    return "\n".join(lines)
