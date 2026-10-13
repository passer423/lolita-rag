"""闸门消融工具的测试。

最要紧的一条是**恢复现场**：消融是直接改运行期对象（settings 字段、模块常量）实现的，
如果跑完不恢复，后续所有评测都会被悄悄污染——这种 bug 极难发现，因为工具本身"看起来成功了"。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.config import Settings
from src.pipeline.ablation import (
    _describe_changes,
    _temporary,
    build_configs,
    render,
    run_ablation,
)
from src.retriever import hybrid as hybrid_module
from src.schema import Chunk, Retrieved


def make_hits(text: str = "带亮片的柄图不能机洗，建议冷水手洗。") -> list[Retrieved]:
    chunk = Chunk(chunk_id="p1#0", post_id="p1", text=text, url="https://e/p1")
    return [Retrieved(chunk=chunk, score=1.0, source="vector")]


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.bm25 = SimpleNamespace(min_ratio=0.2)

    def search(self, question, top_k=None, recall_k=None):
        # 简单规则：问题里含「没有」就拒答，方便构造结果变化
        return [] if "没有" in question else list(self.hits)


def make_bundle(hits):
    # 注意 chunks 要放 **Chunk**（评测会用它校验标注 id 是否存在于语料），不是 Retrieved
    return SimpleNamespace(
        retriever=StubRetriever(hits), chunks=[hit.chunk for hit in hits], embedder=None
    )


def write_eval(tmp_path, items) -> str:
    """写一个最小评测集。

    ⚠️ 必须带 ``relevant`` 或 ``expect_no_answer``：评测会跳过**没有标注**的题
    （实测踩过——不写标注时 recall 恒为 0，看起来像「消融把检索搞坏了」）。
    """
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )
    return str(path)


# ------------------------------------------------------------------ 恢复现场

def test_temporary_restores_every_knob_it_touches(tmp_path):
    settings = Settings(
        data_dir=tmp_path, min_vector_score=0.44, min_bm25_ratio=0.2, distinctive_df_ratio=0.10
    )
    bundle = make_bundle(make_hits())
    original = (
        settings.min_vector_score,
        settings.min_bm25_ratio,
        settings.distinctive_df_ratio,
        bundle.retriever.bm25.min_ratio,
        hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES,
    )

    def mutate() -> None:
        settings.min_vector_score = 0.0
        settings.min_bm25_ratio = 0.0
        settings.distinctive_df_ratio = 1.0
        bundle.retriever.bm25.min_ratio = 0.0
        hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES = 0

    with _temporary(settings, bundle, mutate):
        assert settings.min_vector_score == 0.0, "块内应当已应用消融"
        assert hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES == 0

    assert (
        settings.min_vector_score,
        settings.min_bm25_ratio,
        settings.distinctive_df_ratio,
        bundle.retriever.bm25.min_ratio,
        hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES,
    ) == original, "退出后必须完全恢复，否则会污染后续评测"


def test_temporary_restores_even_when_the_body_raises(tmp_path):
    settings = Settings(data_dir=tmp_path, min_vector_score=0.44)
    bundle = make_bundle(make_hits())

    try:
        with _temporary(settings, bundle, lambda: setattr(settings, "min_vector_score", 0.0)):
            raise RuntimeError("消融途中炸了")
    except RuntimeError:
        pass
    assert settings.min_vector_score == 0.44


# ------------------------------------------------------------------ 结果比对

def test_describe_changes_reports_flips_in_both_directions():
    before = [("A", True, False), ("B", False, True), ("C", True, False)]
    after = [("A", False, False), ("B", False, False), ("C", True, False)]

    changed = _describe_changes(before, after)
    assert len(changed) == 2
    assert "A：命中 → 未命中" in changed[0]
    assert "B：拒答 → 未命中" in changed[1]


def test_configs_cover_every_gate_we_added():
    names = {name for name, _, _ in build_configs()}
    assert names == {
        "no_vector_floor",
        "no_bm25_ratio",
        "no_distinctive",
        "no_keyword_evidence",
        "no_absent_refusal",
    }


# ------------------------------------------------------------------ 端到端（假检索器）

def test_run_ablation_reports_baseline_first_and_restores_state(tmp_path):
    settings = Settings(data_dir=tmp_path, min_vector_score=0.44)
    bundle = make_bundle(make_hits())
    eval_path = write_eval(
        tmp_path,
        [
            {"question": "亮片裙子怎么洗", "relevant": ["p1"]},
            {"question": "没有答案的问题", "expect_no_answer": True},
        ],
    )

    results = run_ablation(eval_path, settings, bundle=bundle)

    assert results[0].name == "baseline"
    assert len(results) == 1 + len(build_configs())
    assert settings.min_vector_score == 0.44, "整体跑完也要恢复"
    assert hybrid_module.BM25_ONLY_MIN_CONTENT_MATCHES == 2

    baseline = results[0]
    assert baseline.recall_at_k == 1.0
    assert baseline.refusal_accuracy == 1.0
    assert baseline.refused_cases == 1

    text = render(results)
    assert "baseline" in text
    assert "关掉的闸门" in text


def test_ablation_marks_a_gate_as_not_load_bearing_when_nothing_changes(tmp_path):
    """假检索器不看闸门，所以每一条都应当被标成「在这份评测集上不承重」。"""
    settings = Settings(data_dir=tmp_path)
    bundle = make_bundle(make_hits())
    eval_path = write_eval(tmp_path, [{"question": "亮片裙子怎么洗", "relevant": ["p1"]}])

    results = run_ablation(eval_path, settings, bundle=bundle)
    for item in results[1:]:
        assert item.changed == []
    assert "不承重" in render(results)
