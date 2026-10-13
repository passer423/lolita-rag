"""评测逻辑测试。

关键不是为了验证「指标算得对」，而是验证**评测集与语料不匹配时会报错而不是
给出误导性的 0.000** —— 用 mock 评测集去打真实语料，正是最容易犯的错。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.pipeline.eval import EvalSetMismatchError, evaluate, load_eval_set
from src.schema import Chunk, Retrieved


class FakeRetriever:
    """按问题返回预设的召回顺序。"""

    def __init__(self, plan: dict[str, list[str]]):
        self.plan = plan

    def search(self, question: str, top_k: int | None = None, recall_k: int | None = None):
        hits: list[Retrieved] = []
        for post_id in self.plan.get(question, []):
            chunk = Chunk(chunk_id=f"{post_id}#0", post_id=post_id, text="内容", url=f"https://e/{post_id}")
            hits.append(Retrieved(chunk=chunk, score=1.0, source="hybrid"))
        return hits


def make_bundle(plan: dict[str, list[str]], corpus_ids: list[str]):
    chunks = [Chunk(chunk_id=f"{pid}#0", post_id=pid, text="内容") for pid in corpus_ids]
    return SimpleNamespace(retriever=FakeRetriever(plan), chunks=chunks)


def write_eval_set(tmp_path, items: list[dict]) -> str:
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items), encoding="utf-8")
    return str(path)


def test_metrics_are_computed_correctly(tmp_path):
    items = [
        {"question": "Q1", "relevant": ["p1"]},
        {"question": "Q2", "relevant": ["p2", "p3"]},
    ]
    # Q1 命中且排第 1；Q2 只召回 p2 且排第 2（另一条相关博文没召回）
    plan = {"Q1": ["p1", "p9"], "Q2": ["p9", "p2"]}
    report = evaluate(
        write_eval_set(tmp_path, items),
        Settings(top_k=5),
        bundle=make_bundle(plan, ["p1", "p2", "p3", "p9"]),
    )

    assert report.scored == 2
    assert report.total == 2
    assert report.recall_at_k == pytest.approx(0.75)      # (1.0 + 0.5) / 2
    assert report.mrr == pytest.approx(0.75)              # (1.0 + 0.5) / 2
    assert report.hit_rate == pytest.approx(1.0)


def test_mismatched_eval_set_raises_instead_of_reporting_zero(tmp_path):
    """拿 mock 评测集打真实语料时必须报错——0.000 会被误读成「检索坏了」。"""
    items = [{"question": "Q", "relevant": ["mock-001", "mock-002"]}]

    with pytest.raises(EvalSetMismatchError, match="不是同一批数据"):
        evaluate(
            write_eval_set(tmp_path, items),
            Settings(top_k=5),
            bundle=make_bundle({}, ["real-1", "real-2"]),
        )


def test_partially_missing_labels_only_warns(tmp_path, caplog):
    items = [{"question": "Q", "relevant": ["p1", "已被删除的博文"]}]

    with caplog.at_level("WARNING"):
        report = evaluate(
            write_eval_set(tmp_path, items),
            Settings(top_k=5),
            bundle=make_bundle({"Q": ["p1"]}, ["p1", "p2"]),
        )

    assert report.scored == 1
    assert report.recall_at_k == pytest.approx(0.5)
    assert any("不在当前语料中" in record.getMessage() for record in caplog.records)


def test_no_answer_case_is_counted_when_it_refuses(tmp_path):
    items = [{"question": "语料里没有的问题", "expect_no_answer": True}]
    report = evaluate(
        write_eval_set(tmp_path, items),
        Settings(top_k=5),
        bundle=make_bundle({}, ["p1"]),
    )

    assert report.refusal_total == 1
    assert report.refusal_correct == 1
    assert report.refusal_accuracy == pytest.approx(1.0)
    assert report.scored == 0, "拒答用例不参与召回指标"


def test_no_answer_case_fails_when_hits_are_returned(tmp_path):
    """这正是当前的真实短板：稠密检索总会返回最近的若干条，于是拒答失败。"""
    items = [{"question": "语料里没有的问题", "expect_no_answer": True}]
    report = evaluate(
        write_eval_set(tmp_path, items),
        Settings(top_k=5),
        bundle=make_bundle({"语料里没有的问题": ["p1"]}, ["p1"]),
    )

    assert report.refusal_total == 1
    assert report.refusal_correct == 0
    assert report.refusal_accuracy == pytest.approx(0.0)


def test_load_eval_set_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "// 这一行是注释\n\n" + json.dumps({"question": "Q", "relevant": ["p1"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert len(load_eval_set(path)) == 1


def test_load_eval_set_reports_bad_line_number(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text('{"question": "Q"}\n{坏 JSON}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        load_eval_set(path)


def test_missing_eval_set_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "不存在.jsonl")
