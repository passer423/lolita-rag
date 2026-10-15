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


def test_split_metrics_are_reported_separately(tmp_path):
    """dev 调参、test 验收——只报一个总分的话，test 半区退化了会被 dev 稀释掉。"""
    items = [
        {"question": "D1", "relevant": ["p1"]},
        {"question": "D2", "relevant": ["p2"]},
        {"question": "T1", "relevant": ["p3"], "split": "test"},
        {"question": "T2", "relevant": ["p4"], "split": "test"},
        {"question": "T3", "expect_no_answer": True, "split": "test"},
    ]
    plan = {
        "D1": ["p1"],            # dev 满分
        "D2": ["p9", "p2"],      # dev 第 2 名
        "T1": ["p9", "p9b", "p3"],  # test 第 3 名
        "T2": ["p4"],            # test 满分
        "T3": [],                # test 正确拒答
    }
    report = evaluate(
        write_eval_set(tmp_path, items),
        Settings(top_k=5),
        bundle=make_bundle(plan, ["p1", "p2", "p3", "p4", "p9", "p9b"]),
    )

    assert set(report.by_split) == {"dev", "test"}
    dev = report.by_split["dev"]
    test = report.by_split["test"]

    assert dev["total"] == 2 and dev["mrr"] == pytest.approx(0.75)      # (1 + 0.5)/2
    assert test["total"] == 3
    assert test["mrr"] == pytest.approx((1 / 3 + 1.0) / 2)              # 拒答题不计入 MRR
    assert test["hit_rate"] == pytest.approx(1.0)
    assert test["refusal_accuracy"] == pytest.approx(1.0)

    # 总分是两半合起来算的，不能等于任何一半
    assert report.mrr == pytest.approx((1 + 0.5 + 1 / 3 + 1.0) / 4)
    assert "按划分" in report.render()


def test_single_split_eval_set_has_no_split_table(tmp_path):
    """只有一种划分时不显示分表，免得给简单评测集添噪音。"""
    items = [{"question": "Q1", "relevant": ["p1"]}]
    report = evaluate(
        write_eval_set(tmp_path, items),
        Settings(top_k=5),
        bundle=make_bundle({"Q1": ["p1"]}, ["p1"]),
    )

    assert report.by_split == {}
    assert "按划分" not in report.render()


def test_evaluate_can_run_one_split_only(tmp_path):
    """`--split test` 要能只跑验收半区。"""
    items = [
        {"question": "D1", "relevant": ["p1"]},
        {"question": "T1", "relevant": ["p2"], "split": "test"},
    ]
    bundle = make_bundle({"D1": ["p1"], "T1": ["p2"]}, ["p1", "p2"])

    only_test = evaluate(write_eval_set(tmp_path, items), Settings(top_k=5), bundle=bundle, split="test")

    assert only_test.total == 1
    assert only_test.results[0].question == "T1"

    with pytest.raises(ValueError, match="split="):
        evaluate(write_eval_set(tmp_path, items), Settings(top_k=5), bundle=bundle, split="nope")


def test_question_result_marks_expectation_so_failures_are_distinguishable(tmp_path):
    """拒答用例漏答必须能认出是"该拒未拒"，不能和正例未命中混在一起（实测踩过）。"""
    items = [{"question": "无关问题", "expect_no_answer": True}]
    report = evaluate(
        write_eval_set(tmp_path, items),
        Settings(top_k=5),
        bundle=make_bundle({"无关问题": ["p1"]}, ["p1"]),
    )

    result = report.results[0]
    assert result.expect_no_answer is True and result.refused is False
    assert "[该拒未拒 1 条]" in report.render()


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
