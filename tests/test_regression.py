"""回归对比逻辑的测试。

重点是那个**真实踩过的统计 bug**：pytest 的摘要行在本环境不出现时，退化为「数所有
``. F E s x`` 字符」会把警告文本里的 ``sentence_transformers`` 也算进去，
实测把 213 个用例报成了 254。这种错会悄悄写进基线，所以必须锁住。
"""

from __future__ import annotations

from src.pipeline.regression import compare, parse_counts, render


def dots(count: int) -> str:
    """程序化构造进度行——手数点很容易数错，测试本身就不该依赖手数。"""
    return "." * count


# 真实捕获到的样子：3 行进度（72 + 72 + 69 = 213）+ 含英文单词/路径的警告文本。
# 陷阱就在警告里：``sentence_transformers``、``starlette.testclient`` 这些词里的
# s/x 字符会被「数所有 . F E s x」这种错误实现算成用例。
REAL_OUTPUT = (
    f"{dots(72)} [ 33%]\n"
    f"{dots(72)} [ 66%]\n"
    f"{dots(69)}        [100%]\n"
    "============================= warnings summary ===============================\n"
    "..\\..\\..\\..\\..\\..\\canda\\Lib\\site-packages\\jieba\\_compat.py:18\n"
    "  C:\\Python\\Lib\\site-packages\\jieba\\_compat.py:18: UserWarning: pkg_resources is deprecated\n"
    "    import pkg_resources\n"
    "\n"
    "..\\..\\..\\..\\..\\..\\canda\\Lib\\site-packages\\fastapi\\testclient.py:1\n"
    "  C:\\Python\\Lib\\site-packages\\fastapi\\testclient.py:1: StarletteDeprecationWarning\n"
    "    from starlette.testclient import TestClient as TestClient  # noqa\n"
    "\n"
    "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
)


def test_parse_counts_ignores_letters_in_warning_text():
    """213 个点就是 213 个用例——不能因为警告里有 sentence_transformers 就变大。"""
    passed, failed = parse_counts(REAL_OUTPUT)
    assert (passed, failed) == (213, 0)


def test_parse_counts_prefers_the_summary_line_when_present():
    passed, failed = parse_counts(REAL_OUTPUT + "\n254 passed, 2 warnings in 31.20s\n")
    assert (passed, failed) == (254, 0), "「2 warnings」不是失败，别把它当 failed"

    passed, failed = parse_counts(REAL_OUTPUT + "\n252 passed, 1 failed, 2 warnings in 31.20s\n")
    assert (passed, failed) == (252, 1)


def test_parse_counts_detects_failures():
    output = f"{dots(30)}F{dots(8)}\n{dots(10)}E{dots(5)}\n"
    passed, failed = parse_counts(output)
    assert passed == 53
    assert failed == 2


def test_parse_counts_handles_empty_output():
    assert parse_counts("") == (0, 0)


def _snapshot(mrr: float = 0.885, recall: float = 1.0, failed: int = 0, chunks: int = 1372) -> dict:
    return {
        "eval_path": "eval/qa_real.jsonl",
        "fingerprint": {
            "chunks": chunks,
            "embedder": "bge:x",
            "fusion_mode": "weighted",
            "min_vector_score": 0.44,
        },
        "metrics": {"recall_at_k": recall, "mrr": mrr, "refusal_accuracy": 1.0},
        "answer_metrics": {
            "citation_valid_rate": 1.0,
            "citation_coverage": 1.0,
            "unsupported_rate": 0.0,
            "refusal_rate_without_context": 1.0,
            "problem_count": 0.0,
        },
        "tests": {"passed": 246, "failed": failed},
    }


def test_answer_metric_drop_is_also_a_regression():
    """生成层指标此前完全没人守：改提示词把引用格式改坏，检索指标是看不出来的。"""
    current = _snapshot()
    current["answer_metrics"]["citation_valid_rate"] = 0.6
    regressions, _ = compare(current, _snapshot())
    assert any("生成层 citation_valid_rate" in item for item in regressions)


def test_metrics_that_should_go_down_are_compared_in_the_right_direction():
    """``unsupported_rate`` / ``problem_count`` 越小越好，方向不能反。

    反了会很危险：模型开始编造（0 → 0.3）不会被判回退，而它变好（0.3 → 0）反被
    报成回退。这个方向错误我一开始用「现有指标都是越高越好」把自己说服了，其实是 bug。
    """
    worse = _snapshot()
    worse["answer_metrics"]["unsupported_rate"] = 0.3
    worse["answer_metrics"]["problem_count"] = 5.0
    regressions, _ = compare(worse, _snapshot())
    assert any("unsupported_rate 从 0.0 升到" in item for item in regressions)
    assert any("problem_count 从 0.0 升到" in item for item in regressions)

    better = _snapshot()
    better["answer_metrics"]["unsupported_rate"] = 0.0
    assert compare(better, _snapshot())[0] == []


def test_metric_drop_is_reported_as_a_regression():
    regressions, notes = compare(_snapshot(mrr=0.80), _snapshot(mrr=0.885))
    assert any("mrr" in item for item in regressions)
    assert notes == []


def test_tiny_float_wobble_is_not_a_regression():
    regressions, _ = compare(_snapshot(mrr=0.883), _snapshot(mrr=0.885))
    assert regressions == []


def test_test_failures_are_a_regression_even_if_metrics_hold():
    regressions, _ = compare(_snapshot(failed=1), _snapshot())
    assert any("测试失败" in item for item in regressions)


def test_fingerprint_change_is_only_a_note_not_a_failure():
    """换了语料/阈值后指标不可直接比——但那不是回退，不能把回归判成失败。"""
    regressions, notes = compare(_snapshot(chunks=900), _snapshot(chunks=1372))
    assert regressions == []
    assert any("指纹" in note for note in notes)


def test_none_metrics_are_skipped_instead_of_crashing():
    current = _snapshot()
    current["metrics"]["refusal_accuracy"] = None
    regressions, _ = compare(current, _snapshot())
    assert regressions == []


def test_render_shows_delta_against_baseline():
    text = render(_snapshot(mrr=0.80), _snapshot(mrr=0.885))
    assert "-0.0850" in text
    assert "1372 片段" in text
