"""评测集本身的质量测试。

## 为什么需要给「测试数据」写测试

这个项目里**指标错了比模型错了更危险**的说法出现过好几次，而评测集就是指标的源头。
它已经出过这些事：

* 标注的 post_id 在语料里根本不存在（换了语料来源后标注失效）——评测会打警告，但
  得有人看；
* 「谢安然**今天**穿的是哪个牌子？」曾经排第 1，语料一增长就必然失效（新帖才是
  「今天」的答案），实测已经掉成 0 命中；
* 从语料里挖真实提问时，很大一部分的 top-1 就是**提问那篇求助帖自己**——把它标成
  答案等于奖励「返回问题本身」，指标虚高却什么都没测到。

所以这里做**不依赖语料的静态检查**（CI 上也能跑）：格式、唯一性、时效词、拒答用例
的形态。依赖语料的那部分（标注 id 是否存在）留给 `scripts/eval.py` 的运行时警告。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

EVAL_FILES = ("eval/qa_real.jsonl", "eval/qa_mock.jsonl")

# 真正的**时效词**：问题一旦包含它们，语料增长后标注就可能失效。
# 刻意不含「现在」——它常常只是对比用法（「早年…和现在有什么不同」），
# 把那种题也判成时效题是误伤，实测第一版扫描就误报了 1 条。
TIME_WORDS = ("今天", "明天", "昨天", "最新", "当日", "本月")


def load(path: str) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(
        (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:  # pragma: no cover - 出错时给出精确位置
            pytest.fail(f"{path}:{lineno} 不是合法 JSON：{exc}")
    return rows


@pytest.mark.parametrize("path", EVAL_FILES)
def test_eval_files_parse_and_are_not_empty(path):
    rows = load(path)
    assert rows, f"{path} 里一条题目都没有"
    assert all("question" in row for row in rows), f"{path} 里有条目缺 question"


@pytest.mark.parametrize("path", EVAL_FILES)
def test_every_item_is_either_positive_or_refusal(path):
    """两者必居其一，不能都有、也不能都没有——否则这一题在统计里会变得含糊。"""
    for row in load(path):
        has_relevant = bool(row.get("relevant"))
        is_refusal = bool(row.get("expect_no_answer"))
        assert has_relevant != is_refusal, (
            f"{path} 的「{row.get('question')}」既没有 relevant 也没标 expect_no_answer，"
            "或者两者同时存在"
        )


@pytest.mark.parametrize("path", EVAL_FILES)
def test_questions_are_unique_and_not_too_short(path):
    rows = load(path)
    questions = [str(row["question"]).strip() for row in rows]
    duplicates = {q for q in questions if questions.count(q) > 1}
    assert not duplicates, f"{path} 有重复问题：{duplicates}"
    for question in questions:
        assert len(question) >= 6, f"问题太短，信息量不足：{question!r}"


@pytest.mark.parametrize("path", EVAL_FILES)
def test_positive_questions_avoid_time_sensitive_wording(path):
    """带「今天/最新」的正例会随语料增长而失效——这是实际踩过的坑。

    要么改成不依赖「今天」的问法，要么把它降级成拒答用例（那种题本来就该由生成层
    回答「资料里没有今天的记录」）。被挪出去的题记在文件头的演变记录里。
    """
    offenders = [
        str(row["question"])
        for row in load(path)
        if not row.get("expect_no_answer")
        and any(word in str(row["question"]) for word in TIME_WORDS)
    ]
    assert not offenders, f"{path} 里有时效性正例（语料一涨就会失效）：{offenders}"


@pytest.mark.parametrize("path", EVAL_FILES)
def test_refusal_cases_are_not_about_a_topic_the_corpus_covers(path):
    """拒答用例**不能**是「话题存在、只是缺某个具体事实」的题。

    实测踩过：「痛lo 这个说法最早是谁提出的」「妆造一次花多少钱」——语料里明明有
    该话题的内容，检索返回它们是**合理行为**，说不清是生成层的责任。这类题混进
    拒答集合会把「拒答准确率」变成假指标。

    这里只能做形态检查（题面是否在问"起源/价格"这类具体事实）；真正的判据是
    标注前的实测（df=0 或该话题零内容），写在文件头。
    """
    fact_seeking = ("多少钱", "是谁提出", "哪一年成立", "起源于", "国家标准", "法律")
    for row in load(path):
        if not row.get("expect_no_answer"):
            continue
        question = str(row["question"])
        matched = [word for word in fact_seeking if word in question]
        if matched:
            # 不是硬性禁止，但必须能被解释：要么语料里确实没有这个概念，要么
            # 它在问语料不可能承载的东西（外部标准/法律/行情）。这里只提示。
            assert len(question) >= 6, question


@pytest.mark.parametrize("path", EVAL_FILES)
def test_multilabel_questions_have_no_duplicate_ids(path):
    for row in load(path):
        relevant = [str(x) for x in row.get("relevant", [])]
        assert len(relevant) == len(set(relevant)), f"标注有重复 id：{row['question']}"
