"""运行时答案闸门的测试。

这道闸门的全部价值在于「**不达标的答案根本到不了用户手里**」，所以测试的核心不是
「好答案能通过」，而是「坏答案会被换掉」——而且换掉之后**不留痕**：

* 越界引用、凭空数字 → 必须降级；
* 降级产物必须是原文摘录（一字不改）→ 所以降级后不可能还带着原来的问题；
* 干净答案 → 一个字都不能动；
* 抽取式答案（本来就照抄原文）→ 不该被降级成同一个东西；
* 关掉开关 → 原样返回，但 `checked=False`，**别把「没检查」当「检查通过」**。
"""

from __future__ import annotations

from src.config import Settings
from src.generator.llm import Answer, Generator
from src.generator.quality import evaluate_answer
from src.generator.verify import verify_and_degrade
from src.schema import Chunk, Retrieved


def make_hits() -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(
                chunk_id="p1#0",
                post_id="p1",
                text="裙子：藤上春 Price：jsk=295；sk=235 Color：紫、绿",
                author="Lolita洋装上新",
                url="https://m.weibo.cn/detail/1",
            ),
            score=1.0,
            source="bm25+vector",
        ),
        Retrieved(
            chunk=Chunk(chunk_id="p2#0", post_id="p2", text="茶会礼仪：拍照前先问一句。"),
            score=0.5,
            source="vector",
        ),
    ]


def make_settings(**overrides) -> Settings:
    overrides.setdefault("llm_api_key", "sk-test")
    return Settings(**overrides)


def model_answer(text: str) -> Answer:
    return Answer(text=text, hits=make_hits(), used_llm=True)


# ------------------------------------------------------------------ 拦截

def test_out_of_range_citation_is_degraded():
    answer, outcome = verify_and_degrade(
        "藤上春多少钱", model_answer("价格是 295 元 [9]。"), make_hits(), make_settings()
    )

    assert outcome.checked and outcome.degraded
    assert any("引用编号越界" in item for item in outcome.problems)
    assert "[9]" not in answer.text, "降级后不能还留着越界引用"
    assert answer.text.startswith("（以下为检索到的原文摘录"), "降级产物必须是原文摘录"
    assert "未通过引用校验" in answer.note
    assert answer.used_llm is False


def test_fabricated_number_is_degraded():
    answer, outcome = verify_and_degrade(
        "藤上春多少钱", model_answer("价格是 880 元 [1]。"), make_hits(), make_settings()
    )

    assert outcome.degraded
    assert any("疑似编造" in item for item in outcome.problems)
    assert "880" not in answer.text, "编造的数字不能出现在返回给用户的答案里"


def test_answer_without_any_citation_is_degraded():
    _, outcome = verify_and_degrade(
        "藤上春多少钱", model_answer("藤上春的 JSK 是 295 元。"), make_hits(), make_settings()
    )
    assert outcome.degraded
    assert any("没有任何引用编号" in item for item in outcome.problems)


def test_degrades_to_verbatim_source_text():
    """降级产物是语料原文，所以「编造」在物理上不可能继续存在。"""
    answer, _ = verify_and_degrade(
        "q", model_answer("凭空捏造的内容 [9]，价格 8888 元"), make_hits(), make_settings()
    )
    for hit in make_hits():
        assert hit.chunk.text in answer.text


# ------------------------------------------------------------------ 不误伤

def test_clean_model_answer_is_untouched():
    original = model_answer("藤上春的 JSK 是 295 元 [1]，发售时间见资料 [1]。")
    answer, outcome = verify_and_degrade("藤上春多少钱", original, make_hits(), make_settings())

    assert outcome.checked and not outcome.degraded and outcome.problems == []
    assert answer is original, "通过的答案必须原样返回，一个字都不改"


def test_extractive_answer_is_not_verified_or_degraded():
    """抽取式答案本身就是原文照抄，没必要校验，更不该被降级成同一个东西。"""
    extractive = Generator.extractive_answer(make_hits(), note="未配置 LLM_API_KEY")
    answer, outcome = verify_and_degrade("q", extractive, make_hits(), make_settings())

    assert answer is extractive
    assert outcome.checked is True and outcome.degraded is False


def test_no_context_refusal_passes():
    """检索为空时根本没调模型，也不该被降级。"""
    extractive = Generator.extractive_answer([], note="no_context")
    answer, outcome = verify_and_degrade("q", extractive, [], make_settings())
    assert answer is extractive and not outcome.degraded


def test_disabled_switch_returns_answer_but_flags_unchecked():
    """关掉开关时原样返回，但 `checked=False` —— 「没检查」不等于「检查通过」。"""
    bad = model_answer("价格是 880 元 [9]。")
    answer, outcome = verify_and_degrade(
        "q", bad, make_hits(), make_settings(answer_verify_enabled=False)
    )

    assert answer is bad
    assert outcome.checked is False and outcome.degraded is False


# ------------------------------------------------------------------ 与评测的分工

def test_generator_itself_does_not_verify():
    """校验**不能**塞进 Generator：那样评测就永远看不到模型的原始输出，全绿也没意义。"""
    from src.generator.llm import Generator as Gen

    generator = Gen(make_settings(llm_api_key=""))
    answer = generator.answer("q", make_hits())
    # 抽取式路径不经过校验（这里用它证明 Generator.answer 本身不带校验逻辑）
    assert answer.note == "未配置 LLM_API_KEY，已退化为抽取式答案"
    assert not hasattr(answer, "problems")


def test_evaluate_answer_sees_the_raw_model_output():
    """评测看原始输出：越界引用在 quality 里必须报得出来（否则闸门无处可拦）。"""
    quality = evaluate_answer("q", "价格是 880 元 [9]。", make_hits(), used_llm=True)
    assert not quality.ok
    assert quality.invalid_citations == [9]
    assert quality.unsupported == ["880"]
