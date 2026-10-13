"""答案质量校验的测试。

分两层：

1. **校验器本身**：每条规则都用正反例锁住（越界引用、无引用作答、编造价格、
   该拒答没拒答，以及**不该误报**的情况）；
2. **能不能真的抓到坏答案**：注入一个假生成器故意编造，看框架是否报警。
   这一层很重要——否则「指标全绿」可能只是质检器什么都没检查。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.config import Settings
from src.generator.quality import (
    evaluate_answer,
    extract_citations,
    extract_facts,
    is_refusal,
    normalize,
    unsupported_facts,
)
from src.schema import Chunk, Retrieved


def make_hits() -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(
                chunk_id="p1#0",
                post_id="p1",
                text="裙子：藤上春 Time：10.20 Price：jsk=295；sk=235；外套=135 Color：紫、绿",
                author="Lolita洋装上新",
                created_at="Fri Oct 09 17:45:00 +0800 2026",
                url="https://m.weibo.cn/detail/1",
            ),
            score=1.0,
            source="bm25+vector",
        ),
        Retrieved(
            chunk=Chunk(
                chunk_id="p2#0",
                post_id="p2",
                text="茶会礼仪：不要和别人撞柄，拍照前先问一句。",
                url="https://m.weibo.cn/detail/2",
            ),
            score=0.5,
            source="vector",
        ),
    ]


# ------------------------------------------------------------------ 基础工具

def test_extract_citations_handles_multiple_and_fullwidth():
    assert extract_citations("结论 A [1]，结论 B [2][3]，还有全角［4］") == [1, 2, 3, 4]
    assert extract_citations("重复引用 [2] 与 [2]") == [2]
    assert extract_citations("没有引用") == []


def test_normalize_folds_fullwidth_and_spaces():
    assert normalize("ｊｓｋ＝２９５ 元") == "jsk=295元"
    assert normalize("Time：10.20") == "time:10.20"


def test_is_refusal_recognizes_the_real_wording():
    assert is_refusal("提供的资料中没有相关内容。")
    assert is_refusal("没有检索到与该问题相关的资料，无法回答。")
    assert not is_refusal("藤上春的 JSK 是 295 元 [1]。")


def test_extract_facts_skips_single_digits_and_citations_and_urls():
    facts = extract_facts("有 3 种颜色 [1]，价格 295 元，见 https://m.weibo.cn/detail/5352165263544105")
    assert "295" in facts
    assert "3" not in facts, "单个数字往往是正确归纳，不该当成事实去回查"
    assert not any("5352165263544105" in fact for fact in facts), "URL 里的数字要剔除"


def test_unsupported_facts_flags_invented_numbers_only():
    context = "裙子：藤上春 Time：10.20 Price：jsk=295"
    assert unsupported_facts("价格 295 元 [1]”，售价 880 元", context) == ["880"]
    assert unsupported_facts("价格 295 元，时间 10.20", context) == []


def test_unsupported_facts_tolerates_reformatted_dates():
    """资料写 10.20、答案写 10月20日：数字分量都在资料里，不该误报。"""
    assert unsupported_facts("发售时间 10月20日", "Time：10.20 发售") == []


# ------------------------------------------------------------------ 单条答案

def test_good_extractive_answer_passes_all_checks():
    text = "（以下为检索到的原文摘录，未经模型归纳）\n\n[1] 裙子：藤上春 Price：jsk=295"
    quality = evaluate_answer("藤上春多少钱", text, make_hits())
    assert quality.ok, quality.problems
    assert quality.citation_valid
    assert quality.uncited_sentences == 0, "（以下为……）这类说明句不该算成无引用的结论"


def test_out_of_range_citation_is_a_hard_error():
    quality = evaluate_answer("Q", "据资料 [1]，另外 [7] 也提到 295 元", make_hits())
    assert not quality.ok
    assert quality.invalid_citations == [7]
    assert any("引用编号越界" in item for item in quality.problems)


def test_citation_covers_the_whole_paragraph():
    """引用按**段落**生效：抽取式答案是「[1] + 整段原文」，段内后续句子不算无引用。

    实测踩过：按句子判定把覆盖率误报成 0.485（真实是每条摘录都有引用）。
    """
    text = "[1] 藤上春有三种颜色。紫色最热门。绿色很清新。\n\n[2] 另一条裙子的说明。"
    quality = evaluate_answer("Q", text, make_hits())
    assert quality.sentences == 4
    assert quality.uncited_sentences == 0


def test_extractive_answer_collapses_blank_lines_inside_a_quote():
    """引文自带的空行必须在**源头**折叠：否则一条摘录裂成两段，后一段就没有引用了。

    实测把覆盖率从 1.000 误报成 0.930，就是被这类原文害的。
    """
    from src.config import Settings
    from src.generator.llm import Generator

    chunk = Chunk(chunk_id="p1#0", post_id="p1", text="第一句。\n\n第二句（原文空行）。")
    hit = Retrieved(chunk=chunk, score=1.0, source="vector")
    answer = Generator(Settings(data_dir=".")).answer("Q", [hit])

    assert "\n\n第二句" not in answer.text, "引文内部的空行应被折叠"
    quality = evaluate_answer("Q", answer.text, [hit])
    assert quality.uncited_sentences == 0
    assert quality.ok


def test_uncited_paragraph_is_still_counted():
    text = "[1] 有引用的一句。\n\n这段没有引用。而且有两句。"
    quality = evaluate_answer("Q", text, make_hits())
    assert quality.uncited_sentences == 2


def test_refusal_sentences_are_not_counted_as_uncited():
    """拒答句没有结论，不该拉低引用覆盖率。

    实测漏掉这一条会把真实语料的覆盖率算成 0.930——那 16 句正是 8 条拒答 × 2 句。
    """
    text = "没有检索到与该问题相关的资料，无法回答。请换一种问法，或先补充语料后重建索引。"
    quality = evaluate_answer("Q", text, [])
    assert quality.refused
    assert quality.sentences == 0
    assert quality.uncited_sentences == 0


def test_answer_without_any_citation_is_flagged():
    quality = evaluate_answer("Q", "藤上春的 JSK 是 295 元。", make_hits())
    assert not quality.ok
    assert any("没有任何引用编号" in item for item in quality.problems)


def test_fabricated_price_is_flagged():
    quality = evaluate_answer("Q", "藤上春的 JSK 是 880 元 [1]。", make_hits())
    assert not quality.ok
    assert quality.unsupported == ["880"]
    assert any("疑似编造" in item for item in quality.problems)


def test_empty_retrieval_must_refuse():
    bad = evaluate_answer("Q", "藤上春的 JSK 是 295 元 [1]。", [])
    assert not bad.ok
    assert any("检索为空" in item for item in bad.problems)

    good = evaluate_answer("Q", "没有检索到与该问题相关的资料，无法回答。", [])
    assert good.ok, good.problems
    assert good.refused


def test_refusal_with_context_is_not_a_problem_but_is_counted():
    """有资料也叫不准时拒答，是 prompt 要求的行为，不该判成错误。"""
    quality = evaluate_answer("Q", "提供的资料中没有相关内容。", make_hits())
    assert quality.ok
    assert quality.refused and quality.has_context
    assert quality.mentions_insufficiency
    assert not quality.caveated


def test_partial_answer_is_caveated_not_refused():
    """真模型很爱「先声明资料局限，再给出信息」——那是作答，不是拒答。

    实测把两者混为一谈，会把「有资料却拒答」从 2 条误报成 9 条。
    """
    text = "提供的资料中没有统一结论，不同资料偏好不同：op 更方便凉快 [1]，jsk 更好搭 [2]。"
    quality = evaluate_answer("Q", text, make_hits())

    assert quality.mentions_insufficiency is True
    assert quality.refused is False, "有引用就说明它给出了信息"
    assert quality.caveated is True
    assert quality.ok
    # 带引用的段落照常计入引用覆盖（它不是纯拒答句）
    assert quality.sentences > 0
    assert quality.uncited_sentences == 0


def test_pure_refusal_paragraph_is_excluded_from_coverage():
    text = "提供的资料中没有相关内容。\n\n有资料提到 [1] 这条裙子是紫色的。"
    quality = evaluate_answer("Q", text, make_hits())
    # 第一段是纯拒答 → 不计入；第二段有引用 → 计入且已覆盖
    assert quality.sentences == 1
    assert quality.uncited_sentences == 0


# ------------------------------------------------------------------ 整体流程 + 假生成器

class FakeGenerator:
    """故意编造的假生成器：用来证明质检器真的会报警。"""

    def __init__(self, template: str, *, used_llm: bool = True):
        self.template = template
        self.used_llm = used_llm
        self.calls = 0

    def answer(self, question, hits):
        self.calls += 1
        return SimpleNamespace(
            text=self.template.format(n=len(hits)), used_llm=self.used_llm, hits=list(hits)
        )


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, question, top_k=None):
        return list(self.hits)


def make_bundle(hits) -> SimpleNamespace:
    return SimpleNamespace(retriever=StubRetriever(hits), chunks=list(hits))


def test_framework_catches_a_hallucinating_generator(tmp_path):
    """核心保证：如果生成层编造，评测必须报出来（否则「全绿」毫无意义）。"""
    from src.pipeline.answer_eval import evaluate_answers

    eval_file = tmp_path / "qa.jsonl"
    eval_file.write_text('{"question": "藤上春多少钱？"}\n', encoding="utf-8")

    generator = FakeGenerator("藤上春的 JSK 是 880 元，另有时装周限定 [9]。")
    report = evaluate_answers(
        eval_file,
        Settings(data_dir=tmp_path),
        bundle=make_bundle(make_hits()),
        generator=generator,
    )

    assert report.evaluated == 1
    assert report.citation_valid == 0            # [9] 越界
    assert report.answers_with_unsupported == 1  # 880 编造
    assert report.unsupported_total == 1
    assert report.problems, "有问题的答案必须被收集出来"


def test_framework_passes_a_clean_generator(tmp_path):
    from src.pipeline.answer_eval import evaluate_answers

    eval_file = tmp_path / "qa.jsonl"
    eval_file.write_text('{"question": "藤上春多少钱？"}\n', encoding="utf-8")

    generator = FakeGenerator("藤上春的 JSK 是 295 元 [1]，发售时间 10.20 [1]。")
    report = evaluate_answers(
        eval_file,
        Settings(data_dir=tmp_path),
        bundle=make_bundle(make_hits()),
        generator=generator,
    )

    assert report.citation_valid_rate == 1.0
    assert report.unsupported_rate == 0.0
    assert report.citation_coverage == 1.0
    assert report.problems == []


def test_report_counts_refusal_when_retrieval_is_empty(tmp_path):
    from src.pipeline.answer_eval import evaluate_answers

    eval_file = tmp_path / "qa.jsonl"
    eval_file.write_text('{"question": "起球了怎么办？"}\n', encoding="utf-8")

    generator = FakeGenerator("没有检索到与该问题相关的资料，无法回答。", used_llm=False)
    report = evaluate_answers(
        eval_file,
        Settings(data_dir=tmp_path),
        bundle=make_bundle([]),
        generator=generator,
    )

    assert report.no_context_cases == 1
    assert report.no_context_refused == 1
    assert report.refusal_rate_without_context == 1.0


def test_load_questions_skips_comments_and_blank_lines(tmp_path):
    from src.pipeline.answer_eval import load_questions

    path = tmp_path / "qa.jsonl"
    path.write_text(
        "// 这是注释\n\n{\"question\": \"A\"}\n{不是 JSON\n{\"question\": \"B\"}\n",
        encoding="utf-8",
    )
    assert [item["question"] for item in load_questions(path)] == ["A", "B"]
