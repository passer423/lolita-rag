"""LLM-as-judge 的测试。

评审器最危险的失败模式不是「判错一次」，而是**把评审自己的故障算成被评审者的错**：
模型没吐 JSON、超时、返回空，如果都当成「不忠实」，系统就会开始自己造假阳性，
然后你会去改一个根本没坏的提示词。所以下面一半用例在守这条线。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.config import Settings
from src.generator.judge import AnswerJudge, parse_verdict, summarize_issues
from src.schema import Chunk, Retrieved


def make_hits() -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(
                chunk_id="p1#0",
                post_id="p1",
                text="宝石芭蕾：具体预约/尾款详情于10.3-10.9左右公布",
                author="petitbijou",
                url="https://m.weibo.cn/detail/1",
            ),
            score=1.0,
            source="bm25+vector",
        )
    ]


class FakeClient:
    """假客户端：按预置文本返回，用于在不联网的情况下测解析与统计。"""

    def __init__(self, text: str | None = None, exc: Exception | None = None):
        self.text = text
        self.exc = exc
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))]
        )


def make_settings(**overrides) -> Settings:
    overrides.setdefault("llm_api_key", "sk-test")
    return Settings(**overrides)


# ------------------------------------------------------------------ 解析

def test_parses_plain_json():
    verdict = parse_verdict('{"faithful": true, "issues": []}')
    assert verdict.faithful is True
    assert verdict.issues == []
    assert not verdict.failed


def test_parses_json_wrapped_in_code_fence():
    """实测模型常把 JSON 包在 ```json 里，或前后加一句话——不能因此判成评审失败。"""
    text = '好的，我的判断是：\n```json\n{"faithful": false, "issues": [{"kind": "false_insufficiency", "detail": "资料里有日期，答案却说没有"}]}\n```\n'
    verdict = parse_verdict(text)
    assert verdict.faithful is False
    assert verdict.issues[0].kind == "false_insufficiency"


def test_parses_issue_given_as_plain_string():
    verdict = parse_verdict('{"faithful": false, "issues": ["把张三的裙子算到李四头上"]}')
    assert verdict.faithful is False
    assert verdict.issues[0].kind == "unknown"


def test_unparseable_output_is_unknown_not_unfaithful():
    """这条最关键：评审员自己故障，绝不能算成被评审的答案不忠实。"""
    for text in ("", "我觉得挺好的", "[1, 2, 3]", "```\nnot json\n```"):
        verdict = parse_verdict(text)
        assert verdict.faithful is None, f"不该把 {text!r} 当成判定结果"
        assert verdict.failed


def test_missing_faithful_field_is_unknown():
    verdict = parse_verdict('{"issues": [{"kind": "x", "detail": "y"}]}')
    assert verdict.failed
    assert "faithful" in verdict.error


def test_unfaithful_without_reason_is_downgraded_to_unknown():
    """判「不忠实」却不给理由 → 无从排查的假阳性，降级为未知。"""
    verdict = parse_verdict('{"faithful": false, "issues": []}')
    assert verdict.failed
    assert "理由" in verdict.error


# ------------------------------------------------------------------ 调用与兜底

def test_judge_calls_the_model_with_context_and_answer():
    client = FakeClient('{"faithful": true, "issues": []}')
    judge = AnswerJudge(make_settings(), client=client)

    verdict = judge.judge("预约什么时候公布？", "资料中没有公布时间 [1]。", make_hits())

    assert verdict.faithful is True
    call = client.calls[0]
    assert call["temperature"] == 0.0, "评审要稳，不能发挥"
    prompt = call["messages"][1]["content"]
    assert "10.3-10.9" in prompt, "资料要交给评审员，否则它没法判断「资料里到底有没有」"
    assert "资料中没有公布时间" in prompt


def test_judge_returns_unknown_when_the_call_fails():
    client = FakeClient(exc=RuntimeError("连接超时"))
    verdict = AnswerJudge(make_settings(), client=client).judge("q", "a [1]", make_hits())
    assert verdict.failed
    assert "连接超时" in verdict.error


def test_judge_skips_without_context_or_key():
    no_context = AnswerJudge(make_settings(), client=FakeClient("{}")).judge("q", "a", [])
    assert no_context.failed and "没有资料" in no_context.error

    no_key = AnswerJudge(make_settings(llm_api_key=""), client=FakeClient("{}")).judge(
        "q", "a [1]", make_hits()
    )
    assert no_key.failed and "LLM_API_KEY" in no_key.error


def test_judge_skips_empty_answer():
    verdict = AnswerJudge(make_settings(), client=FakeClient("{}")).judge("q", "   ", make_hits())
    assert verdict.failed


def test_summarize_issues_counts_by_kind():
    verdicts = [parse_verdict(
        '{"faithful": false, "issues": [{"kind": "false_insufficiency", "detail": "a"},'
        ' {"kind": "false_insufficiency", "detail": "b"}, {"kind": "misattribution", "detail": "c"}]}'
    )]
    text = summarize_issues(verdicts)
    assert "false_insufficiency ×2" in text
    assert "misattribution ×1" in text
    assert summarize_issues([]) == "（没有发现问题）"


# ------------------------------------------------------------------ 接进评测

def test_answer_eval_reports_faithfulness_and_excludes_failed_judgements(tmp_path):
    """忠实率的分母只算**已有判定**的：评审失败既不算忠实也不算不忠实。"""
    from src.pipeline.answer_eval import evaluate_answers

    eval_file = tmp_path / "qa.jsonl"
    eval_file.write_text('{"question": "预约什么时候公布？"}\n', encoding="utf-8")

    class StubRetriever:
        bm25 = SimpleNamespace(min_ratio=0.2)

        def search(self, question, top_k=None, recall_k=None):
            return make_hits()

    class StubGenerator:
        used_llm = True

        def answer(self, question, hits):
            from src.generator.llm import Answer

            return Answer(text="资料中没有公布时间 [1]。", hits=list(hits), used_llm=True)

    class StubJudge:
        def judge(self, question, answer_text, hits):
            return parse_verdict(
                '{"faithful": false, "issues": [{"kind": "false_insufficiency", "detail": "资料里有日期"}]}'
            )

    report = evaluate_answers(
        eval_file,
        Settings(data_dir=tmp_path),
        bundle=SimpleNamespace(retriever=StubRetriever(), chunks=[h.chunk for h in make_hits()]),
        generator=StubGenerator(),
        judge=StubJudge(),
    )

    assert report.judged == 1 and report.unfaithful == 1
    assert report.faithfulness_rate == 0.0
    assert report.issues_by_kind == {"false_insufficiency": 1}
    assert report.unfaithful_cases[0]["question"] == "预约什么时候公布？"
