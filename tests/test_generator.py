"""生成层测试：无 Key 时的抽取式兜底、无资料时的拒答。

这两个行为是「离线也能跑通」和「不硬答」的保证，属于必须锁住的契约。
"""

from __future__ import annotations

from src.config import Settings
from src.generator.llm import Generator
from src.generator.prompt import format_context
from src.schema import Chunk, Retrieved


def make_hits() -> list[Retrieved]:
    chunk = Chunk(
        chunk_id="c1",
        post_id="p1",
        text="带亮片的柄图不能机洗，建议冷水手洗。",
        url="https://example.com/p1",
        author="洗裙子洗哭了",
        created_at="2026-02-15 16:55",
    )
    return [Retrieved(chunk=chunk, score=0.8, source="bm25+vector")]


def test_without_api_key_falls_back_to_extractive():
    generator = Generator(Settings(llm_api_key=""))
    answer = generator.answer("亮片裙子怎么洗？", make_hits())
    assert answer.used_llm is False
    assert "未配置 LLM_API_KEY" in answer.note
    assert "冷水手洗" in answer.text
    assert answer.citations == ["https://example.com/p1"]


def test_no_hits_returns_refusal():
    generator = Generator(Settings(llm_api_key=""))
    answer = generator.answer("语料里没有的问题", [])
    assert answer.used_llm is False
    assert answer.note == "no_context"
    assert "没有检索到" in answer.text


def test_context_formatting_numbers_sources_from_one():
    text = format_context(make_hits())
    assert text.startswith("[1] ")
    assert "作者：洗裙子洗哭了" in text
    assert "链接：https://example.com/p1" in text


def test_llm_error_degrades_instead_of_raising(monkeypatch):
    generator = Generator(Settings(llm_api_key="fake-key", llm_model="fake-model"))

    class BrokenClient:
        class chat:  # noqa: N801 - 模拟 openai 客户端的属性链
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("网络不可达")

    monkeypatch.setattr(generator, "_get_client", lambda: BrokenClient())
    answer = generator.answer("亮片裙子怎么洗？", make_hits())
    assert answer.used_llm is False
    assert "LLM 调用失败" in answer.note
    assert "冷水手洗" in answer.text
