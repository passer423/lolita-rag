"""检索层与输出的测试。

「只检索不生成」是有意做成独立开关的：先确认语料里到底有没有相关内容，
比直接看一段模型总结更有用——召回为空说明语料里没有，
召回一堆不相关的则说明该调阈值或加精排。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.pipeline.ask import hits_to_dicts, render_hits, retrieve
from src.schema import Chunk, Retrieved


class FakeRetriever:
    def __init__(self, hits: list[Retrieved]):
        self.hits = hits
        self.calls: list[tuple[str, int | None]] = []

    def search(self, question: str, top_k: int | None = None, recall_k: int | None = None):
        self.calls.append((question, top_k))
        return self.hits[: (top_k or len(self.hits))]


def make_hits() -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(
                chunk_id="p1#0",
                post_id="p1",
                text="带亮片的柄图不能机洗，建议冷水手洗。",
                author="洗裙子洗哭了",
                created_at="2026-02-15 16:55",
                url="https://e/p1",
            ),
            score=0.0328,
            source="bm25+vector",
        ),
        Retrieved(
            chunk=Chunk(chunk_id="p2#0", post_id="p2", text="茶会礼仪整理。", url="https://e/p2"),
            score=0.0156,
            source="vector",
        ),
    ]


def bundle_with(hits: list[Retrieved]):
    return SimpleNamespace(retriever=FakeRetriever(hits), chunks=[])


def test_retrieve_returns_hits_and_does_not_need_an_llm():
    bundle = bundle_with(make_hits())
    hits = retrieve("亮片裙子怎么洗", Settings(top_k=5), bundle=bundle)

    assert len(hits) == 2
    assert bundle.retriever.calls == [("亮片裙子怎么洗", None)]


def test_retrieve_respects_top_k():
    bundle = bundle_with(make_hits())
    hits = retrieve("亮片裙子怎么洗", Settings(top_k=5), top_k=1, bundle=bundle)

    assert len(hits) == 1
    assert bundle.retriever.calls == [("亮片裙子怎么洗", 1)]


def test_retrieve_rejects_blank_query():
    with pytest.raises(ValueError, match="不能为空"):
        retrieve("   ", bundle=bundle_with([]))


def test_render_hits_shows_rank_score_route_and_source():
    text = render_hits(make_hits(), "亮片")

    assert "命中 2 条" in text
    assert "[1]" in text
    assert "0.0328" in text
    assert "bm25+vector" in text
    assert "洗裙子洗哭了" in text
    assert "https://e/p1" in text


def test_render_hits_handles_empty_result_without_crashing():
    assert "没有检索到" in render_hits([], "语料外的问题")


def test_render_hits_marks_multipart_post():
    chunk = Chunk(chunk_id="p9#1", post_id="p9", text="第二段内容", position=1, total=3)
    text = render_hits([Retrieved(chunk=chunk, score=0.5, source="bm25")], "q")

    assert "第 2/3 段" in text


def test_hits_to_dicts_is_json_serializable_and_numbered():
    payload = json.dumps(hits_to_dicts(make_hits()), ensure_ascii=False)
    data = json.loads(payload)

    assert [row["rank"] for row in data] == [1, 2]
    assert data[0]["post_id"] == "p1"
    assert data[0]["retriever"] == "bm25+vector"
    assert data[1]["author"] == ""      # 缺失字段保持空串，不抛异常


def test_render_hits_handles_unknown_source_gracefully():
    chunk = Chunk(chunk_id="p3#0", post_id="p3", text="内容")
    text = render_hits([Retrieved(chunk=chunk, score=0.1, source="bm25")], "q")
    assert "未知来源" in text
