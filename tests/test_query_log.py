"""查询日志的测试。

这个模块的价值全在两条硬约束上，所以测试也围着它们转：

1. **写日志失败绝不能让检索失败**（日志是附属品，不是主链路）；
2. **评测/排查脚本不进日志**（否则自己的评测会把拒答率这类统计污染掉）。

第 2 条尤其容易在重构中被破坏——比如有人图省事把评测也改成走 ``retrieve()``。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.config import Settings
from src.pipeline.ask import retrieve
from src.pipeline.query_log import (
    build_record,
    log_query,
    read_records,
    summarize,
    write_record,
)
from src.schema import Chunk, Retrieved


def make_hits(count: int = 3) -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(
                chunk_id=f"p{index}#0",
                post_id=f"p{index}",
                text=f"第 {index} 条正文",
                author=f"作者{index}",
            ),
            score=1.0 - 0.1 * index,
            source="bm25+vector" if index % 2 else "vector",
        )
        for index in range(1, count + 1)
    ]


class FakeRetriever:
    def __init__(self, hits: list[Retrieved]):
        self.hits = hits
        self.calls = 0

    def search(self, question: str, top_k: int | None = None, recall_k: int | None = None):
        self.calls += 1
        return self.hits[: (top_k or len(self.hits))]


def make_bundle(hits: list[Retrieved]) -> SimpleNamespace:
    return SimpleNamespace(
        retriever=FakeRetriever(hits),
        chunks=[hit.chunk for hit in hits],
        embedder=SimpleNamespace(name="bge:test-model"),
    )


def make_settings(tmp_path, **overrides) -> Settings:
    return Settings(data_dir=tmp_path, **overrides)


# ------------------------------------------------------------------ 记录内容

def test_record_carries_the_fields_needed_for_analysis(tmp_path):
    settings = make_settings(tmp_path, query_log_top_hits=2)
    record = build_record(
        "柄图是什么",
        make_hits(4),
        settings=settings,
        source="web",
        elapsed_ms=4.5,
        bundle=make_bundle(make_hits(4)),
    )

    assert record.source == "web"
    assert record.query == "柄图是什么"
    assert record.hits == 4
    assert record.refused is False
    assert record.elapsed_ms == 4.5
    assert "柄图" in record.terms
    # top_hits 限制生效：只存摘要，不存全文，否则日志会迅速膨胀
    assert len(record.top) == 2
    assert record.top[0]["chunk_id"] == "p1#0"
    assert record.top[0]["route"] == "bm25+vector"
    # 参数指纹：语料/阈值一变，历史日志就不可比，所以必须带上
    assert record.fingerprint["chunks"] == 4
    assert record.fingerprint["embedder"] == "bge:test-model"
    assert record.fingerprint["min_vector_score"] == settings.min_vector_score


def test_zero_hits_is_recorded_as_a_refusal(tmp_path):
    """拒答（召回 0 条）是 bad case 分析里最值得看的一类，必须能被单独筛出来。"""
    settings = make_settings(tmp_path)
    record = build_record("lo 裙起球了怎么办？", [], settings=settings, source="web")
    assert record.refused is True
    assert record.hits == 0
    assert record.top == []


# ------------------------------------------------------------------ 落盘与读取

def test_write_then_read_roundtrip(tmp_path):
    settings = make_settings(tmp_path)
    write_record(build_record("茶会礼仪有哪些", make_hits(2), settings=settings, source="web"), settings)
    write_record(build_record("机洗可以吗", [], settings=settings, source="cli"), settings)

    rows = read_records(settings.query_log_path)
    assert [row["query"] for row in rows] == ["茶会礼仪有哪些", "机洗可以吗"]
    assert rows[1]["refused"] is True


def test_logging_failure_never_breaks_search(tmp_path):
    """硬约束：把日志路径指向一个**目录**，写入必然失败，但检索必须照常返回。"""
    settings = make_settings(tmp_path)
    blocker = settings.query_log_path
    blocker.mkdir(parents=True)          # 让路径本身是目录，open(...,'a') 必失败

    bundle = make_bundle(make_hits())
    hits = retrieve("柄图是什么", settings, bundle=bundle)

    assert len(hits) == 3, "写日志失败不能影响检索结果"
    assert bundle.retriever.calls == 1


def test_disabled_setting_writes_nothing(tmp_path):
    settings = make_settings(tmp_path, query_log_enabled=False)
    assert log_query("柄图是什么", make_hits(), settings=settings, source="web") is None
    assert not settings.query_log_path.exists()


def test_read_records_skips_broken_lines(tmp_path):
    """日志是追加写的，进程被杀可能留下半行；读取必须扛住，否则分析脚本会崩。"""
    settings = make_settings(tmp_path)
    write_record(build_record("正常一行", make_hits(1), settings=settings), settings)
    with settings.query_log_path.open("a", encoding="utf-8") as handle:
        handle.write('{"query": "被截断的一行"\n')
        handle.write("\n")

    rows = read_records(settings.query_log_path)
    assert [row["query"] for row in rows] == ["正常一行"]


def test_limit_keeps_the_most_recent(tmp_path):
    settings = make_settings(tmp_path)
    for index in range(5):
        write_record(build_record(f"查询{index}", make_hits(1), settings=settings), settings)
    rows = read_records(settings.query_log_path, limit=2)
    assert [row["query"] for row in rows] == ["查询3", "查询4"]


# ------------------------------------------- 谁进日志：这一条防的是「自己污染自己」

def test_retrieve_logs_but_eval_style_direct_search_does_not(tmp_path):
    """``retrieve()`` 写日志；直接调 ``retriever.search``（评测/排查的做法）不写。

    评测要是也进日志，「拒答率」这种统计就会被自己出的题目刷满，
    真实用户搜不到什么反而被淹没——所以这条边界必须锁死。
    """
    settings = make_settings(tmp_path)
    bundle = make_bundle(make_hits())

    retrieve("真实用户的问题", settings, bundle=bundle, source="web")
    bundle.retriever.search("评测集里的问题")          # 评测/explain_retrieval 的走法

    rows = read_records(settings.query_log_path)
    assert [row["query"] for row in rows] == ["真实用户的问题"]
    assert rows[0]["source"] == "web"


def test_ask_also_logs_the_query(tmp_path):
    """只有问题、没有生成层可用时也要留下痕迹——否则「搜了什么」会缺一半。"""
    from src.pipeline.ask import ask

    settings = make_settings(tmp_path)
    bundle = make_bundle(make_hits(1))
    ask("藤上春多少钱", settings, bundle=bundle)

    rows = read_records(settings.query_log_path)
    assert len(rows) == 1 and rows[0]["query"] == "藤上春多少钱"


# ------------------------------------------------------------------ 汇总

def test_summarize_reports_refusal_rate_latency_and_potential_bad_cases(tmp_path):
    settings = make_settings(tmp_path)
    rows = [
        build_record("柄图是什么", make_hits(3), settings=settings, source="web", elapsed_ms=4.0),
        build_record("lo 裙起球了怎么办？", [], settings=settings, source="web", elapsed_ms=6.0),
        build_record("柄图是什么", make_hits(2), settings=settings, source="cli", elapsed_ms=2.0),
    ]
    summary = summarize([row.to_dict() for row in rows])

    assert summary["total"] == 3
    assert summary["refused"] == 1
    assert round(summary["refusal_rate"], 4) == round(1 / 3, 4)
    assert summary["latency_ms"]["p50"] > 0
    assert summary["by_source"] == {"cli": 1, "web": 2}
    assert summary["popular"][0] == ("柄图是什么", 2)
    assert summary["refused_queries"] == ["lo 裙起球了怎么办？"]

    web_only = summarize([row.to_dict() for row in rows], source="web")
    assert web_only["total"] == 2 and web_only["refused"] == 1


def test_records_are_valid_jsonl_on_disk(tmp_path):
    settings = make_settings(tmp_path)
    write_record(build_record("中文查询", make_hits(2), settings=settings, source="web"), settings)
    line = settings.query_log_path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["query"] == "中文查询"
