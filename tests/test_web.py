"""Web 检索接口测试。

用注入的假 bundle，因此不加载 BGE、不碰 chroma。重点验证：
* 页面能正常返回，且**不引用任何外部资源**（离线可用）；
* 检索接口的排序、条数、参数校验；
* 生成层失败时**不能连检索结果一起丢**。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.schema import Chunk, Retrieved
from src.web.app import create_app


class FakeRetriever:
    def __init__(self, hits: list[Retrieved]):
        self.hits = hits
        self.calls: list[tuple[str, int | None]] = []

    def search(self, question: str, top_k: int | None = None, recall_k: int | None = None):
        self.calls.append((question, top_k))
        return self.hits[: (top_k or len(self.hits))]


def make_hits(count: int = 2) -> list[Retrieved]:
    hits: list[Retrieved] = []
    for index in range(1, count + 1):
        hits.append(
            Retrieved(
                chunk=Chunk(
                    chunk_id=f"p{index}#0",
                    post_id=f"p{index}",
                    text=f"第 {index} 条关于柄图和裙撑的测试内容。",
                    author=f"作者{index}",
                    created_at="2026-03-01 10:00",
                    url=f"https://example.com/p{index}",
                ),
                score=0.03 / index,
                source="bm25+vector" if index == 1 else "vector",
            )
        )
    return hits


def make_client(hits=None, *, data_dir=None, **settings_kw):
    hits = hits if hits is not None else make_hits()
    bundle = SimpleNamespace(
        retriever=FakeRetriever(hits),
        chunks=[hit.chunk for hit in hits],
        backend="simple",
    )
    kwargs = {"llm_api_key": ""}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    kwargs.update(settings_kw)
    return TestClient(create_app(bundle=bundle, settings=Settings(**kwargs))), bundle


def test_json_responses_declare_utf8_charset():
    """缺 charset 时 PowerShell 5.1 会把中文 JSON 解成乱码，命令行验证会误判成服务坏了。"""
    client, _ = make_client(make_hits(1))
    with client:
        response = client.get("/api/search", params={"q": "柄图"})
    assert "charset=utf-8" in response.headers["content-type"].lower()


def test_index_page_is_served():
    client, _ = make_client()
    with client:
        response = client.get("/")
    assert response.status_code == 200
    assert "检索" in response.text
    assert 'id="q"' in response.text


def test_page_has_no_external_resources():
    """离线环境也必须能打开：不能引用 CDN 的脚本/字体/样式。"""
    client, _ = make_client()
    with client:
        html = client.get("/").text
    assert "<script src=" not in html
    assert "<link " not in html
    assert "cdn." not in html.lower()


def test_health_reports_readiness():
    client, _ = make_client(make_hits(3))
    with client:
        assert client.get("/api/health").json() == {"status": "ok", "chunks": 3, "ready": True}


def test_index_info_reports_counts_and_settings():
    client, _ = make_client(make_hits(2))
    with client:
        data = client.get("/api/index").json()
    assert data["posts"] == 2
    assert data["chunks"] == 2
    assert data["llm_enabled"] is False
    assert data["top_k"] == Settings(llm_api_key="").top_k


def test_index_info_flags_mock_corpus(tmp_path):
    manifest_dir = tmp_path / "index"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {"source": "mock", "backend": "simple", "embedder": "hash:512", "dim": 512},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client, _ = make_client(make_hits(1), data_dir=tmp_path)
    with client:
        assert client.get("/api/index").json()["is_mock"] is True


def test_index_info_does_not_flag_real_corpus(tmp_path):
    manifest_dir = tmp_path / "index"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"source": "data\\raw\\posts.jsonl", "embedder": "bge:x"}), encoding="utf-8"
    )
    client, _ = make_client(make_hits(1), data_dir=tmp_path)
    with client:
        data = client.get("/api/index").json()
    assert data["is_mock"] is False
    assert data["source"].startswith("data")


def test_index_info_flags_a_stale_index(tmp_path):
    """每日定时更新会重写索引，但服务是启动时装载的——不重启就一直用旧的。

    实测踩过：夜间更新跑完，页面上的片段数还是旧的，看起来像「自动更新没生效」。
    """
    manifest_dir = tmp_path / "index"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"source": "data\\raw\\posts.jsonl", "chunks": 99}), encoding="utf-8"
    )
    client, _ = make_client(make_hits(2), data_dir=tmp_path)   # 内存里只有 2 个片段
    with client:
        data = client.get("/api/index").json()

    assert data["chunks"] == 2
    assert data["manifest_chunks"] == 99
    assert data["stale"] is True


def test_index_info_is_not_stale_when_counts_match(tmp_path):
    manifest_dir = tmp_path / "index"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"source": "data\\raw\\posts.jsonl", "chunks": 2}), encoding="utf-8"
    )
    client, _ = make_client(make_hits(2), data_dir=tmp_path)
    with client:
        assert client.get("/api/index").json()["stale"] is False


def test_search_returns_ranked_hits_and_highlight_terms():
    client, _ = make_client(make_hits(3))
    with client:
        data = client.get("/api/search", params={"q": "柄图", "top_k": 3}).json()

    assert data["query"] == "柄图"
    assert data["count"] == 3
    assert [row["rank"] for row in data["hits"]] == [1, 2, 3]
    assert data["hits"][0]["post_id"] == "p1"
    assert data["hits"][0]["retriever"] == "bm25+vector"
    # 分词结果回传前端做高亮；「柄图」应当被切出来
    assert "柄图" in data["terms"]


def test_search_respects_top_k():
    client, bundle = make_client(make_hits(5))
    with client:
        data = client.get("/api/search", params={"q": "柄图", "top_k": 2}).json()

    assert data["count"] == 2
    assert bundle.retriever.calls == [("柄图", 2)]


def test_search_rejects_blank_query():
    client, _ = make_client()
    with client:
        response = client.get("/api/search", params={"q": "   "})
    assert response.status_code == 400
    assert "不能为空" in response.json()["detail"]


def test_search_without_query_is_a_validation_error():
    client, _ = make_client()
    with client:
        assert client.get("/api/search").status_code == 422


def test_generate_without_api_key_falls_back_to_extractive():
    client, _ = make_client(make_hits(2))
    with client:
        data = client.get("/api/search", params={"q": "柄图", "generate": "true"}).json()

    assert data["count"] == 2, "生成不影响检索结果"
    assert data["answer"]["used_llm"] is False
    assert "LLM_API_KEY" in data["answer"]["note"]
    assert "柄图" in data["answer"]["text"]
    assert data["answer"]["citations"][0] == "https://example.com/p1"


def test_answer_payload_exposes_runtime_gate_verdict():
    """界面要显示「这条是否被降级」，所以判定结果必须随 payload 出去。

    抽取式答案不走校验（它本来就是原文照抄），但字段必须在——
    前端靠 `verified === false` 区分「没检查」和「检查通过」。
    """
    client, _ = make_client(make_hits(2))
    with client:
        answer = client.get("/api/search", params={"q": "柄图", "generate": "true"}).json()["answer"]

    assert answer["verified"] is True
    assert answer["degraded"] is False
    assert answer["problems"] == []


def test_generate_on_empty_result_refuses():
    client, _ = make_client([])
    with client:
        data = client.get("/api/search", params={"q": "语料外的问题", "generate": "true"}).json()

    assert data["count"] == 0
    assert data["hits"] == []
    assert "没有检索到" in data["answer"]["text"]


def test_generation_failure_does_not_lose_search_hits(monkeypatch):
    """生成层挂了也必须把检索结果返回给用户，否则页面上一片空白、无法排查。"""

    class BrokenGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def answer(self, question, hits):
            raise RuntimeError("模型服务不可达")

    monkeypatch.setattr("src.web.app.Generator", BrokenGenerator)
    client, _ = make_client(make_hits(2))
    with client:
        response = client.get("/api/search", params={"q": "柄图", "generate": "true"})

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert "生成失败" in data["answer"]["note"]


def test_search_failure_returns_500_with_reason():
    class ExplodingRetriever(FakeRetriever):
        def search(self, question, top_k=None, recall_k=None):
            raise RuntimeError("向量库损坏")

    bundle = SimpleNamespace(retriever=ExplodingRetriever([]), chunks=[], backend="simple")
    client = TestClient(create_app(bundle=bundle, settings=Settings(llm_api_key="")))
    with client:
        response = client.get("/api/search", params={"q": "柄图"})

    assert response.status_code == 500
    assert "向量库损坏" in response.json()["detail"]


@pytest.mark.parametrize("top_k", [1, 10])
def test_top_k_boundaries_are_accepted(top_k: int):
    client, _ = make_client(make_hits(2))
    with client:
        assert client.get("/api/search", params={"q": "柄图", "top_k": top_k}).status_code == 200


def test_top_k_above_limit_is_rejected():
    client, _ = make_client()
    with client:
        assert client.get("/api/search", params={"q": "柄图", "top_k": 999}).status_code == 422
