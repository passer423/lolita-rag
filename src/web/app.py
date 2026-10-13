"""检索用的本地 Web 服务（FastAPI）。

三个刻意的设计决定：

1. **索引只在启动时装一次**并缓存在 ``app.state``：加载 BGE 约 5 秒，
   每个请求重新加载不可接受；
2. 检索接口写成**同步 ``def``**：FastAPI 会把它丢进线程池执行，
   向量编码是 CPU 密集操作，写成 ``async`` 反而会堵住事件循环；
3. 前端不引用任何 CDN 资源，离线也能正常打开。

``create_app`` 支持注入 ``bundle``，因此接口层可以完全脱离 BGE/chroma 做单元测试。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from src.config import Settings, settings as default_settings
from src.generator.llm import Generator
from src.generator.verify import verify_and_degrade
from src.pipeline.ask import hits_to_dicts, retrieve
from src.pipeline.runtime import read_manifest
from src.retriever.bm25 import tokenize

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class UTF8JSONResponse(JSONResponse):
    """显式声明 ``charset=utf-8``。

    JSON 按 RFC 8259 本来就是 UTF-8，浏览器与 requests/httpx 都能正确处理；
    但 Windows PowerShell 5.1 的 ``Invoke-WebRequest`` 在缺少 charset 时会按 GBK
    解码，中文全变乱码——在命令行验证接口的人会误以为服务坏了。实测如此。
    """

    media_type = "application/json; charset=utf-8"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if getattr(app.state, "bundle", None) is None:
        logger.info("正在装载索引……")
        app.state.bundle = _load(app)
        logger.info("索引就绪：%s 个片段", len(getattr(app.state.bundle, "chunks", []) or []))
    yield


def _load(app: FastAPI):
    from src.pipeline.runtime import load_retriever

    return load_retriever(app.state.settings)


def _chunks(app: FastAPI) -> list:
    return list(getattr(getattr(app.state, "bundle", None), "chunks", []) or [])


def _posts(app: FastAPI) -> list:
    """读博文（而不是片段）——热门推荐要用互动数与发布时间，这些只在博文层。

    按 mtime 缓存：一次解析 1478 条 JSONL 约十几毫秒，但没必要每个请求都做一遍；
    语料重建后 mtime 会变，缓存自动失效。
    """
    path = app.state.settings.raw_posts_path
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cache = app.state.posts_cache
    if cache.get("mtime") != mtime:
        from src.storage import load_posts

        cache["posts"] = load_posts(path)
        cache["mtime"] = mtime
    return cache.get("posts") or []


def _document_frequency(app: FastAPI):
    """语料里的文档频率（给「建议查询」挑罕见词用）；拿不到就返回 None。"""
    retriever = getattr(getattr(app.state, "bundle", None), "retriever", None)
    bm25 = getattr(retriever, "bm25", None)
    frequency = getattr(bm25, "document_frequency", None)
    return frequency if callable(frequency) else None


def create_app(
    *,
    bundle: Any | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    app = FastAPI(
        title="lolita-rag 检索",
        version="0.1.0",
        lifespan=_lifespan,
        default_response_class=UTF8JSONResponse,
    )
    app.state.settings = settings or default_settings
    app.state.bundle = bundle
    app.state.posts_cache = {}

    # ------------------------------------------------------------------ 页面
    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail=f"缺少前端文件：{page}")
        return FileResponse(page, media_type="text/html; charset=utf-8")

    # ------------------------------------------------------------------ 元信息
    @app.get("/api/health")
    def health() -> dict:
        chunks = _chunks(app)
        return {"status": "ok", "chunks": len(chunks), "ready": bool(chunks)}

    @app.get("/api/index")
    def index_info() -> dict:
        settings_ = app.state.settings
        bundle_ = getattr(app.state, "bundle", None)
        chunks = _chunks(app)
        manifest = read_manifest(settings_)
        source = str(manifest.get("source") or "")
        return {
            "posts": len({chunk.post_id for chunk in chunks}),
            "chunks": len(chunks),
            "backend": getattr(bundle_, "backend", "") or str(manifest.get("backend") or ""),
            "embedder": str(manifest.get("embedder") or ""),
            "dim": int(manifest.get("dim") or 0),
            "built_at": str(manifest.get("built_at") or ""),
            "source": source,
            # 明确告诉使用者当前索引是什么语料，避免把 mock 演示数据当成真实结果
            "is_mock": source == "mock",
            "top_k": settings_.top_k,
            "recall_k": settings_.recall_k,
            "fusion_mode": settings_.fusion_mode,
            "min_vector_score": settings_.min_vector_score,
            "min_bm25_ratio": settings_.min_bm25_ratio,
            "llm_enabled": settings_.llm_enabled,
        }

    # ------------------------------------------------------------------ 热门推荐
    @app.get("/api/hot")
    def api_hot(
        days: int | None = Query(None, ge=1, le=30, description="时间窗口（天）"),
        limit: int | None = Query(None, ge=1, le=20, description="返回条数"),
    ) -> dict:
        """「当日热门」——锚定语料最新一天，不是墙上时钟（见 pipeline/hot_posts.py）。"""
        from src.pipeline.hot_posts import select_hot_posts

        settings_ = app.state.settings
        posts = _posts(app)
        selection = select_hot_posts(
            posts,
            window_days=days or settings_.hot_window_days,
            limit=limit or settings_.hot_limit,
            per_author_cap=settings_.hot_per_author_cap,
            document_frequency=_document_frequency(app),
        )
        payload = selection.to_dict()
        payload["total_posts"] = len(posts)
        if not posts:
            # 空栏必须说清为什么，否则用户会以为功能坏了
            payload["hint"] = (
                f"读不到博文语料（{settings_.raw_posts_path}）。"
                "热门推荐基于采集到的原帖（要互动数与发布时间），不是索引片段。"
            )
        elif not payload["items"]:
            payload["hint"] = (
                f"语料最新一天（{selection.anchor_label}）里没有同时满足"
                "「有互动、够长、与 Lolita 相关」条件的帖子。"
            )
        return payload

    # ------------------------------------------------------------------ 检索
    @app.get("/api/search")
    def api_search(
        q: str = Query(..., min_length=1, description="查询词或问题"),
        top_k: int | None = Query(None, ge=1, le=50, description="返回条数"),
        generate: bool = Query(False, description="是否调用生成层产出答案"),
    ) -> dict:
        query = q.strip()
        if not query:
            raise HTTPException(status_code=400, detail="查询不能为空")

        try:
            # source="web" 会写进查询日志（data/logs/queries.jsonl）——
            # 这是唯一不受「自己出评测题」影响的真实查询来源。
            hits = retrieve(
                query,
                app.state.settings,
                top_k=top_k,
                bundle=app.state.bundle,
                source="web",
            )
        except Exception as exc:  # noqa: BLE001 - 统一转成 500 并带上原因
            logger.exception("检索失败")
            raise HTTPException(status_code=500, detail=f"检索失败：{exc}") from exc

        payload: dict[str, Any] = {
            "query": query,
            "count": len(hits),
            # 把 jieba 分词结果一起返回，前端用它做关键词高亮（中文按词高亮才准）
            "terms": tokenize(query),
            "hits": hits_to_dicts(hits),
        }

        if generate:
            try:
                answer = Generator(app.state.settings).answer(query, hits)
                # 运行时闸门：不达标的模型答案换成原文摘录，并把原因告诉前端
                answer, outcome = verify_and_degrade(query, answer, list(hits), app.state.settings)
                payload["answer"] = {
                    "text": answer.text,
                    "used_llm": answer.used_llm,
                    "note": answer.note,
                    "citations": answer.citations,
                    "verified": outcome.checked,
                    "degraded": outcome.degraded,
                    "problems": outcome.problems,
                }
            except Exception as exc:  # noqa: BLE001 - 生成失败不应让检索结果一起丢失
                logger.exception("生成失败")
                payload["answer"] = {
                    "text": "",
                    "used_llm": False,
                    "note": f"生成失败：{exc}",
                    "citations": [],
                    "verified": False,
                    "degraded": False,
                    "problems": [],
                }
        return payload

    return app
