"""采集真实微博数据并落盘，支持两种采集语义。

**两种模式刻意分开，因为它们的页码语义完全不同：**

* ``backfill``（补历史）——沿 ``crawl_state.json`` 的 ``next_page`` 往后翻，
  适合一次性把历史数据补厚；中断后重跑会从中断页继续。
* ``daily``（每日更新）——**每晚固定重扫第 1~N 页**。微博搜索结果的新内容
  出现在顶部，沿游标往后翻只会拿到越来越旧的内容；重复抓到的博文靠 id
  去重即可，只有真正的新博文会进入后续的切分与向量化。

``daily`` **不会写 ``next_page``**，因此不会破坏 ``backfill`` 的进度——
这两件事必须能同时存在。

进度账本 ``data/raw/crawl_state.json``：

    {"keywords": {"Lolita": {"next_page": 3, "collected": 57, "updated_at": "..."}},
     "daily":    {"last_run": "...", "keywords": ["Lolita"], "pages": 2, "fetched_total": 120}}

落盘时机：backfill 每翻完一页落一次盘（中断最多丢一页）；daily 在结束时
统一落账本（博文仍然每页落盘）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.config import Settings, settings as default_settings
from src.crawler.weibo_client import WeiboClient
from src.preprocessing.clean import clean_text, dedupe_posts
from src.schema import Post
from src.storage import load_posts, save_posts

logger = logging.getLogger(__name__)

MODE_BACKFILL = "backfill"
MODE_DAILY = "daily"
VALID_MODES = (MODE_BACKFILL, MODE_DAILY)


def load_state(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or default_settings
    path = settings.crawl_state_path
    if not path.exists():
        return {"keywords": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("采集进度文件损坏，将从头开始：%s", path)
        return {"keywords": {}}
    state.setdefault("keywords", {})
    return state


def save_state(state: dict[str, Any], settings: Settings | None = None) -> None:
    settings = settings or default_settings
    path = settings.crawl_state_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_and_save(posts: list[Post], settings: Settings) -> tuple[int, int]:
    """按博文 id 合并进 posts.jsonl（幂等：重复采集同一页不会产生重复数据）。"""
    path = settings.raw_posts_path
    merged: dict[str, Post] = {post.id: post for post in load_posts(path)}
    before = len(merged)
    for post in posts:
        merged[post.id] = post
    save_posts(path, list(merged.values()))
    return before, len(merged)


def _resolve_targets(
    keywords: list[str] | None,
    state: dict[str, Any],
    mode: str,
) -> list[str]:
    if keywords:
        return list(dict.fromkeys(keywords))
    if mode == MODE_DAILY:
        recorded = (state.get("daily") or {}).get("keywords") or []
        targets = [str(k) for k in recorded]
        if not targets:
            raise ValueError(
                "daily 模式没有可用关键词：请用 --keyword 指定，"
                "或先成功跑过一次以写入进度账本。"
            )
        return targets
    targets = list(state["keywords"])
    if not targets:
        raise ValueError("没有指定关键词，进度文件里也没有记录。请用 --keyword 指定关键词。")
    return targets


def merge_and_save(posts: list[Post], settings: Settings | None = None) -> tuple[int, int]:
    """把博文按 id 合并进 ``data/raw/posts.jsonl``，返回 (原有, 合并后)。

    「手工导入」这条路也要落进同一个语料文件，才能被 build_index 直接使用。
    """
    settings = settings or default_settings
    settings.ensure_dirs()
    return _merge_and_save(posts, settings)


def crawl(
    keywords: list[str] | None = None,
    settings: Settings | None = None,
    *,
    pages: int | None = None,
    dry_run: bool = False,
    preview: int = 3,
    resume: bool = True,
    mode: str = MODE_BACKFILL,
    client: Any | None = None,
) -> list[Post]:
    """按关键词采集，返回本次拿到的博文（已清洗去重）。

    ``client`` 可注入（只要实现 ``search_page(keyword, page) -> list[Post]``），
    这样断点续爬与每日更新的逻辑都能脱离网络做单元测试。
    """
    if mode not in VALID_MODES:
        raise ValueError(f"未知采集模式：{mode}（可选 {' / '.join(VALID_MODES)}）")

    settings = settings or default_settings
    settings.ensure_dirs()
    total_pages = pages or settings.crawl_pages

    state = load_state(settings)
    if not resume and mode == MODE_BACKFILL:
        state = {"keywords": {}}
    targets = _resolve_targets(keywords, state, mode)

    if client is None:
        client = WeiboClient(settings)  # 没有 Cookie 时会立刻抛 WeiboCrawlError

    collected: list[Post] = []
    for keyword in targets:
        entry: dict[str, Any] | None = None
        if mode == MODE_BACKFILL:
            entry = state["keywords"].setdefault(
                keyword, {"next_page": 1, "collected": 0, "updated_at": None}
            )
            start_page = 1 if dry_run else int(entry.get("next_page") or 1)
            if start_page > total_pages:
                logger.info(
                    "关键词「%s」已采到第 %s 页，超过本次 --pages=%s，跳过（要重采请加 --restart）",
                    keyword,
                    start_page - 1,
                    total_pages,
                )
                continue
        else:
            # daily：永远从第 1 页开始重扫，靠 id 去重
            start_page = 1

        for page in range(start_page, total_pages + 1):
            batch = client.search_page(keyword, page)
            if not batch:
                logger.info("关键词「%s」第 %s 页无结果，视为已到末页", keyword, page)
                if entry is not None:
                    entry["next_page"] = page
                break

            for post in batch:
                post.text = clean_text(post.text)
            batch = [post for post in batch if post.text]
            collected.extend(batch)

            if entry is not None:
                entry["next_page"] = page + 1
                entry["collected"] = int(entry.get("collected") or 0) + len(batch)
                entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
            logger.info(
                "关键词「%s」第 %s 页：%s 条（%s 模式）",
                keyword,
                page,
                len(batch),
                mode,
            )
            if not dry_run:
                if entry is not None:
                    save_state(state, settings)  # 每页落一次进度：中断最多丢一页
                _merge_and_save(batch, settings)

    if mode == MODE_DAILY and not dry_run:
        daily = state.setdefault("daily", {})
        daily["last_run"] = datetime.now().isoformat(timespec="seconds")
        daily["keywords"] = targets
        daily["pages"] = total_pages
        daily["fetched_total"] = int(daily.get("fetched_total") or 0) + len(collected)
        save_state(state, settings)

    deduped = dedupe_posts(collected)
    logger.info("本次采集 %s 条，清洗去重后 %s 条", len(collected), len(deduped))

    if dry_run:
        for post in deduped[:preview]:
            logger.info("[预览] %s | %s | %s", post.id, post.author, post.text[:60])
        logger.info("dry-run 模式不写盘、不更新进度")
    return deduped
