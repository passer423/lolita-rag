"""导入「手工保存」的微博响应或页面，作为自动采集的替代路径。

为什么需要这个模块：微博的关键词检索接口必须要登录态（实测未登录返回
``ok: -100``，话题页直接 HTTP 432）。如果你不想把 Cookie 交给别人、
或者不想让它留在聊天记录里，可以走这条路：

    浏览器里登录 m.weibo.cn → 搜索 → 把接口响应 / 页面存成文件 → 交给本模块

**这不需要任何绕过**：用的是你自己会话里已经加载好的公开内容，只是把
「浏览器已经拿到的数据」换个地方保存而已。

支持的输入形态（自动识别）：

* ``.json`` —— container 接口响应 ``{"ok":1,"data":{"cards":[...]}}``；
  也容忍裸的 cards 数组、裸的博文对象数组、单个博文对象
* ``.html`` / ``.htm`` —— m.weibo.cn 页面，从内嵌的 ``$render_data`` 里提取
* ``.jsonl`` —— 本项目 ``data/raw/posts.jsonl`` 的格式

识别不了时不会静默丢弃，而是把「看到了什么结构」写进 ``notes``，
方便据此补适配。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from bs4 import BeautifulSoup

from src.crawler.weibo_client import WeiboClient
from src.preprocessing.clean import clean_text, dedupe_posts
from src.schema import Post
from src.storage import read_jsonl

logger = logging.getLogger(__name__)

_RENDER_DATA_RE = re.compile(r"\$render_data\s*=\s*")
SUPPORTED_SUFFIXES = {".json", ".html", ".htm", ".jsonl", ".txt", ".har"}


@dataclass(slots=True)
class ImportedFile:
    path: Path
    kind: str = "unknown"
    posts: int = 0
    notes: list[str] = field(default_factory=list)
    parsed: list[Post] = field(default_factory=list, repr=False)

    def render(self) -> str:
        detail = f"；{'；'.join(self.notes)}" if self.notes else ""
        return f"  [{self.kind:<12}] {self.posts:>4} 条  {self.path.name}{detail}"


@dataclass(slots=True)
class ImportResult:
    files: list[ImportedFile] = field(default_factory=list)
    posts: list[Post] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"扫描到 {len(self.files)} 个文件，合计解析出 {len(self.posts)} 条博文："]
        lines.extend(item.render() for item in self.files)
        if self.notes:
            lines.append("说明：")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------- 工具
def _extract_balanced(text: str, start: int) -> str | None:
    """从 ``text[start]`` 处开始，取出配平的 ``[...]`` 或 ``{...}``。

    不能用正则：页面里嵌的 JSON 有多层嵌套，非贪婪匹配会在第一个 ``]`` 处截断。
    这里逐字符扫描并跟踪字符串状态，避免被 JSON 里的括号字符骗到。
    """
    if start >= len(text) or text[start] not in "[{":
        return None
    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _posts_from_obj(obj: Any, notes: list[str]) -> list[Post]:
    """把各种可能的结构都尝试解析成 Post 列表。"""
    if isinstance(obj, list):
        posts: list[Post] = []
        for item in obj:
            posts.extend(_posts_from_obj(item, notes))
        return posts

    if not isinstance(obj, dict):
        return []

    # container 接口响应
    data = obj.get("data")
    if isinstance(data, dict) and isinstance(data.get("cards"), list):
        return WeiboClient.parse_container(obj)
    if isinstance(obj.get("cards"), list):
        return WeiboClient.parse_container({"data": obj})

    # 单张卡片：交回给正式解析器，**不要在这里维护第二套卡片逻辑**。
    # 真实微博搜索响应大量使用 card_group 嵌套，之前这里漏掉了它，
    # 导致一页只导入第一条（测试抓出来的）。
    if "card_type" in obj or "mblog" in obj or "card_group" in obj:
        return WeiboClient.parse_container({"data": {"cards": [obj]}})

    if isinstance(obj.get("mblog"), dict):
        parsed = WeiboClient._parse_mblog(obj["mblog"])  # noqa: SLF001 - 复用同一套清洗逻辑
        return [parsed] if parsed else []
    if obj.get("text") and (obj.get("id") or obj.get("mid")):
        parsed = WeiboClient._parse_mblog(obj)  # noqa: SLF001
        return [parsed] if parsed else []

    if notes is not None:
        notes.append(f"未识别的对象结构，顶层键：{sorted(obj.keys())[:8]}")
    return []


def _posts_from_json_text(text: str, notes: list[str]) -> list[Post]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        notes.append(f"JSON 解析失败：{exc}")
        return []
    return _posts_from_obj(payload, notes)


def _posts_from_html(text: str, notes: list[str]) -> list[Post]:
    match = _RENDER_DATA_RE.search(text)
    if not match:
        notes.append("页面里没有找到 $render_data，可能是登录页或页面结构已变")
        return []
    start = match.end()
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    raw = _extract_balanced(text, start)
    if not raw:
        notes.append("$render_data 存在但括号不配平，无法提取")
        return []
    notes.append(f"从 $render_data 提取到 {len(raw)} 个字符")
    return _posts_from_json_text(raw, notes)


def _posts_from_pc_search_html(text: str, notes: list[str]) -> list[Post]:
    """解析 ``s.weibo.com/weibo?q=...`` 这类 PC 搜索页（Ctrl+S 保存下来的那种）。

    与移动端 API 的结构完全不同，选择器来自对真实保存页面的观察：

    * 博文卡片：``div[action-type="feed_list_item"][mid]``
      （用户卡片也带 ``card-wrap`` 但没有 mid/action-type，靠这个区分）
    * 正文：``p[node-type="feed_list_content_full"]`` 优先，
      没有就取 ``p[node-type="feed_list_content"]``（长文时前者才是完整正文）
    * 作者：卡片内任意节点的 ``nick-name`` 属性
    * 时间：``div.from`` 里的第一个链接文本

    ⚠️ PC 页面里的转发/评论/点赞数是用 JS 加载的，**HTML 里没有**，
    因此本方法把这些计数统一记 0——不要以为是解析失败。
    """
    soup = BeautifulSoup(text, "html.parser")
    cards = soup.find_all(attrs={"action-type": "feed_list_item"})
    if not cards:
        card_wraps = soup.find_all(class_="card-wrap")
        notes.append(
            f"没有找到 action-type=feed_list_item 的博文卡片"
            f"（页面里 card-wrap 有 {len(card_wraps)} 个），可能是登录页或结构已变"
        )
        return []

    posts: list[Post] = []
    skipped = 0
    for card in cards:
        mid = str(card.get("mid") or "").strip()
        if not mid or mid == "-":
            skipped += 1
            continue

        # 先剔除纯 UI 控件，否则它们的文字会混进正文：
        #   <a action-type="fl_fold">收起<i class="wbicon">d</i></a>
        #   <a action-type="fl_unfold">展开<i class="wbicon">c</i></a>
        # wbicon 是图标字体，字形本身就是 d/c/L 这类字母，必须一并去掉。
        # 按属性/类名精确剔除，不要用字符串替换——正文里真的可能出现「收起」二字。
        for node in card.select('[action-type="fl_fold"], [action-type="fl_unfold"], i.wbicon'):
            node.decompose()

        body = card.find(attrs={"node-type": "feed_list_content_full"})
        if body is None:
            body = card.find(attrs={"node-type": "feed_list_content"})
        if body is None:
            skipped += 1
            continue
        raw_text = body.get_text(" ", strip=True)
        if not raw_text:
            skipped += 1
            continue

        author = ""
        author_node = card.find(attrs={"nick-name": True})
        if author_node is not None:
            author = str(author_node.get("nick-name") or "").strip()

        created_at = ""
        from_node = card.find("div", class_="from")
        if from_node is not None:
            time_link = from_node.find("a")
            if time_link is not None:
                created_at = time_link.get_text(strip=True)

        posts.append(
            Post(
                id=mid,
                text=clean_text(raw_text),
                author=author,
                created_at=created_at,
                url=f"https://m.weibo.cn/detail/{mid}",
                source="weibo-html",
            )
        )

    notes.append(f"PC 搜索页：识别 {len(cards)} 张博文卡片，跳过 {skipped} 张（无 mid 或无正文）")
    notes.append("PC 页面不含互动数（由 JS 加载），likes/comments/reposts 记为 0")
    return [post for post in posts if post.text]


def _posts_from_har(text: str, notes: list[str]) -> list[Post]:
    """从浏览器导出的 HAR 抓包里提取所有微博响应。

    这是**扩充语料最省事的办法**：在 Chrome 里 F12 → Network → 翻若干页 →
    右键 → *Save all as HAR with content*，一个文件就包含全部页面的响应，
    不必一页一页 Ctrl+S。

    只处理 URL 里带 weibo 且类型是 json/html 的条目，其余（JS/CSS/图片）直接跳过；
    图片即使带内容也是 base64，读进来只会白白占内存。
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        notes.append(f"HAR JSON 解析失败：{exc}")
        return []

    raw_entries = (payload.get("log") or {}).get("entries")
    if raw_entries is None:
        notes.append("HAR 里没有 log.entries，可能不是标准抓包文件")
        return []
    if not isinstance(raw_entries, list):
        notes.append(f"HAR 的 log.entries 类型异常：{type(raw_entries).__name__}")
        return []
    if not raw_entries:
        notes.append(
            "HAR 的 log.entries 是空的：导出时要在 Network 面板先翻页，"
            "并确认勾选了 *with content*"
        )
        return []
    entries = raw_entries

    posts: list[Post] = []
    used: dict[str, int] = {}
    skipped_images = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = str(((entry.get("request") or {}).get("url")) or "")
        content = (entry.get("response") or {}).get("content") or {}
        mime = str(content.get("mimeType") or "").lower()
        body = content.get("text") or ""
        if not body or "weibo" not in url:
            continue
        if content.get("encoding") == "base64" and "json" not in mime and "html" not in mime:
            skipped_images += 1
            continue

        scratch: list[str] = []
        if "json" in mime or "getindex" in url.lower():
            found = _posts_from_json_text(body, scratch)
            label = "container-json"
        elif "html" in mime and "$render_data" in body:
            found = _posts_from_html(body, scratch)
            label = "render_data"
        elif "html" in mime:
            found = _posts_from_pc_search_html(body, scratch)
            label = "pc-search"
        else:
            continue

        if found:
            used[label] = used.get(label, 0) + 1
            posts.extend(found)

    summary = "、".join(f"{key} {value} 个响应" for key, value in sorted(used.items()))
    notes.append(f"HAR：扫描 {len(entries)} 条记录" + (f"，用上 {summary}" if summary else "，没有可用的微博响应"))
    if skipped_images:
        notes.append(f"跳过 {skipped_images} 条二进制内容（图片等）")
    return posts


def load_file(path: Path) -> ImportedFile:
    """解析单个文件；**永不抛异常**，问题都记在 notes 里。"""
    path = Path(path)
    result = ImportedFile(path=path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result.notes.append(f"读取失败：{exc}")
        return result

    suffix = path.suffix.lower()
    stripped = text.lstrip()

    if suffix == ".har":
        posts = _posts_from_har(text, result.notes)
        result.kind = "har"
        result.parsed = posts
        result.posts = len(posts)
        return result

    if suffix == ".jsonl" or (stripped.startswith("{") and "\n{" in text):
        rows = read_jsonl(path)
        posts = [
            Post.from_dict(row)
            for row in rows
            if row.get("text") and row.get("id")
        ]
        result.kind = "posts.jsonl"
        result.parsed = posts
        result.posts = len(posts)
        result.notes.append("按本项目的 posts.jsonl 格式读取")
        return result

    if suffix in {".html", ".htm"} or stripped.startswith("<"):
        if "$render_data" in text:
            posts = _posts_from_html(text, result.notes)
            result.kind = "html/$render_data"
        else:
            # Ctrl+S 保存下来的多半是 s.weibo.com 的 PC 搜索页
            posts = _posts_from_pc_search_html(text, result.notes)
            result.kind = "html/pc-search"
    else:
        posts = _posts_from_json_text(text, result.notes)
        result.kind = "json"

    result.parsed = posts
    result.posts = len(posts)
    return result


def import_paths(
    paths: list[Path | str],
    *,
    exclude: Sequence[str] = (),
) -> ImportResult:
    """导入文件或目录（目录只扫一层支持的后缀）。

    ``exclude`` 里的关键词命中正文就丢弃该条。主要用途是**提高语料的知识密度**：
    实测微博搜索页里大量是「出物/转发抽奖/上新/图透」这类营销或交易贴，
    它们对问答没什么价值，却会稀释检索结果。默认不过滤（保持原样最诚实），
    需要时显式传入。
    """
    result = ImportResult()
    candidates: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
            )
        elif path.exists():
            candidates.append(path)
        else:
            result.notes.append(f"路径不存在，已跳过：{path}")

    if not candidates:
        result.notes.append("没有找到可导入的文件（支持 .json / .html / .har / .jsonl）")
        return result

    collected: list[Post] = []
    for path in candidates:
        item = load_file(path)
        posts = list(item.parsed)
        for post in posts:
            post.text = clean_text(post.text)
        posts = [post for post in posts if post.text]
        item.posts = len(posts)
        item.parsed = posts
        result.files.append(item)
        collected.extend(posts)

    if exclude:
        before = len(collected)
        collected = [
            post for post in collected if not any(word in post.text for word in exclude)
        ]
        dropped = before - len(collected)
        if dropped:
            result.notes.append(f"按关键词排除 {dropped} 条（{'、'.join(exclude)}）")

    before_dedupe = len(collected)
    result.posts = dedupe_posts(collected)
    if before_dedupe != len(result.posts):
        result.notes.append(f"内容去重：{before_dedupe} → {len(result.posts)} 条")
    return result
