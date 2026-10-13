"""微博采集客户端。

验证状态（如实记录，别再含糊）：
* 真实接口连通性、登录态自检 ``check_login`` 已在真实网络上验证过；
* **卡片解析逻辑仍未经有效 Cookie 验证**——目前只对着合成夹具做过回归测试。
  拿到有效 Cookie 后第一件事就是 ``python scripts/crawl.py --keyword Lolita --pages 1 --dry-run``。

红线（不要修改）：
* 请求间隔由 ``CRAWL_INTERVAL`` 控制，默认 2.5 秒，**不要为了快而调小**；
* 遇到验证码 / 登录墙直接停止并报错，**不实现任何绕过逻辑**，
  也不做「用账号密码自动登录」这类事情——登录必须由人在浏览器里完成；
* 仅供个人研究使用，采集结果不要二次公开发布。
"""

from __future__ import annotations

import html
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from src.config import Settings, settings as default_settings
from src.preprocessing.clean import clean_text
from src.schema import Post

logger = logging.getLogger(__name__)

CONTAINER_API = "https://m.weibo.cn/api/container/getIndex"
CONFIG_API = "https://m.weibo.cn/api/config"
# 长文全文接口：container/getIndex 对长博文只给截断版（实测 141 字 vs 真实 1365 字），
# 全文要单独取。规模与说明见 pipeline/longtext.py。
LONG_TEXT_API = "https://m.weibo.cn/statuses/extend"
DETAIL_URL = "https://m.weibo.cn/detail/{mid}"
MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
)
_TAG_RE = re.compile(r"<[^>]+>")


class WeiboCrawlError(RuntimeError):
    """采集过程中的可预期错误（缺 Cookie、被风控、返回结构变化等）。"""


def build_session(cookie: str = "", proxy: str = "") -> requests.Session:
    """构造带移动端请求头的会话。Cookie 为空时不带 Cookie 头（用于游客请求）。"""
    session = requests.Session()
    headers = {
        "User-Agent": MOBILE_UA,
        "Referer": "https://m.weibo.cn/",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    if cookie.strip():
        headers["Cookie"] = cookie.strip()
    session.headers.update(headers)
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


@dataclass(slots=True)
class LoginStatus:
    """Cookie 的登录态判定结果。"""

    ok: bool                     # 接口本身是否正常响应
    logged_in: bool              # 服务端是否认为这是登录会话（data.login）
    uid: str = ""
    screen_name: str = ""
    detail: str = ""
    present: list[str] = field(default_factory=list)   # 已出现的常见字段
    missing: list[str] = field(default_factory=list)   # 缺失的常见字段

    def render(self) -> str:
        if self.logged_in:
            who = self.screen_name or self.uid or "未知账号"
            return f"登录态有效：{who}"
        return f"登录态无效：{self.detail or '未知原因'}"

    def cookie_hint(self) -> str:
        """缺失字段的提示——手工复制 Cookie 最常见的失败原因就是漏字段。"""
        if not self.missing:
            return ""
        return (
            "当前 Cookie 里没有这些常见字段："
            + "、".join(self.missing)
            + "（多半是复制不完整；HAR 抓包里的 Cookie 请求头才是完整的）"
        )


# 一个「正常登录会话」通常具备的字段。不是硬性要求（微博会变），
# 但缺得越多，越可能是复制不完整而不是会话过期。
EXPECTED_COOKIE_NAMES = (
    "SUB",
    "SUBP",
    "SSOLoginState",
    "ALF",
    "_T_WM",
    "WBPSESS",
    "XSRF-TOKEN",
)


def cookie_names(cookie: str) -> set[str]:
    """取出 Cookie 字符串里出现的字段名。"""
    names: set[str] = set()
    for part in (cookie or "").split(";"):
        name = part.strip().split("=", 1)[0].strip()
        if name:
            names.add(name)
    return names


def audit_cookie(cookie: str) -> tuple[list[str], list[str]]:
    """返回 (常见的已出现字段, 常见的缺失字段)，都已排序。"""
    present = cookie_names(cookie)
    expected = set(EXPECTED_COOKIE_NAMES)
    return sorted(present & expected), sorted(expected - present)


def check_login(
    settings: Settings | None = None,
    session: requests.Session | None = None,
) -> LoginStatus:
    """用**一个**请求判定 Cookie 是否为有效登录会话。

    这是「失效即停止」策略的入口：定时任务每晚先花 1 个请求做这个检查，
    判定为未登录就直接退出，不去浪费采集请求。

    依据是 ``m.weibo.cn/api/config`` 的 ``data.login`` 字段。实测一个失效的
    Cookie 会得到 ``ok=1`` 但 ``login=false``——所以**不能只看 ok**。
    """
    settings = settings or default_settings
    if not settings.weibo_cookie.strip():
        return LoginStatus(ok=False, logged_in=False, detail="未配置 WEIBO_COOKIE")

    present, missing = audit_cookie(settings.weibo_cookie)
    session = session or build_session(settings.weibo_cookie, settings.http_proxy)
    try:
        response = session.get(CONFIG_API, timeout=settings.crawl_timeout)
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - 检查失败不应抛异常，交由调用方决定
        return LoginStatus(
            ok=False,
            logged_in=False,
            detail=f"登录态检查失败：{exc}",
            present=present,
            missing=missing,
        )

    data = payload.get("data") or {}
    logged_in = bool(data.get("login"))
    detail = ""
    if not logged_in:
        detail = (
            "Cookie 不是有效登录会话（/api/config 返回 login=false）。"
            "请确认浏览器里确实处于登录状态（能看到自己的头像），"
            "再复制完整 Cookie；用 scripts/extract_cookie.py 从 HAR 抓包里取最稳妥"
        )
    return LoginStatus(
        ok=payload.get("ok") == 1,
        logged_in=logged_in,
        uid=str(data.get("uid") or ""),
        screen_name=str(data.get("screen_name") or ""),
        detail=detail,
        present=present,
        missing=missing,
    )


class WeiboClient:
    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None):
        self.settings = settings or default_settings
        if not self.settings.weibo_cookie.strip():
            raise WeiboCrawlError(
                "未配置 WEIBO_COOKIE。请在 .env 中填入登录后的 Cookie；"
                "只想跑通流程请改用 mock 语料（scripts/build_index.py --source mock）。"
            )
        self.session = session or build_session(self.settings.weibo_cookie, self.settings.http_proxy)
        self._last_request_at = 0.0

    # ------------------------------------------------------------------ 基础请求
    def _throttle(self) -> None:
        """强制最小请求间隔，并加一点随机抖动，避免形成固定频率特征。"""
        wait = self.settings.crawl_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        time.sleep(random.uniform(0, 0.6))

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.crawl_max_retry + 1):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=self.settings.crawl_timeout)
                self._last_request_at = time.monotonic()
                if response.status_code != 200:
                    raise WeiboCrawlError(
                        f"接口返回 HTTP {response.status_code}，登录态可能已失效或触发了风控。"
                        "请更新 WEIBO_COOKIE，或调大 CRAWL_INTERVAL 后重试。"
                    )
                if "登录" in response.text[:500] and '"ok":1' not in response.text[:500]:
                    # 微博被风控时常返回 200 + 登录页，状态码看不出来，只能看正文
                    raise WeiboCrawlError("返回内容像是登录页，Cookie 可能已过期，请重新获取。")
                payload = response.json()
                if payload.get("ok") != 1:
                    data = payload.get("data")
                    # 微博在「这个关键词已经没有结果了」时返回的是
                    #   {"ok": 0, "msg": "这里还没有内容", "data": {"cards": []}}
                    # ——这是**正常的尽头信号，不是故障**。早期实现把它当异常抛出，
                    # 后果很严重：整轮采集直接中断，后面的关键词一个都采不到。
                    # 长尾关键词（知识型搜索就是）往往一两页就见底，所以这个假故障
                    # 会把「多关键词批量采集」基本废掉。
                    # 这里按「空页」放行，由调用方按末页处理；其余 ok=0 仍然报错。
                    if isinstance(data, dict) and data.get("cards") == []:
                        logger.info(
                            "关键词已无更多结果（接口返回 %r），按空页处理",
                            payload.get("msg") or "ok=0 且 cards 为空",
                        )
                        return payload
                    raise WeiboCrawlError(f"接口返回异常：{str(payload)[:200]}")
                return payload
            except WeiboCrawlError:
                raise
            except Exception as exc:  # noqa: BLE001 - 需要把网络异常统一成重试
                last_error = exc
                backoff = min(2 ** attempt, 30)
                logger.warning("第 %s 次请求失败（%s），%s 秒后重试", attempt, exc, backoff)
                time.sleep(backoff)
        raise WeiboCrawlError(f"请求连续失败 {self.settings.crawl_max_retry} 次：{last_error}")

    # ------------------------------------------------------------------ 解析
    @staticmethod
    def _parse_mblog(mblog: dict[str, Any]) -> Post | None:
        text_raw = mblog.get("text") or ""
        if not text_raw:
            return None
        mid = str(mblog.get("id") or mblog.get("mid") or "")
        if not mid:
            return None
        # 清洗后为空（纯空白、纯表情）的博文直接丢弃，否则会变成空片段污染索引
        cleaned = clean_text(_TAG_RE.sub(" ", html.unescape(text_raw)))
        if not cleaned:
            return None
        user = mblog.get("user") or {}
        return Post(
            id=mid,
            text=cleaned,
            author=str(user.get("screen_name") or ""),
            created_at=str(mblog.get("created_at") or ""),
            url=DETAIL_URL.format(mid=mid),
            likes=int(mblog.get("attitudes_count") or 0),
            comments=int(mblog.get("comments_count") or 0),
            reposts=int(mblog.get("reposts_count") or 0),
            is_repost=bool(mblog.get("retweeted_status")),
            source="weibo",
        )

    @classmethod
    def parse_container(cls, payload: dict[str, Any]) -> list[Post]:
        """从 container 接口响应里提取博文，过滤掉广告与超话卡片。

        必须**递归**遍历 cards / card_group：真实响应里博文有两种摆法——直接挂在
        顶层卡片上（``card_type=9``），或者藏在一张**包装卡片**里。搜索接口第 2 页
        起就是后者：顶层是 ``card_type=11``，真正的博文在它的 ``card_group[0]``
        （``card_type=9``）。

        早期版本只认顶层 ``card_type == 9``，于是第 2 页起解析结果恒为空、
        被判成「已到末页」——**采集永远只拿到第 1 页**。实测证据：8 个关键词各只
        得到 13 条左右，而 ``cardlistInfo.total`` 是 751；同一个关键词直接请求
        ``page=2`` 明明返回了 10 条（``page_size=10``）。

        同一条博文若同时出现在包装卡片和分组里，按 ``id`` 去重。
        """
        posts: list[Post] = []
        seen: set[str] = set()

        def walk(node: Any) -> None:
            if isinstance(node, list):
                for item in node:
                    walk(item)
                return
            if not isinstance(node, dict):
                return
            # card_type 实测既有整数 9 也有字符串 "9"，两种都要认
            if node.get("card_type") in (9, "9"):
                mblog = node.get("mblog")
                if isinstance(mblog, dict):
                    parsed = cls._parse_mblog(mblog)
                    if parsed and parsed.id not in seen:
                        seen.add(parsed.id)
                        posts.append(parsed)
            for child in node.get("card_group") or []:
                walk(child)

        walk((payload.get("data") or {}).get("cards") or [])
        return posts

    # ------------------------------------------------------------------ 对外接口
    def search_page(self, keyword: str, page: int) -> list[Post]:
        """采集**单页**。

        分页状态由调用方（``pipeline.crawl``）负责持久化，这样才能做到断点续爬；
        本方法刻意不自己循环翻页。

        开了 ``crawl_fetch_longtext`` 时，会把本页被截断的长博文补成全文——
        实测 22.4% 的采到的博文是截断的，而且**都是最长、信息量最大的那批**。
        """
        payload = self._get_json(
            CONTAINER_API,
            {"containerid": f"100103type=1&q={keyword}", "page_type": "searchall", "page": page},
        )
        posts = self.parse_container(payload)
        if self.settings.crawl_fetch_longtext and posts:
            posts = self._fill_long_text(posts)
        return posts

    def fetch_long_text(self, post_id: str) -> str:
        """取长博文全文；取不到就返回空串（调用方负责标注截断）。

        返回的 HTML 片段交给调用方清洗——这里只负责拿内容。
        """
        try:
            payload = self._get_json(LONG_TEXT_API, {"id": post_id})
        except WeiboCrawlError as exc:
            logger.warning("取全文失败（%s）：%s", post_id, exc)
            return ""
        data = payload.get("data") or {}
        return str(data.get("longTextContent") or "")

    def _fill_long_text(self, posts: list[Post]) -> list[Post]:
        """把本页里被截断的博文换成全文（失败的标注为已截断）。"""
        from src.pipeline.longtext import is_truncated, refresh_long_posts

        if not any(is_truncated(post.text) for post in posts):
            return posts
        refreshed, stats = refresh_long_posts(posts, self.fetch_long_text)
        if stats.candidates:
            logger.info(
                "本页长文补全：待补 %s，成功 %s，失败 %s",
                stats.candidates,
                stats.fetched,
                stats.failed,
            )
        return refreshed

    def search(self, keyword: str, pages: int | None = None) -> list[Post]:
        """一次性采集多页（便利方法，内部不保存进度，仅用于快速验证）。"""
        pages = pages or self.settings.crawl_pages
        collected: list[Post] = []
        seen: set[str] = set()
        for page in range(1, pages + 1):
            batch = self.search_page(keyword, page)
            if not batch:
                logger.info("第 %s 页没有解析到博文，可能已到末页或接口结构变化", page)
                break
            fresh = [p for p in batch if p.id not in seen]
            seen.update(p.id for p in fresh)
            logger.info("第 %s 页解析到 %s 条（新增 %s 条）", page, len(batch), len(fresh))
            collected.extend(fresh)
        return collected
