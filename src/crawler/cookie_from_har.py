"""从浏览器 HAR 抓包里提取 Cookie，并可写入 ``.env``。

为什么需要它：手工从 F12 复制 Cookie 很容易漏字段——实测踩过一次，
复制出来的值缺 ``SSOLoginState`` 与 ``_T_WM``，服务端只认游客身份
（``/api/config`` 返回 ``login=false``）。而 **HAR 里存的是浏览器当时发出去的
完整 Cookie 请求头**，不存在漏字段的问题。

顺带一提：HAR 同时也是语料来源（见 ``src/crawler/importer.py``），
所以**存一次 HAR 就同时拿到了语料和 Cookie**。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from src.crawler.weibo_client import audit_cookie, cookie_names

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HarCookie:
    cookie: str = ""
    source_url: str = ""
    candidates: int = 0

    @property
    def found(self) -> bool:
        return bool(self.cookie)

    def present_missing(self) -> tuple[list[str], list[str]]:
        return audit_cookie(self.cookie)

    def render(self) -> str:
        if not self.found:
            return "没有在 HAR 里找到任何微博请求的 Cookie 头"
        present, missing = self.present_missing()
        lines = [
            f"来源请求      : {self.source_url}",
            f"候选 Cookie 数: {self.candidates}",
            f"长度          : {len(self.cookie)} 字符",
            f"常见字段已出现: {'、'.join(present) or '（无）'}",
        ]
        if missing:
            lines.append(f"常见字段缺失  : {'、'.join(missing)}")
        return "\n".join(lines)


def _cookie_domain_rank(url: str) -> int:
    """Cookie 域与采集用的 ``m.weibo.cn`` 的接近程度，越大越优先。

    微博的登录态分在**两个不同的 Cookie 域**上：``.weibo.com``（PC 站）和
    ``.weibo.cn``（移动站 / API）。采集只走 ``m.weibo.cn``，而 Cookie 是按域
    发送的，所以只有 ``.weibo.cn`` 那一族才用得上。

    实测踩过这个坑：只按「字段最多」挑，会选中 ``weibo.com`` 那份（10 个字段，
    看着最全），但它**没有** ``SSOLoginState`` / ``_T_WM``，拿去请求
    ``m.weibo.cn/api/config`` 得到 ``login=false``；而同一份 HAR 里
    ``api.weibo.cn``（同为 ``.weibo.cn`` 域）那份字段更少却能登录。
    """
    host = urlparse(url).netloc.lower()
    if host == "m.weibo.cn" or host.endswith(".m.weibo.cn"):
        return 2
    if host == "weibo.cn" or host.endswith(".weibo.cn"):
        return 1
    return 0


def _cookie_score(url: str, cookie: str) -> tuple[int, int, int]:
    """排序键：域优先级 > 常见字段命中数 > 字段总数。"""
    present, _ = audit_cookie(cookie)
    return (_cookie_domain_rank(url), len(present), len(cookie_names(cookie)))


def extract_cookie_from_har(path: Path | str) -> HarCookie:
    """从 HAR 里挑出最完整的那份 Cookie 请求头。

    优先级：Cookie 域（``m.weibo.cn`` > 其它 ``*.weibo.cn`` > ``weibo.com`` 等）
    > 常见登录字段命中数 > 字段总数。
    只读 ``request.headers``，不碰响应体，因此即使导出时没勾 *with content* 也能用。
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("HAR 读取失败：%s", exc)
        return HarCookie()

    entries = ((payload.get("log") or {}).get("entries")) or []
    if not isinstance(entries, list):
        return HarCookie()

    best = HarCookie()
    best_score = (0, -1, -1)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        request = entry.get("request") or {}
        url = str(request.get("url") or "")
        if "weibo" not in url:
            continue
        for header in request.get("headers") or []:
            if not isinstance(header, dict):
                continue
            if str(header.get("name", "")).lower() != "cookie":
                continue
            value = str(header.get("value") or "").strip()
            if not value:
                continue
            best.candidates += 1
            score = _cookie_score(url, value)
            if score > best_score:
                best.cookie = value
                best.source_url = url
                best_score = score
    return best


def write_cookie_to_env(env_path: Path | str, cookie: str) -> bool:
    """把 Cookie 写进 ``.env`` 的 ``WEIBO_COOKIE=`` 行，其余内容原样保留。

    返回 True 表示替换了已有行，False 表示新增了一行。
    """
    env_path = Path(env_path)
    cookie = cookie.strip().replace("\r", "").replace("\n", "")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    replaced = False
    output: list[str] = []
    for line in lines:
        if line.strip().startswith("WEIBO_COOKIE="):
            output.append(f"WEIBO_COOKIE={cookie}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        if output and output[-1].strip():
            output.append("")
        output.append("# 由 scripts/extract_cookie.py 从 HAR 写入")
        output.append(f"WEIBO_COOKIE={cookie}")

    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return replaced
