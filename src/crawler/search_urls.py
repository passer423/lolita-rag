"""生成微博搜索页链接，用于「批量打开 → 一次 HAR 导出」的语料扩充路径。

放在 ``src/`` 而不是脚本里，是因为它是逻辑（需要被测试），
脚本只负责打印与调用浏览器。
"""

from __future__ import annotations

from urllib.parse import quote

# 选词思路：超话名与品类黑话的内容高度集中。
# 只搜「Lolita」很容易混进小说/电影/无关内容，实测如此。
DEFAULT_KEYWORDS = (
    "LOLITA超话",
    "lolita装扮",
    "lolita日常",
    "lo裙",
    "洛丽塔 穿搭",
    "lolita 茶会",
    "JSK 裙子",
    "痛lo",
)

SEARCH_BASE = "https://s.weibo.com/weibo"

# 实测这些词能筛掉约一半内容（21 → 11 条），留下的知识密度明显更高。
PROMO_EXCLUDE = "出物,抽奖,转发,上新,图透,预约,尾款,定金,定金,转/发"


def build_search_urls(
    keywords,
    pages: int = 5,
    base: str = SEARCH_BASE,
) -> list[str]:
    """生成搜索页链接。``s.weibo.com`` 的分页参数就是 ``page``。"""
    urls: list[str] = []
    for keyword in keywords:
        keyword = str(keyword).strip()
        if not keyword:
            continue
        for page in range(1, max(1, pages) + 1):
            urls.append(f"{base}?q={quote(keyword)}&page={page}")
    return urls
