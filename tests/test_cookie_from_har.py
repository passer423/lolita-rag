"""从 HAR 提取 Cookie 的测试。

这是「手工复制 Cookie 老漏字段」的替代方案，所以两个点必须锁住：
能挑出最完整的那份，以及**输出里绝不能泄漏 Cookie 内容**。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.crawler.cookie_from_har import extract_cookie_from_har, write_cookie_to_env

FULL_COOKIE = "SUB=abc; SUBP=def; SSOLoginState=123; ALF=456; _T_WM=789; WBPSESS=xyz; XSRF-TOKEN=tok"
PARTIAL_COOKIE = "SUB=abc; SUBP=def; WBPSESS=xyz"


def _har(entries: list[dict]) -> str:
    return json.dumps({"log": {"entries": entries}}, ensure_ascii=False)


def _entry(url: str, cookie: str | None) -> dict:
    headers = [{"name": "User-Agent", "value": "x"}]
    if cookie is not None:
        headers.append({"name": "Cookie", "value": cookie})
    return {"request": {"url": url, "headers": headers}, "response": {"content": {}}}


def write_har(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "capture.har"
    path.write_text(_har(entries), encoding="utf-8")
    return path


def test_extracts_cookie_from_har(tmp_path):
    path = write_har(tmp_path, [_entry("https://m.weibo.cn/api/config", FULL_COOKIE)])
    result = extract_cookie_from_har(path)

    assert result.found
    assert result.cookie == FULL_COOKIE
    assert result.source_url == "https://m.weibo.cn/api/config"
    assert result.candidates == 1


def test_prefers_mobile_and_most_complete_cookie(tmp_path):
    path = write_har(
        tmp_path,
        [
            _entry("https://weibo.com/ajax/side/hotSearch", PARTIAL_COOKIE),
            _entry("https://m.weibo.cn/api/container/getIndex", PARTIAL_COOKIE),
            _entry("https://m.weibo.cn/api/config", FULL_COOKIE),
        ],
    )
    result = extract_cookie_from_har(path)

    assert result.cookie == FULL_COOKIE, "应优先 m.weibo.cn 且字段最全的那份"
    assert result.candidates == 3


def test_prefers_mobile_over_pc_when_equally_complete(tmp_path):
    path = write_har(
        tmp_path,
        [
            _entry("https://weibo.com/ajax/statuses/x", FULL_COOKIE),
            _entry("https://m.weibo.cn/detail/123", FULL_COOKIE),
        ],
    )
    assert "m.weibo.cn" in extract_cookie_from_har(path).source_url


# 真实踩过的坑：只按「字段最多」挑，会选中 weibo.com 那份（看着最全），
# 但它缺 SSOLoginState / _T_WM，请求 m.weibo.cn 得到 login=false。
# 而同一份 HAR 里 api.weibo.cn（同为 .weibo.cn 域，Cookie 按域发送）那份更少字段却能登录。
PC_ONLY_COOKIE = "SUB=a; SUBP=b; ALF=c; WBPSESS=d; XSRF-TOKEN=e; ULV=f; SINAGLOBAL=g; _s_tentry=h; Apache=i; SCF=j"
CN_DOMAIN_COOKIE = "SUB=a; SUBP=b; SSOLoginState=1; ALF=c; SCF=d"


def test_prefers_weibo_cn_family_over_pc_even_with_fewer_fields(tmp_path):
    path = write_har(
        tmp_path,
        [
            _entry("https://weibo.com/ajax/getClientMessageSettings", PC_ONLY_COOKIE),
            _entry("https://api.weibo.cn/2/xxx", CN_DOMAIN_COOKIE),
        ],
    )
    result = extract_cookie_from_har(path)

    assert result.cookie == CN_DOMAIN_COOKIE, "采集走 m.weibo.cn，必须优先 .weibo.cn 域的 Cookie"
    assert result.source_url == "https://api.weibo.cn/2/xxx"
    assert result.candidates == 2


def test_api_weibo_cn_counts_as_the_mobile_cookie_domain(tmp_path):
    """``api.weibo.cn`` 与 ``m.weibo.cn`` 同属 .weibo.cn，不能被当成 PC 站。"""
    path = write_har(tmp_path, [_entry("https://api.weibo.cn/2/cardlist", CN_DOMAIN_COOKIE)])
    rendered = extract_cookie_from_har(path).render()

    assert "SSOLoginState" in rendered
    assert "缺失" not in rendered.split("常见字段已出现")[0]


def test_still_prefers_m_weibo_cn_over_other_weibo_cn_hosts(tmp_path):
    path = write_har(
        tmp_path,
        [
            _entry("https://api.weibo.cn/2/xxx", FULL_COOKIE),
            _entry("https://m.weibo.cn/api/config", PARTIAL_COOKIE),
        ],
    )
    assert "m.weibo.cn" in extract_cookie_from_har(path).source_url


def test_returns_empty_when_no_cookie_header(tmp_path):
    path = write_har(tmp_path, [_entry("https://m.weibo.cn/api/config", None)])
    result = extract_cookie_from_har(path)

    assert not result.found
    assert "没有在 HAR 里找到" in result.render()


def test_ignores_non_weibo_entries(tmp_path):
    path = write_har(tmp_path, [_entry("https://example.com/api", FULL_COOKIE)])
    assert not extract_cookie_from_har(path).found


def test_missing_or_broken_file_is_handled(tmp_path):
    assert not extract_cookie_from_har(tmp_path / "不存在.har").found

    broken = tmp_path / "broken.har"
    broken.write_text("{不是 JSON", encoding="utf-8")
    assert not extract_cookie_from_har(broken).found


def test_render_never_leaks_the_cookie_value(tmp_path):
    """输出只能给长度和字段名——Cookie 不能出现在日志或终端里。"""
    path = write_har(tmp_path, [_entry("https://m.weibo.cn/api/config", FULL_COOKIE)])
    rendered = extract_cookie_from_har(path).render()

    assert FULL_COOKIE not in rendered
    assert "SSOLoginState=123" not in rendered
    assert f"{len(FULL_COOKIE)} 字符" in rendered
    assert "SSOLoginState" in rendered, "字段名要显示，便于判断是否完整"


def test_write_cookie_replaces_existing_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=\nWEIBO_COOKIE=旧值\nLOG_LEVEL=INFO\n", encoding="utf-8")

    assert write_cookie_to_env(env, FULL_COOKIE) is True
    text = env.read_text(encoding="utf-8")
    assert f"WEIBO_COOKIE={FULL_COOKIE}" in text
    assert "旧值" not in text
    assert "LLM_API_KEY=" in text and "LOG_LEVEL=INFO" in text, "其它行必须原样保留"


def test_write_cookie_appends_when_absent(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LOG_LEVEL=INFO\n", encoding="utf-8")

    assert write_cookie_to_env(env, FULL_COOKIE) is False
    lines = env.read_text(encoding="utf-8").splitlines()
    assert f"WEIBO_COOKIE={FULL_COOKIE}" in lines


def test_write_cookie_creates_file_when_missing(tmp_path):
    env = tmp_path / "sub" / ".env"
    env.parent.mkdir(parents=True)
    assert write_cookie_to_env(env, FULL_COOKIE) is False
    assert f"WEIBO_COOKIE={FULL_COOKIE}" in env.read_text(encoding="utf-8")


def test_write_cookie_strips_newlines(tmp_path):
    """Cookie 里混进换行会把 .env 写坏。"""
    env = tmp_path / ".env"
    write_cookie_to_env(env, "SUB=a\nSUB2=b\r\n")
    assert env.read_text(encoding="utf-8").count("\n") == 2  # 注释行 + 值行
