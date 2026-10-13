"""登录态预检测试。

这是「失效即停止」策略的判定点：判错的两个方向都有代价——
把有效 Cookie 判成失效会让你白忙，把失效 Cookie 判成有效则会浪费采集配额
并可能触发风控，所以两个方向都要测。
"""

from __future__ import annotations

from src.config import Settings
from src.crawler.weibo_client import audit_cookie, check_login, cookie_names


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload=None, exc: Exception | None = None):
        self.payload = payload
        self.exc = exc
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.exc:
            raise self.exc
        return FakeResponse(self.payload)


def test_valid_login_is_parsed():
    session = FakeSession({"ok": 1, "data": {"login": True, "uid": "12345", "screen_name": "裙摆与猫"}})
    status = check_login(Settings(weibo_cookie="SUB=x"), session=session)

    assert status.logged_in is True
    assert status.uid == "12345"
    assert status.screen_name == "裙摆与猫"
    assert "裙摆与猫" in status.render()


def test_ok_but_not_logged_in_is_detected():
    """实测：失效 Cookie 会返回 ok=1 且 login=false —— 只看 ok 会误判成有效。"""
    session = FakeSession({"ok": 1, "data": {"login": False}})
    status = check_login(Settings(weibo_cookie="SUB=x"), session=session)

    assert status.ok is True
    assert status.logged_in is False
    assert "登录态无效" in status.render()
    # 缺字段信息由 missing / cookie_hint() 单独承载，不再塞进 detail
    assert "SSOLoginState" in status.missing


def test_missing_cookie_short_circuits_without_any_request():
    session = FakeSession({"ok": 1, "data": {"login": True}})
    status = check_login(Settings(weibo_cookie=""), session=session)

    assert status.logged_in is False
    assert session.calls == [], "没有 Cookie 时不应该发请求"
    assert "WEIBO_COOKIE" in status.detail


def test_network_error_is_reported_not_raised():
    session = FakeSession(exc=RuntimeError("连接超时"))
    status = check_login(Settings(weibo_cookie="SUB=x"), session=session)

    assert status.ok is False
    assert status.logged_in is False
    assert "连接超时" in status.detail


def test_precheck_costs_exactly_one_request():
    session = FakeSession({"ok": 1, "data": {"login": True}})
    check_login(Settings(weibo_cookie="SUB=x"), session=session)
    assert len(session.calls) == 1


def test_audit_cookie_reports_present_and_missing_names():
    present, missing = audit_cookie("SUB=x; SUBP=y; ALF=z; 未知=1")
    assert present == ["ALF", "SUB", "SUBP"]
    assert "SSOLoginState" in missing and "_T_WM" in missing
    assert "未知" not in present, "只报告预期字段"


def test_cookie_names_handles_values_containing_equals_signs():
    """Cookie 值里常出现 = （例如 SUBP 的 base64 补位），不能按第一个 = 截断错名字。"""
    names = cookie_names("SUBP=a=b=c; SUB=x")
    assert names == {"SUBP", "SUB"}


def test_login_status_hint_points_at_missing_fields():
    """手工复制 Cookie 最常见的失败原因是漏字段，检查结果必须直接指出来。"""
    session = FakeSession({"ok": 1, "data": {"login": False}})
    status = check_login(Settings(weibo_cookie="SUB=x; SUBP=y"), session=session)

    hint = status.cookie_hint()
    assert "SSOLoginState" in hint and "_T_WM" in hint
    assert "HAR" in hint, "要给出可行的替代做法"


def test_no_hint_when_nothing_missing():
    cookie = "SUB=x; SUBP=y; SSOLoginState=z; ALF=a; _T_WM=b; WBPSESS=c; XSRF-TOKEN=d"
    session = FakeSession({"ok": 1, "data": {"login": True}})
    status = check_login(Settings(weibo_cookie=cookie), session=session)

    assert status.cookie_hint() == ""
