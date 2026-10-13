"""每日定时更新的编排逻辑。

顺序刻意如此，每一步都有理由：

1. **单实例锁**（在脚本层持锁）——避免上一次没跑完就被再次触发，请求频率翻倍；
2. **登录态预检**——只花 1 个请求；未登录立刻抛错退出，**不发任何采集请求**，
   这是我们选的「失效即停止」策略的核心；
3. **daily 采集**——每晚重扫前 N 页，靠博文 id 去重（新内容在搜索结果顶部）；
4. **增量建索引**——只有真正的新博文会被切分与编码；
5. **写历史流水**——一行一次运行，便于看「每天新增多少条」的趋势。

注意第 4 步的短路：如果本次没有任何新博文，且索引已存在，就直接跳过建索引。
否则每晚都要白加载一次 BGE 模型（约 5 秒）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

import requests

from src.config import Settings, settings as default_settings
from src.crawler.weibo_client import LoginStatus, WeiboCrawlError, check_login
from src.pipeline.build_index import BuildStats, build_index
from src.pipeline.crawl import MODE_DAILY, crawl
from src.storage import append_jsonl, load_chunks, load_posts

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NOT_LOGGED_IN = 3
EXIT_ALREADY_RUNNING = 4


class NotLoggedInError(RuntimeError):
    """Cookie 不是有效登录会话。此时不发出任何采集请求。"""

    def __init__(self, status: LoginStatus):
        super().__init__(status.render())
        self.status = status


@dataclass(slots=True)
class DailyReport:
    started_at: str = ""
    finished_at: str = ""
    account: str = ""
    keywords: list[str] = field(default_factory=list)
    pages: int = 0
    fetched: int = 0            # 本次抓到的博文数（含与库里重复的）
    new_posts: int = 0          # 语料实际新增条数
    chunks_added: int = 0
    embedded: int = 0
    total_posts: int = 0
    total_chunks: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"运行时间      : {self.started_at} → {self.finished_at}（耗时 {self.seconds:.1f} 秒）",
            f"登录账号      : {self.account or '未知'}",
            f"关键词 / 页数 : {', '.join(self.keywords)} / {self.pages} 页",
            f"本次抓取      : {self.fetched} 条（其中新增入库 {self.new_posts} 条）",
            f"索引更新      : 新增/变更 {self.chunks_added} 个片段，编码 {self.embedded} 条",
            f"语料规模      : {self.total_posts} 条博文 / {self.total_chunks} 个片段",
        ]
        if self.notes:
            lines.append("说明：")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def run_daily_update(
    keywords: list[str] | None = None,
    settings: Settings | None = None,
    *,
    pages: int | None = None,
    client: Any | None = None,
    login_checker: Callable[[], LoginStatus] | None = None,
    notifier: Callable[[Settings], str] | None = None,
) -> DailyReport:
    """执行一次每日更新。**调用方负责持锁**（见 ``scripts/daily_update.py``）。

    ``client`` / ``login_checker`` / ``notifier`` 都可注入，因此整条链路可以离线测试。

    ``notifier`` 刻意**默认不做事**：库函数不该暗地里发网络请求（否则单元测试会去打
    本机的服务端口）。真实通知由入口脚本接上 —— 见 ``scripts/daily_update.py``。
    """
    settings = settings or default_settings
    settings.ensure_dirs()
    started = time.monotonic()
    report = DailyReport(started_at=datetime.now().isoformat(timespec="seconds"))

    report.keywords = list(keywords) if keywords else settings.daily_keyword_list
    if not report.keywords:
        raise ValueError("没有每日关键词：请配置 DAILY_KEYWORDS，或用 --keyword 指定")
    report.pages = pages or settings.daily_pages

    # ---- 1) 登录态预检：未登录直接退出，不浪费采集请求
    checker = login_checker or (lambda: check_login(settings))
    status = checker()
    if not status.logged_in:
        # 先落日志再抛错：这个任务在凌晨无人值守运行，
        # 失败原因必须出现在日志文件里，而不是只在终端的 stderr 上
        logger.error("登录态无效，本次不采集：%s", status.detail)
        raise NotLoggedInError(status)
    report.account = status.screen_name or status.uid or "未知"
    logger.info("登录态检查通过：%s", status.render())

    # ---- 2) daily 采集
    before = len(load_posts(settings.raw_posts_path))
    posts = crawl(
        report.keywords,
        settings,
        pages=report.pages,
        mode=MODE_DAILY,
        client=client,
    )
    after = len(load_posts(settings.raw_posts_path))
    report.fetched = len(posts)
    report.new_posts = max(0, after - before)

    # ---- 3) 增量建索引；无新增且索引已存在时短路，避免白加载模型
    index_ready = settings.manifest_path.exists() and settings.chunks_path.exists()
    if report.new_posts == 0 and index_ready:
        report.notes.append("本次没有新增博文，跳过索引更新")
        report.total_chunks = len(load_chunks(settings.chunks_path))
        logger.info("没有新增博文，跳过索引更新")
    else:
        stats: BuildStats = build_index(source="raw", settings=settings)
        report.chunks_added = stats.added
        report.embedded = stats.embedded
        report.total_chunks = stats.chunks
        report.notes.extend(stats.notes)
        if stats.mode == "rebuild":
            report.notes.append("本次为全量重建")
    report.total_posts = after

    report.finished_at = datetime.now().isoformat(timespec="seconds")
    report.seconds = time.monotonic() - started
    if notifier is not None:
        report.notes.append(notifier(settings))
    append_jsonl(settings.history_path, report.to_dict())
    logger.info("每日更新完成：%s", report.to_dict())
    return report


def notify_server_reload(settings: Settings | None = None) -> str:
    """跑完顺手通知本地服务重载索引；服务没在跑就当没这回事。

    为什么要这一步：`serve.py` 是启动时装载一次索引，而本任务在**另一个进程**里写新索引
    —— 不通知的话，服务要么等自己的后台轮询（默认 30 秒），要么一直用旧索引。

    **刻意做成 fire-and-forget**：服务没开、端口不通、返回 500，都不能让每日更新算失败。
    索引已经在磁盘上写好了，那才是这条流水线的产出。
    """
    settings = settings or default_settings
    url = f"http://127.0.0.1:{settings.web_port}/api/reload"
    try:
        response = requests.post(url, timeout=3)
    except requests.RequestException as exc:
        logger.info("本地服务未运行或不可达，跳过通知重载：%s", exc)
        return "本地服务未运行，未通知重载（下次启动服务时会装载新索引）"
    if response.status_code != 200:
        logger.warning("通知重载被拒绝：HTTP %s %s", response.status_code, response.text[:120])
        return f"通知服务重载失败：HTTP {response.status_code}"
    logger.info("已通知本地服务重载索引：%s", response.text[:120])
    return "已通知本地服务重载索引"


__all__ = [
    "EXIT_ALREADY_RUNNING",
    "EXIT_ERROR",
    "EXIT_NOT_LOGGED_IN",
    "EXIT_OK",
    "DailyReport",
    "NotLoggedInError",
    "WeiboCrawlError",
    "run_daily_update",
]
