"""生成 Windows 计划任务的定义（XML）。

为什么用「生成 XML + ``schtasks /Create /XML``」而不是 PowerShell 的
``New-ScheduledTask*`` 系列 cmdlet：

1. ``.ps1`` 在这台机器上有个真实的编码陷阱——本机只有 Windows PowerShell 5.1，
   它按 ANSI 解码**没有 BOM** 的 UTF-8 文件；中文会变乱码并吞掉引号，
   直接变成语法错误。而脚本文件在任何一次编辑后都可能丢掉 BOM（实测如此），
   所以「靠 BOM 保住中文 .ps1」是不可靠的方案。
2. 纯 ASCII 的 ``schtasks.exe`` 没有这个问题，且能通过 XML 精确设置
   ``StartWhenAvailable`` / ``MultipleInstancesPolicy`` 等 CLI 开关无法表达的项。

XML 的构造用 ``ElementTree`` 而不是拼字符串：路径里的 ``&``、引号等
由它负责转义，不用我自己想。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

TASK_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
DEFAULT_TASK_NAME = "lolita-rag-daily-update"

ET.register_namespace("", TASK_NS)


class TaskSpecError(ValueError):
    """参数不合法（时间格式等）。"""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_name: str
    command: str                # 通常是 python.exe 的绝对路径
    arguments: str              # 通常是 "…\\scripts\\daily_update.py"
    working_directory: str
    start_time: str = "02:00"   # 24 小时制 HH:MM
    description: str = ""
    user_id: str = ""
    execution_limit: str = "PT1H"


def parse_start_time(value: str) -> time:
    """解析 ``HH:MM``（兼容 ``H:MM``）。"""
    text = (value or "").strip()
    parts = text.split(":")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise TaskSpecError(f"时间格式应为 HH:MM（24 小时制），收到：{value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise TaskSpecError(f"时间超出范围：{value!r}")
    return time(hour=hour, minute=minute)


def next_start_boundary(start_time: str, now: datetime | None = None) -> datetime:
    """返回下一个该时刻的本地时间，作为 ``StartBoundary``。

    如果今天这个点还没到就用今天，否则用明天——计划任务要求
    ``StartBoundary`` 是一个具体时刻。
    """
    moment = parse_start_time(start_time)
    now = now or datetime.now()
    candidate = datetime.combine(now.date(), moment)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _sub(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    node = ET.SubElement(parent, f"{{{TASK_NS}}}{tag}")
    if text is not None:
        node.text = text
    return node


def build_task_xml(spec: TaskSpec, boundary: datetime | None = None) -> str:
    """生成任务 XML **正文**（不含 XML 声明）。

    不含声明是为了方便测试直接 ``ET.fromstring`` 解析；
    写文件时由 :func:`write_task_xml` 补上声明与 UTF-16 BOM。
    """
    boundary = boundary or next_start_boundary(spec.start_time)

    root = ET.Element(f"{{{TASK_NS}}}Task", {"version": "1.2"})

    info = _sub(root, "RegistrationInfo")
    if spec.description:
        _sub(info, "Description", spec.description)
    if spec.user_id:
        _sub(info, "Author", spec.user_id)

    triggers = _sub(root, "Triggers")
    calendar = _sub(triggers, "CalendarTrigger")
    _sub(calendar, "StartBoundary", boundary.strftime("%Y-%m-%dT%H:%M:%S"))
    _sub(calendar, "Enabled", "true")
    schedule = _sub(calendar, "ScheduleByDay")
    _sub(schedule, "DaysInterval", "1")

    principals = _sub(root, "Principals")
    principal = ET.SubElement(principals, f"{{{TASK_NS}}}Principal", {"id": "Author"})
    if spec.user_id:
        _sub(principal, "UserId", spec.user_id)
    _sub(principal, "LogonType", "InteractiveToken")
    _sub(principal, "RunLevel", "LeastPrivilege")

    settings = _sub(root, "Settings")
    # 只允许一个实例：与项目里的文件锁形成双保险
    _sub(settings, "MultipleInstancesPolicy", "IgnoreNew")
    _sub(settings, "DisallowStartIfOnBatteries", "false")
    _sub(settings, "StopIfGoingOnBatteries", "false")
    _sub(settings, "AllowHardTerminate", "true")
    # 错过的触发时间（休眠/关机）在恢复后尽快补跑
    _sub(settings, "StartWhenAvailable", "true")
    _sub(settings, "RunOnlyIfNetworkAvailable", "false")
    idle = _sub(settings, "IdleSettings")
    _sub(idle, "StopOnIdleEnd", "false")
    _sub(idle, "RestartOnIdle", "false")
    _sub(settings, "AllowStartOnDemand", "true")
    _sub(settings, "Enabled", "true")
    _sub(settings, "Hidden", "false")
    _sub(settings, "RunOnlyIfIdle", "false")
    _sub(settings, "WakeToRun", "false")
    _sub(settings, "ExecutionTimeLimit", spec.execution_limit)
    _sub(settings, "Priority", "7")

    actions = ET.SubElement(root, f"{{{TASK_NS}}}Actions", {"Context": "Author"})
    exec_node = _sub(actions, "Exec")
    _sub(exec_node, "Command", spec.command)
    _sub(exec_node, "Arguments", spec.arguments)
    _sub(exec_node, "WorkingDirectory", spec.working_directory)

    return ET.tostring(root, encoding="unicode")


def write_task_xml(path: Path | str, body: str) -> Path:
    """以 UTF-16（带 BOM）+ XML 声明写盘：这是 ``schtasks /XML`` 最稳的格式。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = '<?xml version="1.0" encoding="UTF-16"?>\n' + body
    path.write_bytes(document.encode("utf-16"))
    return path


def schtasks_create_command(spec: TaskSpec, xml_path: Path | str) -> list[str]:
    return ["schtasks", "/Create", "/TN", spec.task_name, "/XML", str(xml_path), "/F"]


def schtasks_delete_command(task_name: str) -> list[str]:
    return ["schtasks", "/Delete", "/TN", task_name, "/F"]


def schtasks_run_command(task_name: str) -> list[str]:
    return ["schtasks", "/Run", "/TN", task_name]


def schtasks_query_command(task_name: str) -> list[str]:
    # /V 详细、/FO LIST 便于人读；刻意不解析输出——中文系统的字段名是本地化的，
    # 解析会很脆。真正的「上次结果」看 data/logs/history.jsonl 更可靠。
    return ["schtasks", "/Query", "/TN", task_name, "/V", "/FO", "LIST"]
