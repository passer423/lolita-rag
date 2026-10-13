"""计划任务 XML 生成的测试。

这里能测的是「XML 是否表达了我们想要的语义」——真正的注册动作需要
系统权限，只能手工验证（脚本里有 --dry-run 可以只打印不执行）。
"""

from __future__ import annotations

from datetime import datetime, time
from xml.etree import ElementTree as ET

import pytest

from src.pipeline.task_scheduler import (
    TASK_NS,
    TaskSpec,
    TaskSpecError,
    build_task_xml,
    next_start_boundary,
    parse_start_time,
    schtasks_create_command,
    schtasks_delete_command,
    schtasks_query_command,
    schtasks_run_command,
    write_task_xml,
)

NS = {"t": TASK_NS}


def make_spec(**overrides) -> TaskSpec:
    # 用通用占位路径，避免把开发机的实际路径/用户名写进公开仓库
    base = dict(
        task_name="lolita-rag-daily-update",
        command=r"C:\Python312\python.exe",
        arguments=r'"C:\proj\lolita-rag\scripts\daily_update.py"',
        working_directory=r"C:\proj\lolita-rag",
        start_time="02:00",
        description="每日更新",
        user_id=r"DESKTOP\testuser",
    )
    base.update(overrides)
    return TaskSpec(**base)


def test_xml_expresses_the_intended_semantics():
    body = build_task_xml(make_spec(), datetime(2026, 10, 13, 2, 0, 0))
    root = ET.fromstring(body)

    assert root.tag == f"{{{TASK_NS}}}Task"
    assert root.get("version") == "1.2"

    assert root.find("t:Triggers/t:CalendarTrigger/t:StartBoundary", NS).text == "2026-10-13T02:00:00"
    assert root.find("t:Triggers/t:CalendarTrigger/t:ScheduleByDay/t:DaysInterval", NS).text == "1"

    settings = root.find("t:Settings", NS)
    assert settings.find("t:StartWhenAvailable", NS).text == "true"        # 错过就补跑
    assert settings.find("t:MultipleInstancesPolicy", NS).text == "IgnoreNew"  # 不重叠
    assert settings.find("t:ExecutionTimeLimit", NS).text == "PT1H"
    assert settings.find("t:DisallowStartIfOnBatteries", NS).text == "false"
    assert settings.find("t:Enabled", NS).text == "true"

    principal = root.find("t:Principals/t:Principal", NS)
    assert principal.get("id") == "Author"
    assert principal.find("t:UserId", NS).text == r"DESKTOP\testuser"
    assert principal.find("t:LogonType", NS).text == "InteractiveToken"
    assert principal.find("t:RunLevel", NS).text == "LeastPrivilege"

    exec_node = root.find("t:Actions/t:Exec", NS)
    assert exec_node.find("t:Command", NS).text == r"C:\Python312\python.exe"
    assert exec_node.find("t:Arguments", NS).text.endswith("daily_update.py\"")
    assert exec_node.find("t:WorkingDirectory", NS).text == r"C:\proj\lolita-rag"
    assert root.find("t:RegistrationInfo/t:Description", NS).text == "每日更新"


def test_parse_start_time_accepts_and_rejects():
    assert parse_start_time("02:00") == time(2, 0)
    assert parse_start_time(" 3:5 ") == time(3, 5)
    assert parse_start_time("23:59") == time(23, 59)
    for bad in ("", "2", "25:00", "aa:bb", "02:00:00", "-1:00"):
        with pytest.raises(TaskSpecError):
            parse_start_time(bad)


def test_next_start_boundary_rolls_forward():
    before = datetime(2026, 10, 13, 1, 30, 0)
    assert next_start_boundary("02:00", before) == datetime(2026, 10, 13, 2, 0, 0)

    after = datetime(2026, 10, 13, 3, 0, 0)
    assert next_start_boundary("02:00", after) == datetime(2026, 10, 14, 2, 0, 0)


def test_write_task_xml_uses_utf16_with_declaration(tmp_path):
    """schtasks /XML 对 UTF-16 + 声明 的兼容性最好，中文描述也不会乱码。"""
    path = write_task_xml(tmp_path / "task.xml", build_task_xml(make_spec()))

    raw = path.read_bytes()
    assert raw.startswith(b"\xff\xfe"), "应当是带 BOM 的 UTF-16LE"
    text = raw.decode("utf-16")
    assert text.startswith('<?xml version="1.0" encoding="UTF-16"?>')

    body = text.split("\n", 1)[1]
    assert ET.fromstring(body).tag == f"{{{TASK_NS}}}Task"
    assert "每日更新" in text


def test_schtasks_commands_are_well_formed():
    spec = make_spec()
    assert schtasks_create_command(spec, "t.xml") == [
        "schtasks", "/Create", "/TN", spec.task_name, "/XML", "t.xml", "/F",
    ]
    assert schtasks_delete_command(spec.task_name) == ["schtasks", "/Delete", "/TN", spec.task_name, "/F"]
    assert schtasks_run_command(spec.task_name) == ["schtasks", "/Run", "/TN", spec.task_name]
    assert "/V" in schtasks_query_command(spec.task_name)


def test_special_characters_in_paths_are_escaped():
    spec = make_spec(working_directory=r"D:\a & b\proj", arguments=r'"D:\a & b\proj\s.py"')
    body = build_task_xml(spec)
    ET.fromstring(body)  # 能被解析就说明转义正确
    assert "&amp;" in body
