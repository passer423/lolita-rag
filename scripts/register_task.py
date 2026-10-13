"""注册 / 查看 / 注销「每日凌晨更新」的 Windows 计划任务。

用法：
    python scripts/register_task.py --dry-run         # 只打印将要做的事，不改系统
    python scripts/register_task.py                   # 注册，每天 02:00
    python scripts/register_task.py --time 03:30
    python scripts/register_task.py --status
    python scripts/register_task.py --run-now         # 立刻触发一次
    python scripts/register_task.py --unregister

为什么用 Python 而不是 PowerShell 脚本：本机只有 Windows PowerShell 5.1，
它按 ANSI 解码**没有 BOM** 的 UTF-8 文件——中文会变成乱码并吞掉引号，
直接导致语法错误；而脚本文件在任何一次编辑后都可能丢掉 BOM（实测如此）。
改用 Python + ``schtasks.exe`` 没有这个陷阱，中文提示也能正常显示。

⚠️ 运行条件：任务在你自己的账户下运行，**需要该用户处于登录状态**
（锁屏可以，注销/关机不行）。要做到「注销后也能跑」，需要用带密码的凭据
（``schtasks /RU /RP``），该步骤需要管理员权限。

关于自动登录：本脚本只负责定时触发；登录始终由人在浏览器里完成，
项目不实现任何验证码绕过或用账号密码自动登录。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import PROJECT_ROOT, settings
from src.pipeline.task_scheduler import (
    DEFAULT_TASK_NAME,
    TaskSpec,
    TaskSpecError,
    build_task_xml,
    next_start_boundary,
    schtasks_create_command,
    schtasks_delete_command,
    schtasks_query_command,
    schtasks_run_command,
    write_task_xml,
)
from src.storage import read_jsonl

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NOT_FOUND = 3
EXIT_REGISTER_FAILED = 5


def _run(cmd: list[str]) -> int:
    """执行外部命令。

    刻意**不捕获输出**（继承 stdio）：在受限沙箱里管道会失败，
    而且 schtasks 的提示文本本来就该原样呈现给用户。
    """
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _print_history(limit: int = 3) -> None:
    rows = read_jsonl(settings.history_path)
    if not rows:
        print(f"（暂无运行记录：{settings.history_path}）")
        return
    print(f"最近 {min(limit, len(rows))} 次运行（来自 {settings.history_path}）：")
    for row in rows[-limit:]:
        print(
            f"  {row.get('started_at')}  抓取 {row.get('fetched')} 条 / 新增 {row.get('new_posts')} 条"
            f" / 编码 {row.get('embedded')} 条 / 耗时 {row.get('seconds')} 秒"
        )


def _latest_log() -> None:
    logs = sorted(settings.logs_dir.glob("update-*.log"))
    if logs:
        print(f"最近日志文件：{logs[-1]}")


def build_spec(args: argparse.Namespace) -> TaskSpec:
    daily = PROJECT_ROOT / "scripts" / "daily_update.py"
    if not daily.exists():
        raise FileNotFoundError(f"找不到入口脚本：{daily}")
    user_id = f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}".strip("\\")
    return TaskSpec(
        task_name=args.task_name,
        command=args.python or sys.executable,
        arguments=f'"{daily}"',
        working_directory=str(PROJECT_ROOT),
        start_time=args.time,
        description=(
            "lolita-rag 每日更新：采集微博后增量重建索引；"
            "Cookie 失效时以退出码 3 停止（退出码：0=成功 2=错误 3=Cookie 失效 4=已有实例在运行）"
        ),
        user_id=user_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="管理每日更新的 Windows 计划任务")
    parser.add_argument("--time", default="02:00", help="每天触发时间，24 小时制 HH:MM（默认 02:00）")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--python", default="", help="python.exe 绝对路径（默认用当前解释器）")
    parser.add_argument("--status", action="store_true", help="查看任务状态与最近运行记录")
    parser.add_argument("--run-now", action="store_true", help="立刻触发一次")
    parser.add_argument("--unregister", action="store_true", help="注销任务")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做的事，不执行任何命令")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)
    settings.ensure_dirs()

    try:
        spec = build_spec(args)
    except (FileNotFoundError, TaskSpecError) as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        boundary = next_start_boundary(spec.start_time)
    except TaskSpecError as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.status:
        code = _run(schtasks_query_command(spec.task_name))
        print()
        _print_history()
        _latest_log()
        if code != 0:
            print(f"（schtasks 退出码 {code}：任务可能尚未注册）")
            return EXIT_NOT_FOUND
        return EXIT_OK

    if args.unregister:
        if args.dry_run:
            print("$ " + " ".join(schtasks_delete_command(spec.task_name)))
            return EXIT_OK
        code = _run(schtasks_delete_command(spec.task_name))
        return EXIT_OK if code == 0 else EXIT_NOT_FOUND

    if args.run_now:
        if args.dry_run:
            print("$ " + " ".join(schtasks_run_command(spec.task_name)))
            return EXIT_OK
        code = _run(schtasks_run_command(spec.task_name))
        if code == 0:
            print("已触发。稍后可用 --status 查看结果；任务内部结果码含义：0=成功 2=错误 3=Cookie 失效 4=已有实例在运行")
        return EXIT_OK if code == 0 else EXIT_ERROR

    # ---- 注册
    body = build_task_xml(spec, boundary)
    xml_path = settings.logs_dir / f"task-{spec.task_name}.xml"

    print("将注册计划任务：")
    print(f"  任务名     : {spec.task_name}")
    print(f"  触发时间   : 每天 {spec.start_time}（首次 {boundary:%Y-%m-%d %H:%M}）")
    print(f"  Python     : {spec.command}")
    print(f"  入口脚本   : {spec.arguments}")
    print(f"  工作目录   : {spec.working_directory}")
    print(f"  运行账户   : {spec.user_id or '(当前用户)'}（需要处于登录状态；锁屏可以，注销不行）")
    print("  关键设置   : 错过后尽快补跑 / 忽略重叠实例 / 超时 1 小时")
    print()

    if args.dry_run:
        print("--- 任务 XML（将写入 %s）---" % xml_path)
        print(body)
        print("--- 将要执行的命令 ---")
        print("$ " + " ".join(schtasks_create_command(spec, xml_path)))
        print()
        print("（--dry-run：以上均未执行）")
        return EXIT_OK

    write_task_xml(xml_path, body)
    print(f"任务定义已写入：{xml_path}")
    code = _run(schtasks_create_command(spec, xml_path))
    if code != 0:
        print(
            "\n[注册失败] 如果提示「拒绝访问」，请改用**管理员身份**的终端重跑本命令；"
            "也可能需要在普通终端（非受限沙箱）中执行。",
            file=sys.stderr,
        )
        return EXIT_REGISTER_FAILED

    print("\n注册成功。常用命令：")
    print("  python scripts/register_task.py --status")
    print("  python scripts/register_task.py --run-now")
    print("  python scripts/register_task.py --unregister")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
