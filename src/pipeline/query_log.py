"""查询日志：把每一次**真实**检索落成一行 jsonl。

## 为什么必须有这个模块

1. **评测集是我自己出的题。** `eval/qa_real.jsonl` 是「读完语料手写」的，拿它做优化
   等于自己出题自己答题——指标涨了也可能只是拟合了那几十道题。线上真实查询是唯一
   不受这层影响的证据来源。
2. **没有日志就只能编查询找问题。** 上一轮优化我是坐下来手写 12 条"用户会问什么"，
   覆盖面完全取决于我的想象。真实日志才能告诉你用户到底在搜什么、什么搜不到。
3. **参数一变、语料一换，历史结果就不可比了。** 所以每行都带**参数指纹**
   （片段数、embedding 模型、阈值、融合方式），回头分析时能分段看。

## 两条硬约束

* **绝不因为写日志而让检索失败**：任何写入异常都吞掉并打 warning。
* **评测与排查脚本不进日志**：`pipeline/eval.py` 与 `scripts/explain_retrieval.py`
  都直接调 `retriever.search`，天然不经过这里；`ask()` / `retrieve()` 才写日志，
  并由调用方声明 ``source``（``web`` / ``cli``）。否则自己的评测会把「拒答率」
  这类统计彻底污染。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config import Settings, settings as default_settings
from src.schema import Retrieved

logger = logging.getLogger(__name__)

# 追加写用一把进程内锁：uvicorn 的同步端点跑在线程池里，可能有并发写。
_WRITE_LOCK = threading.Lock()


@dataclass(slots=True)
class QueryRecord:
    """一行查询日志。字段刻意保持扁平，方便直接丢给 pandas/jq。"""

    ts: str
    source: str
    query: str
    top_k: int
    hits: int
    refused: bool
    elapsed_ms: float
    terms: list[str] = field(default_factory=list)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    top: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def settings_fingerprint(settings: Settings, bundle: Any | None = None) -> dict[str, Any]:
    """把「影响检索结果的参数 + 语料规模」压成一个指纹。

    两个不同指纹下的日志混在一起看是没有意义的：语料从 615 涨到 1372 片段、
    或者阈值从 0.45 调到 0.44，同样的查询结果都会变。
    """
    chunks = getattr(bundle, "chunks", None)
    embedder = getattr(bundle, "embedder", None)
    return {
        "chunks": len(chunks) if chunks is not None else 0,
        "embedder": getattr(embedder, "name", "") or "",
        "fusion_mode": settings.fusion_mode,
        "min_vector_score": settings.min_vector_score,
        "min_bm25_ratio": settings.min_bm25_ratio,
        "distinctive_df_ratio": settings.distinctive_df_ratio,
        "recall_k": settings.recall_k,
        # 消融验证与闸门实现都跟着这两个值变，必须进指纹：
        # 少了它，打开精排会**静默**改变指标，回归却以为还是同一套配置。
        "reranker_enabled": settings.reranker_enabled,
        "semantic_support_k": settings.semantic_support_k,
        "absent_majority_ratio": settings.absent_majority_ratio,
    }


def build_record(
    query: str,
    hits: Sequence[Retrieved],
    *,
    settings: Settings | None = None,
    source: str = "cli",
    elapsed_ms: float = 0.0,
    terms: Sequence[str] | None = None,
    bundle: Any | None = None,
) -> QueryRecord:
    """把一次检索结果整理成待写盘的记录（纯函数，便于测试）。"""
    settings = settings or default_settings
    if terms is None:
        from src.retriever.bm25 import tokenize

        terms = tokenize(query)
    top = [
        {
            "chunk_id": hit.chunk.chunk_id,
            "post_id": hit.chunk.post_id,
            "author": hit.chunk.author,
            "score": round(float(hit.score), 6),
            "route": hit.source,
        }
        for hit in list(hits)[: max(0, settings.query_log_top_hits)]
    ]
    return QueryRecord(
        ts=datetime.now().isoformat(timespec="seconds"),
        source=source,
        query=query,
        top_k=settings.top_k,
        hits=len(hits),
        # 一条都没召回 = 系统按设计拒答。这是 bad case 分析里最值得看的一类。
        refused=not hits,
        elapsed_ms=round(float(elapsed_ms), 2),
        terms=list(terms),
        fingerprint=settings_fingerprint(settings, bundle),
        top=top,
    )


def write_record(record: QueryRecord, settings: Settings | None = None) -> bool:
    """追加一行。**返回是否写成，绝不抛异常**——写日志不该把检索搞挂。"""
    settings = settings or default_settings
    try:
        path = settings.query_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with _WRITE_LOCK:
            # newline="\n"：Windows 上默认会把 \n 翻成 CRLF。日志本身在 data/ 里不入库，
            # 但统一 LF 能让 grep/jq/pandas 在任何平台行为一致。
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 - 见上面那条硬约束
        logger.warning("查询日志写入失败（已忽略，不影响检索）：%s", exc)
        return False


def log_query(
    query: str,
    hits: Sequence[Retrieved],
    *,
    settings: Settings | None = None,
    source: str = "cli",
    elapsed_ms: float = 0.0,
    terms: Sequence[str] | None = None,
    bundle: Any | None = None,
) -> QueryRecord | None:
    """记录一次真实检索。关闭开关时返回 None，不写盘。"""
    settings = settings or default_settings
    if not settings.query_log_enabled:
        return None
    record = build_record(
        query,
        hits,
        settings=settings,
        source=source,
        elapsed_ms=elapsed_ms,
        terms=terms,
        bundle=bundle,
    )
    write_record(record, settings)
    return record


def read_records(path: Path | str, limit: int | None = None) -> list[dict[str, Any]]:
    """读日志。**坏行直接跳过**：日志是追加写的，进程被杀可能留下半行。"""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("跳过损坏的日志行：%s", line[:80])
            continue
        if isinstance(payload, dict):
            records.append(payload)
    if limit is not None and limit > 0:
        records = records[-limit:]
    return records


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(float(ordered[index]), 2)


def summarize(records: Sequence[dict[str, Any]], *, source: str | None = None) -> dict[str, Any]:
    """汇总：拒答率、延迟分位、高频查询、拒答查询清单。

    ``source`` 可只统计某一路（例如只看 ``web``），把 CLI 试玩与真实使用分开。
    """
    rows = [row for row in records if source is None or row.get("source") == source]
    total = len(rows)
    refused = [row for row in rows if row.get("refused")]
    latencies = [float(row.get("elapsed_ms") or 0.0) for row in rows]

    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("query") or "").strip()
        if key:
            counts[key] = counts.get(key, 0) + 1

    return {
        "total": total,
        "refused": len(refused),
        "refusal_rate": round(len(refused) / total, 4) if total else 0.0,
        "avg_hits": round(sum(int(row.get("hits") or 0) for row in rows) / total, 2) if total else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "by_source": {
            name: sum(1 for row in rows if row.get("source") == name)
            for name in sorted({str(row.get("source")) for row in rows})
        },
        "popular": sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20],
        "refused_queries": sorted({str(row.get("query") or "") for row in refused if row.get("query")}),
    }
