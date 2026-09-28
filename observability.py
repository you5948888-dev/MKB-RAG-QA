"""
全链路可观测性（Advanced RAG 迭代 · 第 4 期）。

覆盖两条链路：
  - **问答链路**：接收问题 → 问题改写 → 多查询/HyDE → 查询向量化 → 多路召回
    → 结果融合 → Rerank → 父块回查与上下文组装 → 大模型生成 → 流式输出完成；
  - **文档链路**：文件保存 → 文档解析 → 父/子块切分 → Embedding → 向量写入
    → 数据保存 → 索引完成。

设计原则（对应需求"降级与异常要求"）：
  1. **观测绝不阻断主流程**：所有落库/聚合都包在 try/except 里，失败只记告警；
  2. **敏感信息不落库**：写入前统一走 `sanitize()`，掩码 API Key / 签名密钥 / Redis 密码；
  3. **不自动清理**：数据一直保留，由用户通过 `POST /api/observability/traces/cleanup`
     手动清理（支持 dry-run 先看影响面）。

trace_id 生成规则：HTTP 问答用新 uuid；文档/索引任务直接用 task_id，天然把
"任务表"和"链路追踪"打通，无需额外传参。
"""
import json
import time
import uuid
import logging
import re
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import database

logger = logging.getLogger("rag.observability")

# ── 阶段名称（统一常量，前后端展示一致）──────────────────

STAGE_RECEIVE = "接收问题"
STAGE_REWRITE = "问题改写"
STAGE_EXPAND = "多查询/HyDE"
STAGE_EMBED = "查询向量化"
STAGE_RETRIEVE = "多路召回"
STAGE_FUSE = "结果融合"
STAGE_RERANK = "Rerank"
STAGE_CONTEXT = "父块回查与上下文组装"
STAGE_GENERATE = "大模型生成"
STAGE_STREAM_DONE = "流式输出完成"

# 文档链路
STAGE_FILE_SAVE = "文件保存"
STAGE_PARSE = "文档解析"
STAGE_SPLIT = "父/子块切分"
STAGE_EMBED_DOC = "Embedding"
STAGE_VECTOR_WRITE = "向量写入"
STAGE_CHUNK_SAVE = "数据保存"
STAGE_INDEX_DONE = "索引完成"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded"
STATUS_RUNNING = "running"

# 单条 trace 详情里每个内容字段的截断长度（避免观测库无限膨胀）
DETAIL_TEXT_LIMIT = 300

# ── 敏感信息脱敏 ────────────────────────────────────────

_SENSITIVE_PATTERNS = (
    (re.compile(r"(?i)(x-api-key\"?\s*[:=]\s*\"?)[^\"\s,}]+"), r"\1***"),
    (re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)[^\"\s,}]+"), r"\1***"),
    (re.compile(r"(?i)(authorization\"?\s*[:=]\s*\"?)[^\",\s}]+"), r"\1***"),
    (re.compile(r"(?i)(auth_secret_key\"?\s*[:=]\s*\"?)[^\"\s,}]+"), r"\1***"),
    (re.compile(r"(?i)(password\"?\s*[:=]\s*\"?)[^\"\s,}]+"), r"\1***"),
    (re.compile(r"(redis://[^:@/\s]*:)[^@\s]+(@)"), r"\1***\2"),
    (re.compile(r"(?i)(rag_session=)[^;\s]+"), r"\1***"),
    (re.compile(r"\bak_[0-9a-fA-F]{8,}"), "ak_***"),
    (re.compile(r"\bsk-[A-Za-z0-9]{8,}"), "sk-***"),
)


def mask_secrets(text: str) -> str:
    """把文本里的密钥类信息替换成掩码（密码 / API Key / 签名密钥 / Cookie）。"""
    if not text:
        return ""
    masked = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def sanitize(value):
    """递归脱敏：dict / list / str / 其它类型统一处理后再落库。"""
    if value is None:
        return None
    if isinstance(value, str):
        return mask_secrets(value)[:4000]
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (int, float, bool)):
        return value
    return mask_secrets(str(value))


def _truncate(value, limit: int = DETAIL_TEXT_LIMIT):
    """详情里的长文本统一截断，控制单行体积。"""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"…(已截断，共 {len(value)} 字)"
    return value


# ── 记录器 ──────────────────────────────────────────────

@dataclass
class SpanHandle:
    """一次阶段执行的句柄（用 with 语法时会自动记录耗时与状态）。"""

    stage: str
    input_count: int | None = None
    output_count: int | None = None
    status: str = STATUS_OK
    error: str = ""
    detail: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def set_output(self, count: int | None = None, **detail) -> None:
        if count is not None:
            self.output_count = count
        if detail:
            self.detail.update(detail)


class TraceRecorder:
    """一次业务请求的追踪记录器：先落 trace 行，结束时批量写 span。"""

    def __init__(self, trace_id: str, biz_type: str, *, kb_id: str | None = None,
                 session_id: str | None = None, question: str | None = None,
                 config_snapshot: dict | None = None):
        self.trace_id = trace_id
        self.biz_type = biz_type
        self.kb_id = kb_id
        self.session_id = session_id
        self.question = question
        self.rewritten_question = ""
        self.expansions: list[str] = []
        self.hyde_answer = ""
        self.config_snapshot = config_snapshot or {}
        self.recall_empty = False
        self.first_token_ms: int | None = None
        self.total_ms: int | None = None
        self.status = STATUS_RUNNING
        self.error = ""
        self.detail: dict = {}
        self._spans: list[dict] = []
        self._seq = 0
        self._started = time.time()

    # -- 生命周期 --

    @classmethod
    def start(cls, biz_type: str, **fields) -> "TraceRecorder":
        recorder = cls(str(uuid.uuid4()), biz_type, **fields)
        recorder._persist_trace(insert=True)
        return recorder

    @classmethod
    def for_task(cls, task_id: str, biz_type: str = "upload", **fields) -> "TraceRecorder":
        """文档链路：直接用 task_id 作为 trace_id，任务表与链路天然对齐。"""
        recorder = cls(task_id, biz_type, **fields)
        recorder._persist_trace(insert=True)
        return recorder

    def elapsed_ms(self) -> int:
        return int((time.time() - self._started) * 1000)

    # -- 记录 --

    @contextmanager
    def span(self, stage: str, input_count: int | None = None, **detail):
        handle = SpanHandle(stage=stage, input_count=input_count, detail=dict(detail))
        try:
            yield handle
        except Exception as exc:
            handle.status = STATUS_FAILED
            handle.error = f"{type(exc).__name__}: {exc}"
            self.record(handle)
            raise
        self.record(handle)

    def record(self, handle: SpanHandle, ended_at: float | None = None) -> None:
        end = ended_at if ended_at is not None else time.time()
        self._seq += 1
        self._spans.append({
            "seq": self._seq,
            "stage": handle.stage,
            "started_at": handle.started_at,
            "duration_ms": int((end - handle.started_at) * 1000),
            "status": handle.status,
            "input_count": handle.input_count,
            "output_count": handle.output_count,
            "error": mask_secrets(handle.error)[:500],
            "detail": sanitize({k: _truncate(v) for k, v in handle.detail.items()}),
        })

    def mark(self, stage: str, *, duration_ms: int = 0, status: str = STATUS_OK,
             input_count: int | None = None, output_count: int | None = None,
             error: str = "", **detail) -> None:
        """直接记一条阶段（用于已经有自己计时逻辑的代码，如任务链路）。"""
        now = time.time()
        self.record(SpanHandle(stage=stage, input_count=input_count, output_count=output_count,
                               status=status, error=error, detail=dict(detail),
                               started_at=now - duration_ms / 1000.0))

    def degrade(self, stage: str, reason: str, fallback: str = "") -> None:
        """记录一次降级（需求：每一次降级都应在执行详情中可见）。"""
        self.mark(stage, status=STATUS_DEGRADED, error=reason, fallback=fallback)
        self.add_degradation(stage, reason, fallback)

    def add_degradation(self, stage: str, reason: str, fallback: str = "") -> None:
        """只登记一条「降级记录」，**不**额外产生阶段行。

        用于已经在 `with observability.span(...)` 里手写 `span.status = degraded` 的场景
        （如 Rerank 失败）：此时再调 degrade() 会让同一阶段在阶段表里出现两条（一条 failed、
        一条 degraded），所以只用它补「降级记录」。
        """
        self.detail.setdefault("degradations", []).append(
            {"stage": stage, "reason": mask_secrets(reason)[:300], "fallback": fallback}
        )

    def set_result(self, *, rewritten_question: str | None = None, expansions=None,
                   hyde_answer: str | None = None, recall_empty: bool | None = None,
                   config_snapshot: dict | None = None, **detail) -> None:
        if rewritten_question is not None:
            self.rewritten_question = mask_secrets(rewritten_question)[:500]
        if expansions is not None:
            self.expansions = [mask_secrets(item)[:200] for item in expansions][:10]
        if hyde_answer is not None:
            self.hyde_answer = mask_secrets(hyde_answer)[:2000]
        if recall_empty is not None:
            self.recall_empty = bool(recall_empty)
        if config_snapshot is not None:
            self.config_snapshot = sanitize(config_snapshot)
        if detail:
            self.detail.update(sanitize({k: _truncate(v) for k, v in detail.items()}))

    def finish(self, status: str = STATUS_OK, error: str = "", first_token_ms: int | None = None,
               total_ms: int | None = None) -> str:
        """结束追踪：批量写入 span 并更新 trace 行。返回 trace_id。"""
        self.status = status if self.status != STATUS_FAILED else STATUS_FAILED
        if status == STATUS_FAILED:
            self.status = STATUS_FAILED
        self.error = mask_secrets(error)[:500]
        if first_token_ms is not None:
            self.first_token_ms = int(first_token_ms)
        self.total_ms = int(total_ms if total_ms is not None else self.elapsed_ms())
        try:
            _insert_spans(self.trace_id, self._spans)
            self._persist_trace(insert=False)
        except Exception as exc:  # 观测失败只告警，绝不影响主流程
            logger.warning("写入观测数据失败（不影响主流程）: %s", exc)
        return self.trace_id

    # -- 落库 --

    def _trace_row(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "biz_type": self.biz_type,
            "kb_id": self.kb_id or "",
            "session_id": self.session_id or "",
            "question": mask_secrets(self.question or "")[:1000],
            "rewritten_question": self.rewritten_question,
            "expansions": json.dumps(self.expansions, ensure_ascii=False),
            "hyde_answer": _truncate(self.hyde_answer, 2000),
            "config_snapshot": json.dumps(self.config_snapshot, ensure_ascii=False, default=str),
            "status": self.status,
            "error": self.error,
            "recall_empty": 1 if self.recall_empty else 0,
            "degrade_count": len(self.detail.get("degradations", []) or []),
            "first_token_ms": self.first_token_ms,
            "total_ms": self.total_ms,
            "detail": json.dumps(self.detail, ensure_ascii=False, default=str)[:20000],
            "finished_at": datetime.now().isoformat(timespec="seconds")
            if self.status != STATUS_RUNNING else None,
        }

    def _persist_trace(self, insert: bool) -> None:
        row = self._trace_row()
        try:
            conn = database.get_connection()
            if insert:
                row["created_at"] = datetime.now().isoformat(timespec="seconds")
                columns = ", ".join(row.keys())
                placeholders = ", ".join(["?"] * len(row))
                conn.execute(f"INSERT OR REPLACE INTO rag_traces ({columns}) VALUES ({placeholders})",
                             tuple(row.values()))
            else:
                setters = ", ".join(f"{key}=?" for key in row)
                conn.execute(f"UPDATE rag_traces SET {setters} WHERE trace_id=?",
                             (*row.values(), self.trace_id))
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("持久化 trace 失败（不影响主流程）: %s", exc)


def _insert_spans(trace_id: str, spans: list[dict]) -> None:
    if not spans:
        return
    conn = database.get_connection()
    created_at = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT INTO rag_spans (trace_id, seq, stage, started_at, duration_ms, status,"
        " input_count, output_count, error, detail, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                trace_id, span["seq"], span["stage"],
                datetime.fromtimestamp(span["started_at"]).isoformat(timespec="milliseconds"),
                span["duration_ms"], span["status"], span["input_count"], span["output_count"],
                span["error"], json.dumps(span["detail"], ensure_ascii=False, default=str)[:4000],
                created_at,
            )
            for span in spans
        ],
    )
    conn.commit()
    conn.close()


# ── 上下文传播（免改函数签名的埋点方式）──────────────────

_current: ContextVar[TraceRecorder | None] = ContextVar("rag_trace_recorder", default=None)


def current() -> TraceRecorder | None:
    """当前请求的追踪记录器；没有开启追踪时返回 None（所有埋点自动跳过）。"""
    return _current.get()


@contextmanager
def active_trace(recorder: TraceRecorder | None):
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def mark(stage: str, **kwargs) -> None:
    """便捷埋点：在追踪上下文中记一条阶段，未开启追踪时什么也不做。"""
    recorder = current()
    if recorder is not None:
        recorder.mark(stage, **kwargs)


def degrade(stage: str, reason: str, fallback: str = "") -> None:
    """便捷降级埋点：**既**记一条 degraded 阶段，**也**写进执行详情的「降级记录」。

    【为什么单独有它】模块级 mark(status=degraded) 只写 span，不会进 detail.degradations，
    导致降级在执行详情里"看得见阶段、看不见原因与兜底"——需求要求每次降级都可见，故补此入口。
    未开启追踪时什么也不做。
    """
    recorder = current()
    if recorder is not None:
        recorder.degrade(stage, reason, fallback)


def note_degradation(stage: str, reason: str, fallback: str = "") -> None:
    """只补一条「降级记录」（不新增阶段行），配合手写 span.status=degraded 使用。"""
    recorder = current()
    if recorder is not None:
        recorder.add_degradation(stage, reason, fallback)


@contextmanager
def span(stage: str, input_count: int | None = None, **detail):
    """在追踪上下文中包一个阶段；**没有开启追踪时是零成本空操作**。

    这样业务代码（如检索管道）可以无条件埋点，不必到处判断是否有追踪。
    阶段内抛异常时会被记成 failed 后继续抛出，不影响原有异常语义。
    """
    recorder = current()
    if recorder is None:
        yield SpanHandle(stage=stage, input_count=input_count, detail=dict(detail))
        return
    with recorder.span(stage, input_count, **detail) as handle:
        yield handle


def record_span(trace_id: str, stage: str, *, duration_ms: int = 0, status: str = STATUS_OK,
                input_count: int | None = None, output_count: int | None = None,
                error: str = "", **detail) -> None:
    """给指定 trace 追记一条阶段（用于跨进程阶段，如路由侧的文件保存）。

    seq 固定为 0：读取时按插入顺序展示，这条会排在 Worker 的阶段之前。
    失败只告警，不影响调用方。
    """
    try:
        _insert_spans(trace_id, [{
            "seq": 0,
            "stage": stage,
            "started_at": time.time() - duration_ms / 1000.0,
            "duration_ms": duration_ms,
            "status": status,
            "input_count": input_count,
            "output_count": output_count,
            "error": mask_secrets(error)[:500],
            "detail": sanitize({k: _truncate(v) for k, v in detail.items()}),
        }])
    except Exception as exc:
        logger.warning("追记 span 失败（不影响主流程）: %s", exc)


# ── 查询与聚合（运行观测面板）───────────────────────────

def list_traces(limit: int = 20, offset: int = 0, biz_type: str | None = None,
                status: str | None = None, kb_ids: list[str] | None = None,
                hours: int | None = None) -> dict:
    conn = database.get_connection()
    where, params = [], []
    if biz_type:
        where.append("biz_type = ?")
        params.append(biz_type)
    if status:
        where.append("status = ?")
        params.append(status)
    if hours:
        # 时间快捷范围（"最近 1 小时 / 24 小时 / 7 天"）：用服务端时钟换算，避免时区问题
        cutoff = (datetime.now() - timedelta(hours=max(int(hours), 1))).isoformat(timespec="seconds")
        where.append("created_at >= ?")
        params.append(cutoff)
    if kb_ids is not None:
        if not kb_ids:
            conn.close()
            return {"items": [], "total": 0}
        where.append(f"kb_id IN ({','.join(['?'] * len(kb_ids))})")
        params.extend(kb_ids)
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    total = conn.execute(f"SELECT COUNT(*) FROM rag_traces{clause}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT trace_id, biz_type, kb_id, session_id, question, rewritten_question,"
        f" expansions, hyde_answer, status, error, degrade_count, recall_empty,"
        f" first_token_ms, total_ms, created_at, finished_at"
        f" FROM rag_traces{clause} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    ).fetchall()
    conn.close()
    items = []
    for row in rows:
        item = dict(row)
        try:
            item["expansions"] = json.loads(item["expansions"] or "[]")
        except json.JSONDecodeError:
            item["expansions"] = []
        items.append(item)
    return {"items": items, "total": total}


def get_trace(trace_id: str) -> dict | None:
    conn = database.get_connection()
    trace = conn.execute("SELECT * FROM rag_traces WHERE trace_id = ?", (trace_id,)).fetchone()
    if trace is None:
        conn.close()
        return None
    spans = conn.execute(
        "SELECT * FROM rag_spans WHERE trace_id = ? ORDER BY id", (trace_id,)
    ).fetchall()
    conn.close()

    result = dict(trace)
    for key in ("expansions", "config_snapshot", "detail"):
        try:
            result[key] = json.loads(result.get(key) or ("[]" if key == "expansions" else "{}"))
        except (json.JSONDecodeError, TypeError):
            result[key] = [] if key == "expansions" else {}
    result["recall_empty"] = bool(result.get("recall_empty"))
    span_items = []
    for span in spans:
        item = dict(span)
        try:
            item["detail"] = json.loads(item.get("detail") or "{}")
        except json.JSONDecodeError:
            item["detail"] = {}
        span_items.append(item)
    result["spans"] = span_items
    return result


def overview(hours: int = 24, biz_type: str | None = None, dependencies: dict | None = None,
             slow_ms: int = 5000, kb_ids: list[str] | None = None) -> dict:
    """运行观测面板的聚合数据（问答量、成功率、耗时、能力使用次数、文档任务、错误）。

    kb_ids 非 None 时只统计这些知识库的记录（普通用户只看自己的库）。
    """
    since = (datetime.now() - timedelta(hours=max(hours, 1))).isoformat(timespec="seconds")
    conn = database.get_connection()
    where = "WHERE created_at >= ?"
    params: list = [since]
    if biz_type:
        where += " AND biz_type = ?"
        params.append(biz_type)
    if kb_ids is not None:
        if not kb_ids:
            conn.close()
            return {
                "window_hours": hours, "generated_at": datetime.now().isoformat(timespec="seconds"),
                "qa": {"requests": 0, "success": 0, "failed": 0, "running": 0, "success_rate": 0.0,
                       "avg_total_ms": 0, "p95_total_ms": 0, "slow_requests": 0,
                       "slow_threshold_ms": slow_ms, "avg_first_token_ms": 0,
                       "no_recall_ratio": 0.0, "degrade_total": 0},
                "features": {"rewrite": 0, "multi_query": 0, "hyde": 0},
                "rerank": {"calls": 0, "avg_ms": 0, "failed": 0},
                "document_tasks": {"by_status": {}, "processing": 0, "completed": 0, "failed": 0,
                                   "total": 0, "by_type": {}},
                "recent_errors": [], "dependencies": dependencies or {},
                "scope": {"kb_ids": [], "note": "没有可统计的知识库"},
            }
        where += f" AND kb_id IN ({','.join(['?'] * len(kb_ids))})"
        params.extend(kb_ids)

    rows = conn.execute(
        f"SELECT biz_type, status, total_ms, first_token_ms, recall_empty,"
        f" degrade_count, rewritten_question, expansions, hyde_answer"
        f" FROM rag_traces {where} ORDER BY created_at DESC LIMIT 5000",
        params,
    ).fetchall()
    span_conditions = ["t.created_at >= ?"]
    span_params: list = [since]
    if biz_type:
        span_conditions.append("t.biz_type = ?")
        span_params.append(biz_type)
    if kb_ids is not None:
        span_conditions.append(f"t.kb_id IN ({','.join(['?'] * len(kb_ids))})")
        span_params.extend(kb_ids)
    spans = conn.execute(
        "SELECT s.stage, s.duration_ms, s.status FROM rag_spans s JOIN rag_traces t"
        f" ON t.trace_id = s.trace_id WHERE {' AND '.join(span_conditions)}"
        " AND s.stage IN ('Rerank', '问题改写', '多查询/HyDE', '查询向量化')",
        span_params,
    ).fetchall()

    task_clause, task_params = "", []
    if kb_ids is not None:
        task_clause = f" WHERE kb_id IN ({','.join(['?'] * len(kb_ids))})"
        task_params = list(kb_ids)
    task_rows = conn.execute(
        f"SELECT status, COUNT(*) AS count FROM upload_tasks{task_clause} GROUP BY status",
        task_params,
    ).fetchall()
    task_types = conn.execute(
        f"SELECT task_type, COUNT(*) AS count FROM upload_tasks{task_clause} GROUP BY task_type",
        task_params,
    ).fetchall()
    error_clause = where + " AND status = 'failed'"
    recent_errors = conn.execute(
        f"SELECT trace_id, biz_type, kb_id, question, error, created_at FROM rag_traces"
        f" {error_clause} ORDER BY created_at DESC LIMIT 10",
        params,
    ).fetchall()
    storage = {}
    if kb_ids is None:
        bounds = conn.execute(
            "SELECT MIN(created_at) AS earliest, MAX(created_at) AS latest FROM rag_traces"
        ).fetchone()
        storage = {
            "traces": conn.execute("SELECT COUNT(*) FROM rag_traces").fetchone()[0],
            "spans": conn.execute("SELECT COUNT(*) FROM rag_spans").fetchone()[0],
            # 时间范围：让用户在清理前判断"这批数据还有没有排查价值"
            "earliest": bounds["earliest"] if bounds else None,
            "latest": bounds["latest"] if bounds else None,
        }
    conn.close()

    qa_rows = [row for row in rows if row["biz_type"] == "qa"] or [row for row in rows]
    durations = sorted(int(row["total_ms"] or 0) for row in qa_rows if row["total_ms"])
    first_tokens = [int(row["first_token_ms"]) for row in qa_rows if row["first_token_ms"]]
    success = sum(1 for row in qa_rows if row["status"] == STATUS_OK)
    failed = sum(1 for row in qa_rows if row["status"] == STATUS_FAILED)
    running = sum(1 for row in qa_rows if row["status"] == STATUS_RUNNING)

    def _percentile(values: list[int], ratio: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, int(len(values) * ratio))
        return values[index]

    rerank_spans = [int(span["duration_ms"] or 0) for span in spans if span["stage"] == "Rerank"]
    feature_usage = {
        "rewrite": sum(1 for row in qa_rows if (row["rewritten_question"] or "").strip()),
        "multi_query": sum(1 for row in qa_rows if (row["expansions"] or "[]") not in ("[]", "")),
        "hyde": sum(1 for row in qa_rows if (row["hyde_answer"] or "").strip()),
    }
    task_stats: dict[str, int] = {}
    for row in task_rows:
        task_stats[row["status"]] = row["count"]

    return {
        "window_hours": hours,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "qa": {
            "requests": len(qa_rows),
            "success": success,
            "failed": failed,
            "running": running,
            "success_rate": round(success / len(qa_rows), 4) if qa_rows else 0.0,
            "avg_total_ms": int(sum(durations) / len(durations)) if durations else 0,
            "p95_total_ms": _percentile(durations, 0.95),
            "slow_requests": sum(1 for value in durations if value > slow_ms),
            "slow_threshold_ms": slow_ms,
            "avg_first_token_ms": int(sum(first_tokens) / len(first_tokens)) if first_tokens else 0,
            "no_recall_ratio": round(
                sum(1 for row in qa_rows if row["recall_empty"]) / len(qa_rows), 4
            ) if qa_rows else 0.0,
            "degrade_total": sum(int(row["degrade_count"] or 0) for row in qa_rows),
        },
        "features": feature_usage,
        "rerank": {
            "calls": len(rerank_spans),
            "avg_ms": int(sum(rerank_spans) / len(rerank_spans)) if rerank_spans else 0,
            "failed": sum(1 for span in spans if span["stage"] == "Rerank" and span["status"] == "failed"),
        },
        "document_tasks": {
            "by_status": task_stats,
            "processing": task_stats.get("parsing", 0) + task_stats.get("indexing", 0)
            + task_stats.get("uploaded", 0) + task_stats.get("dispatch_unknown", 0),
            "completed": task_stats.get("completed", 0),
            "failed": task_stats.get("failed", 0),
            "total": sum(task_stats.values()),
            "by_type": {row["task_type"]: row["count"] for row in task_types},
        },
        "storage": storage,
        "recent_errors": [dict(row) for row in recent_errors],
        "dependencies": dependencies or {},
    }


# ── 手动清理（不自动清理）──────────────────────────────

def export_data(limit: int = 2000, kb_ids: list[str] | None = None) -> dict:
    """导出观测数据（traces + 对应 spans），用于"先备份再清理"。

    生产考虑：
      - 有硬上限（默认 2000 条 trace，最大 5000），避免一次性把内存/响应撑爆；
      - `truncated=True` 表示还有更早的数据没导出，提示可以调大 limit 分次导出；
      - 只读操作，不修改任何数据。
    """
    limit = max(1, min(int(limit or 2000), 5000))
    where, params = "", []
    if kb_ids:
        where = f" WHERE kb_id IN ({','.join(['?'] * len(kb_ids))})"
        params = list(kb_ids)

    conn = database.get_connection()
    trace_rows = conn.execute(
        f"SELECT * FROM rag_traces{where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    span_rows: list = []
    trace_ids = [row["trace_id"] for row in trace_rows]
    if trace_ids:
        placeholders = ",".join(["?"] * len(trace_ids))
        span_rows = conn.execute(
            f"SELECT * FROM rag_spans WHERE trace_id IN ({placeholders}) ORDER BY trace_id, id",
            trace_ids,
        ).fetchall()
    total = conn.execute(f"SELECT COUNT(*) FROM rag_traces{where}", params).fetchone()[0]
    conn.close()

    traces = []
    for row in trace_rows:
        item = dict(row)
        for key in ("expansions", "config_snapshot", "detail"):
            try:
                item[key] = json.loads(item.get(key) or ("[]" if key == "expansions" else "{}"))
            except (json.JSONDecodeError, TypeError):
                item[key] = [] if key == "expansions" else {}
        traces.append(item)

    spans = []
    for row in span_rows:
        item = dict(row)
        try:
            item["detail"] = json.loads(item.get("detail") or "{}")
        except json.JSONDecodeError:
            item["detail"] = {}
        spans.append(item)

    return {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "limit": limit,
        "total_traces_in_store": total,
        "truncated": total > len(traces),
        "traces": traces,
        "spans": spans,
    }


def cleanup_traces(mode: str = "all", days: int = 7, trace_ids: list[str] | None = None,
                   dry_run: bool = False, hours: int | None = None) -> dict:
    """手动清理观测数据。

    mode:
      - all        清理全部
      - older_than 清理 N 天前（days 指定；"保留最近 N 天"）
      - range      清理**最近 N 小时内**的记录（hours 指定；对应"像浏览器历史那样按时间范围删"）
      - trace_ids  只清理指定 trace（批量勾选删除、单条删除都走它）
    dry_run=True 时只返回将删除的条数，不真正删除。
    """
    conn = database.get_connection()
    where, params = "", []
    if mode == "older_than":
        cutoff = (datetime.now() - timedelta(days=max(days, 1))).isoformat(timespec="seconds")
        where = " WHERE created_at < ?"
        params = [cutoff]
    elif mode == "range":
        if not hours:
            conn.close()
            raise ValueError("mode=range 必须提供 hours（清理最近 N 小时内的记录）")
        cutoff = (datetime.now() - timedelta(hours=max(int(hours), 1))).isoformat(timespec="seconds")
        where = " WHERE created_at >= ?"
        params = [cutoff]
    elif mode == "trace_ids":
        ids = list(dict.fromkeys(trace_ids or []))
        if not ids:
            conn.close()
            # fail fast：破坏性操作不给"空列表"留含糊空间（前端也会先拦一次）
            raise ValueError("mode=trace_ids 时必须提供 trace_ids（要删除的追踪编号）")
        where = f" WHERE trace_id IN ({','.join(['?'] * len(ids))})"
        params = ids
    elif mode != "all":
        conn.close()
        raise ValueError("mode 仅支持 all / older_than / range / trace_ids")

    matched = conn.execute(f"SELECT COUNT(*) FROM rag_traces{where}", params).fetchone()[0]
    matched_spans = conn.execute(
        f"SELECT COUNT(*) FROM rag_spans WHERE trace_id IN (SELECT trace_id FROM rag_traces{where})",
        params,
    ).fetchone()[0]

    deleted_traces = deleted_spans = 0
    if not dry_run:
        conn.execute(
            f"DELETE FROM rag_spans WHERE trace_id IN (SELECT trace_id FROM rag_traces{where})",
            params,
        )
        deleted_spans = conn.execute("SELECT changes()").fetchone()[0]
        conn.execute(f"DELETE FROM rag_traces{where}", params)
        deleted_traces = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
    # 剩余量：前端据此显示"清理后还剩多少"，让"确实清掉了"可见
    remaining_traces = conn.execute("SELECT COUNT(*) FROM rag_traces").fetchone()[0]
    remaining_spans = conn.execute("SELECT COUNT(*) FROM rag_spans").fetchone()[0]
    conn.close()

    return {
        "mode": mode,
        "days": days if mode == "older_than" else None,
        "dry_run": dry_run,
        "matched_traces": matched,
        "matched_spans": matched_spans,
        "deleted_traces": deleted_traces,
        "deleted_spans": deleted_spans,
        "remaining_traces": remaining_traces,
        "remaining_spans": remaining_spans,
    }


# ── 健康检查（存活 / 就绪）─────────────────────────────

def _check_sqlite() -> dict:
    started = time.time()
    try:
        conn = database.get_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return {"status": "ok", "latency_ms": int((time.time() - started) * 1000), "error": ""}
    except Exception as exc:
        return {"status": "failed", "latency_ms": int((time.time() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"}


def _check_redis(timeout: float = 2.0) -> dict:
    started = time.time()
    try:
        from config import CELERY_BROKER_URL
        import redis  # celery[redis] 已安装

        client = redis.Redis.from_url(
            CELERY_BROKER_URL, socket_connect_timeout=timeout, socket_timeout=timeout
        )
        client.ping()
        return {"status": "ok", "latency_ms": int((time.time() - started) * 1000), "error": ""}
    except Exception as exc:
        return {"status": "failed", "latency_ms": int((time.time() - started) * 1000),
                "error": mask_secrets(f"{type(exc).__name__}: {exc}")}


def _check_milvus(timeout: float = 3.0) -> dict:
    started = time.time()
    try:
        from config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION

        sock = socket.create_connection((MILVUS_HOST, int(MILVUS_PORT)), timeout=timeout)
        sock.close()
        # TCP 通只说明端口可达，再用一次元数据查询确认服务可用
        import milvus_ops
        from pymilvus import utility

        collection_ready = utility.has_collection(MILVUS_COLLECTION)
        return {"status": "ok", "latency_ms": int((time.time() - started) * 1000),
                "error": "", "detail": {"collection": MILVUS_COLLECTION,
                                        "collection_exists": bool(collection_ready)}}
    except Exception as exc:
        return {"status": "failed", "latency_ms": int((time.time() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"}


def _check_bge(timeout: float = 5.0) -> dict:
    started = time.time()
    try:
        import requests
        from config import BGE_URL

        response = requests.post(f"{BGE_URL.rstrip('/')}/api/emb/embedding",
                                 json={"content": "健康检查"}, timeout=timeout)
        ok = response.status_code == 200
        return {"status": "ok" if ok else "failed",
                "latency_ms": int((time.time() - started) * 1000),
                "error": "" if ok else f"HTTP {response.status_code}"}
    except Exception as exc:
        return {"status": "failed", "latency_ms": int((time.time() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"}


def _check_llm(timeout: float = 5.0) -> dict:
    started = time.time()
    try:
        import requests
        from config import LLM_API_URL, LLM_API_KEY

        headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        response = requests.get(f"{LLM_API_URL.rstrip('/')}/models", headers=headers, timeout=timeout)
        ok = response.status_code == 200
        return {"status": "ok" if ok else "failed",
                "latency_ms": int((time.time() - started) * 1000),
                "error": "" if ok else f"HTTP {response.status_code}（服务可达但返回异常）"}
    except Exception as exc:
        return {"status": "failed", "latency_ms": int((time.time() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"}


def check_dependencies(timeout: float = 3.0) -> dict:
    """逐项探测外部依赖：数据库 / Redis / Milvus / Embedding / 大模型。

    每项都返回 status / latency_ms / error，便于明确指出"是哪个依赖不可用"，
    而不是笼统的"系统异常"。
    """
    dependencies = {
        "sqlite": _check_sqlite(),
        "redis": _check_redis(timeout),
        "milvus": _check_milvus(timeout),
        "bge_embedding": _check_bge(timeout + 2),
        "llm": _check_llm(timeout + 2),
    }
    unhealthy = [name for name, item in dependencies.items() if item["status"] != "ok"]
    return {
        "status": "ok" if not unhealthy else "degraded",
        "unhealthy": unhealthy,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "dependencies": dependencies,
    }
