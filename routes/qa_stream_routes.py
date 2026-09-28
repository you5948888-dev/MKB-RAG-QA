"""流式问答路由（Ch12 新增）：三种流式方案 + WebSocket 认证。

方案 A：SSE（Server-Sent Events），业界主流，sse-starlette 序列化。
方案 B：WebSocket，全双工；本节聚焦流式输出，一条连接处理一个问题。
方案 C：HTTP Chunked（NDJSON），最简单。

三者共用同一事件协议（citations → token × N → done | error），
只是传输编码不同。检索与拼上下文统一走 prepare_context。
"""
import json
import threading
import asyncio
import logging

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import auth
import observability
from auth import ensure_kb_access as _ensure_kb_access
from models import QueryRequest
from rag_pipeline import prepare_context
from rag_stream import stream_tokens_generator

logger = logging.getLogger("rag.routes.qa_stream")

router = APIRouter(prefix="/api/qa/stream", tags=["qa-stream"])


def _strategy_override(strategy: dict | None, mode: str | None) -> dict | None:
    """拼接本次请求的策略临时覆盖（Advanced RAG 第 3 期），None 表示沿用知识库策略。"""
    override = dict(strategy or {})
    if mode:
        override["mode"] = mode
    return override or None


def _apply_ctx_to_trace(trace, ctx: dict) -> None:
    """把检索阶段的产物写进追踪记录（问题计划、配置快照、降级）。"""
    plan = ctx.get("query_plan") or {}
    trace.set_result(
        rewritten_question=plan.get("rewritten") or None,
        expansions=plan.get("multi_queries"),
        hyde_answer=plan.get("hyde_answer"),
        config_snapshot=ctx.get("strategy_snapshot"),
        recall_empty=(ctx.get("messages") is None),
    )
    # 降级不在这里补记：检索管道已在发生处写进本 trace（避免同一次降级被登记两遍）。
    # ctx["degradations"] 仍随事件回传前端展示。


def _verify_ws_user(websocket: WebSocket, request: dict) -> dict | None:
    """
    WebSocket 认证：优先从 Cookie 验证 session，其次从消息体 token 验证 API Key。

    难点：FastAPI 的 HTTP 认证依赖（get_current_user）不适用于 WebSocket，
    因此这里手动实现两种认证路径。
    """
    # 方式 1：Cookie 中的签名 session
    cookie_val = websocket.cookies.get(auth.AUTH_COOKIE_NAME)
    if cookie_val:
        user_id = auth._verify_session_cookie(cookie_val)
        if user_id:
            user = auth.get_user_by_id(user_id)
            if user:
                return user
    # 方式 2：消息体中的 API Key
    token = (request.get("token") or "").strip()
    if token:
        user = auth.get_user_by_api_key(token)
        if user:
            return user
    return None


# ── 方案 C：HTTP Chunked（NDJSON）────────────────────────

@router.post("/http")
def stream_http(payload: QueryRequest, user: dict = Depends(auth.get_current_user)):
    """
    方案 C：HTTP Chunked Transfer，NDJSON 格式。

    最简单的流式方案——每行一个 JSON，用 \\n 分隔。
    前端用 fetch + ReadableStream + TextDecoder 逐行解析。
    """
    auth.ensure_kb_access(payload.kb_id, user)
    trace = observability.TraceRecorder.start(
        "qa", kb_id=payload.kb_id, session_id=payload.session_id, question=payload.question)
    with observability.active_trace(trace):
        with trace.span(observability.STAGE_RECEIVE, input_count=1) as span:
            span.set_output(1, transport="http")
        try:
            ctx = prepare_context(payload.kb_id, payload.question, payload.session_id,
                                  strategy=_strategy_override(payload.strategy, payload.mode))
        except ValueError as exc:
            trace.finish(status=observability.STATUS_FAILED, error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
    _apply_ctx_to_trace(trace, ctx)

    session_id = ctx["session_id"]
    citations = ctx["citations"]

    async def ndjson_generator():
        stop_event = threading.Event()
        stream = stream_tokens_generator(ctx["messages"], session_id, citations,
                                         stop_event=stop_event, trace=trace)
        try:
            # strategy_snapshot：前端据此在回答下方标注本次用的是基础/高级模式（只加字段）
            yield json.dumps({"event": "citations",
                              "data": {"citations": citations, "trace_id": trace.trace_id,
                                       "strategy_snapshot": ctx.get("strategy_snapshot")}},
                             ensure_ascii=False) + "\n"
            while (evt := await asyncio.to_thread(next, stream, None)) is not None:
                yield json.dumps(evt, ensure_ascii=False) + "\n"
        finally:
            # 阻塞的模型调用在自身超时下结束后，观察取消标志
            stop_event.set()
            if trace.status == observability.STATUS_RUNNING:
                # 客户端提前断开/取消：补一条收尾，避免追踪永远停在 running
                trace.finish(status=observability.STATUS_OK, error="流式连接提前结束（未生成完成）")

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},  # 禁用反向代理缓冲，否则流式不生效
    )


# ── 方案 A：SSE ──────────────────────────────────────────

@router.post("/sse")
def stream_sse(payload: QueryRequest, user: dict = Depends(auth.get_current_user)):
    """
    方案 A：SSE (Server-Sent Events) 流式问答。

    SSE 是 HTTP/1.1 标准的服务端推送协议。
    使用 sse-starlette 库自动处理 SSE 格式化（event: / data: 前缀）。
    前端用 fetch + ReadableStream 消费（也可用 EventSource，但不支持 POST）。
    """
    from sse_starlette.sse import EventSourceResponse

    auth.ensure_kb_access(payload.kb_id, user)
    trace = observability.TraceRecorder.start(
        "qa", kb_id=payload.kb_id, session_id=payload.session_id, question=payload.question)
    with observability.active_trace(trace):
        with trace.span(observability.STAGE_RECEIVE, input_count=1) as span:
            span.set_output(1, transport="sse")
        try:
            ctx = prepare_context(payload.kb_id, payload.question, payload.session_id,
                                  strategy=_strategy_override(payload.strategy, payload.mode))
        except ValueError as exc:
            trace.finish(status=observability.STATUS_FAILED, error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
    _apply_ctx_to_trace(trace, ctx)

    session_id = ctx["session_id"]
    citations = ctx["citations"]

    async def event_generator():
        stop_event = threading.Event()
        stream = stream_tokens_generator(ctx["messages"], session_id, citations,
                                         stop_event=stop_event, trace=trace)
        try:
            yield {
                "event": "citations",
                "data": json.dumps({"citations": citations, "trace_id": trace.trace_id,
                                    "strategy_snapshot": ctx.get("strategy_snapshot")},
                                   ensure_ascii=False),
            }
            while (evt := await asyncio.to_thread(next, stream, None)) is not None:
                yield {
                    "event": evt["event"],
                    "data": json.dumps(evt["data"], ensure_ascii=False),
                }
        finally:
            stop_event.set()
            if trace.status == observability.STATUS_RUNNING:
                trace.finish(status=observability.STATUS_OK, error="流式连接提前结束（未生成完成）")

    return EventSourceResponse(event_generator())


# ── 方案 B：WebSocket ────────────────────────────────────

@router.websocket("/ws")
async def stream_ws(websocket: WebSocket):
    """
    方案 B：WebSocket 流式问答。

    WebSocket 协议本身支持全双工通信。本章为了聚焦流式输出，
    每条连接只处理首个问题；额外的接收任务只监听取消或断开。

    认证方式：本例从 Cookie 或消息体中手动认证；HTTP 专用依赖不能原样复用。

    ⚠️ 关键设计：stream_tokens_generator 是同步生成器，
    直接在 async def 中迭代会阻塞事件循环（所有 token 一次性涌出）。
    解决方案：用后台线程运行生成器，通过 asyncio.Queue 桥接到异步 WebSocket。
    """
    await websocket.accept()
    stop_event = threading.Event()
    thread = None
    disconnect = None
    pending_event = None
    trace = None

    try:
        # 接收客户端消息
        raw = await websocket.receive_text()
        request = json.loads(raw)

        # 手动认证：从消息中取 token，或从 Cookie 验证
        user = _verify_ws_user(websocket, request)
        if not user:
            await websocket.send_json({"event": "error", "data": {"message": "未认证"}})
            await websocket.close()
            return

        question = request.get("question", "").strip()
        kb_id = request.get("kb_id", "").strip()
        session_id = request.get("session_id")

        if not question or not kb_id:
            await websocket.send_json({"event": "error", "data": {"message": "缺少 question 或 kb_id"}})
            await websocket.close()
            return

        # 检查知识库权限
        try:
            _ensure_kb_access(kb_id, user)
        except Exception as exc:
            await websocket.send_json({"event": "error", "data": {"message": str(exc)}})
            await websocket.close()
            return

        # 检索放在线程中，避免阻塞事件循环。
        trace = observability.TraceRecorder.start(
            "qa", kb_id=kb_id, session_id=session_id, question=question)
        with observability.active_trace(trace):
            with trace.span(observability.STAGE_RECEIVE, input_count=1) as span:
                span.set_output(1, transport="websocket")
            ctx = await asyncio.to_thread(
                prepare_context, kb_id, question, session_id,
                strategy=_strategy_override(request.get("strategy"), request.get("mode")),
            )
        _apply_ctx_to_trace(trace, ctx)

        # 发送引用
        await websocket.send_json({
            "event": "citations",
            "data": {"citations": ctx["citations"], "trace_id": trace.trace_id,
                     "strategy_snapshot": ctx.get("strategy_snapshot")},
        })

        # ── 核心：后台线程 + asyncio.Queue 桥接 ──
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_stream_in_thread():
            """后台线程：运行同步生成器，将事件放入队列。"""
            try:
                for evt in stream_tokens_generator(ctx["messages"], ctx["session_id"], ctx["citations"],
                                                   stop_event=stop_event, trace=trace):
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, evt)
            except Exception as exc:
                if not stop_event.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "data": {"message": str(exc)}})
            finally:
                # None 作为结束信号
                if not stop_event.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_run_stream_in_thread, daemon=True)
        thread.start()
        disconnect = asyncio.create_task(websocket.receive())

        # 异步消费队列，逐 token 推送
        while True:
            pending_get = asyncio.create_task(queue.get())
            ready, _ = await asyncio.wait({pending_get, disconnect}, return_when=asyncio.FIRST_COMPLETED)
            if disconnect in ready:
                break
            evt = pending_get.result()
            pending_get = None
            if evt is None:
                break
            await websocket.send_json(evt)

        await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开连接")
    except Exception as exc:
        logger.warning("WebSocket 异常: %s", exc)
        try:
            await websocket.send_json({"event": "error", "data": {"message": str(exc)}})
        except Exception:
            pass
    finally:
        stop_event.set()
        if trace is not None and trace.status == observability.STATUS_RUNNING:
            trace.finish(status=observability.STATUS_OK, error="WebSocket 连接提前结束（未生成完成）")
        for pending in (disconnect, pending_event):
            if pending is not None:
                pending.cancel()
        if thread is not None:
            await asyncio.to_thread(thread.join, 1)
        try:
            await websocket.close()
        except RuntimeError:
            pass