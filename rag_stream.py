"""
RAG 流式输出共享逻辑（Ch12 新增）。

负责"阶段二：LLM 流式生成器"，与阶段一 prepare_context 分开。
三种流式方案（HTTP Chunked / SSE / WebSocket）复用同一个生成器，
只在其路由层做不同的协议序列化。

事件协议：
  citations → token × N → done | error
"""
import time
import threading
import logging
from uuid import uuid4

import database
import observability
import langsmith_tracing
from embedding import llm_client

logger = logging.getLogger("rag.stream")


def stream_tokens_generator(
    messages: list,
    session_id: str,
    citations: list[dict],
    stop_event=None,
    trace: "observability.TraceRecorder | None" = None,
):
    """
    阶段二：LLM 流式生成器，逐 token yield 字典。

    使用 llm_client.stream() 逐 token 产出。
    生成完成后自动保存消息到数据库。

    yield 格式:
        {"event": "token", "data": {"content": "..."}}  — 每个 token
        {"event": "done", "data": {"answer_id": "...", "session_id": "..."}}  — 完成
        {"event": "error", "data": {"message": "..."}}  — 错误

    调用方（三种路由）负责将 yield 的字典序列化为对应协议格式。

    空召回（messages is None）：发固定提示 token → 保存 → 正常 done。
    取消（stop_event 置位）：在下一个检查点与保存前检测，不保存迟到/部分回答。
    """
    full_answer = ""
    stream = None
    started = time.time()
    first_token_ms = None
    try:
        if stop_event is not None and stop_event.is_set():
            return
        if messages is None:
            full_answer = "未检索到相关内容，请先上传文档到知识库。"
            first_token_ms = int((time.time() - started) * 1000)
            yield {"event": "token", "data": {"content": full_answer}}
        else:
            # LangSmith 未开启时原样返回 llm_client（零开销）
            stream = langsmith_tracing.traced_llm(llm_client, mode="stream", stream=True).stream(messages)
        for chunk in stream or ():
            if stop_event is not None and stop_event.is_set():
                return
            token = chunk.content
            if token:
                if first_token_ms is None:
                    # 首 Token 等待时间：从进入生成器到第一个 token 产出
                    first_token_ms = int((time.time() - started) * 1000)
                full_answer += token
                yield {"event": "token", "data": {"content": token}}

        # 流式结束后，未取消才保存完整回答
        if stop_event is not None and stop_event.is_set():
            return
        database.save_message(session_id, "assistant", full_answer, citations)
        total_ms = int((time.time() - started) * 1000)
        if trace is not None:
            trace.mark(observability.STAGE_GENERATE, duration_ms=total_ms,
                       input_count=len(messages or []), output_count=len(full_answer),
                       first_token_ms=first_token_ms)
            trace.mark(observability.STAGE_STREAM_DONE, duration_ms=total_ms,
                       output_count=len(full_answer))
            trace.finish(status=observability.STATUS_OK, first_token_ms=first_token_ms,
                         total_ms=trace.elapsed_ms())
        yield {
            "event": "done",
            "data": {"answer_id": str(uuid4()), "session_id": session_id,
                     "first_token_ms": first_token_ms, "total_ms": total_ms,
                     "trace_id": trace.trace_id if trace is not None else None},
        }
    except Exception as exc:
        logger.exception("流式生成异常: %s", exc)
        if trace is not None:
            trace.mark(observability.STAGE_GENERATE, status=observability.STATUS_FAILED,
                       duration_ms=int((time.time() - started) * 1000), error=str(exc))
            trace.finish(status=observability.STATUS_FAILED, error=str(exc),
                         first_token_ms=first_token_ms)
        if stop_event is None or not stop_event.is_set():
            yield {"event": "error", "data": {"message": str(exc)}}
    finally:
        if stream is not None and hasattr(stream, "close"):
            stream.close()