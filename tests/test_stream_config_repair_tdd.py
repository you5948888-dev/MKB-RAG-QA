"""
回归测试：流式事件协议、取消语义与管道参数优先级。

测试范围：
  - HTTP NDJSON：事件顺序 citations → token × N → done，并落库完整回答
  - 空召回：固定提示 → 正常 done（不是 error）
  - 生成异常：error 为终态，其后不再有 done，且不保存部分回答
  - SSE：事件被序列化成 SSE 帧
  - WebSocket：未认证被拒；消息体 API Key 认证后正常流式
  - 取消：预置取消标志 / 中途取消都不保存迟到回答
  - 参数优先级：显式参数覆盖 pipeline_config，未传时回落到配置

说明：流式生成器是同步生成器，本文件用离线 LLM 替身驱动它，
从而在没有模型服务时也能精确断言事件序列。
"""
import json
import threading

import database

SESSION_ID = "session-stream-test"


def _patch_prepare_context(monkeypatch, messages=("fake-message",), citations=None):
    """替换三种流式路由共用的检索阶段，只聚焦流式输出行为。"""
    import routes.qa_stream_routes as qa_stream_routes

    ctx = {
        "session_id": SESSION_ID,
        "messages": list(messages) if messages is not None else None,
        "citations": citations or [],
    }
    monkeypatch.setattr(qa_stream_routes, "prepare_context", lambda *args, **kwargs: ctx)
    return ctx


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestHttpStreamEvents:
    """HTTP Chunked（NDJSON）事件协议"""

    def test_event_order_and_persistence(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        fake_llm(tokens=["你", "好"])
        _patch_prepare_context(
            monkeypatch,
            citations=[{"source": "a.txt", "chunk_index": 0, "content": "片段"}],
        )

        resp = client.post("/api/qa/stream/http", json={
            "question": "测试问题", "kb_id": sample_kb["kb_id"],
        }, headers=auth_headers)
        assert resp.status_code == 200

        events = _parse_ndjson(resp.text)
        assert [e["event"] for e in events] == ["citations", "token", "token", "done"]
        assert events[0]["data"]["citations"][0]["source"] == "a.txt"
        assert events[-1]["data"]["session_id"] == SESSION_ID

        answer = "".join(e["data"]["content"] for e in events if e["event"] == "token")
        assert answer == "你好"

        # 完整回答落库（只有 assistant 一条，prepare_context 已被替换）
        stored = database.get_session_messages(SESSION_ID)
        assert [m["role"] for m in stored] == ["assistant"]
        assert stored[0]["content"] == "你好"

    def test_empty_recall_emits_fixed_prompt_then_done(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        """空召回不是错误：发固定提示 token，正常 done 并正常保存。"""
        fake_llm(tokens=["不应被调用"])
        _patch_prepare_context(monkeypatch, messages=None)

        resp = client.post("/api/qa/stream/http", json={
            "question": "空库问题", "kb_id": sample_kb["kb_id"],
        }, headers=auth_headers)

        events = _parse_ndjson(resp.text)
        types = [e["event"] for e in events]
        assert types == ["citations", "token", "done"]
        assert "未检索到相关内容" in events[1]["data"]["content"]

        stored = database.get_session_messages(SESSION_ID)
        assert stored[0]["content"] == "未检索到相关内容，请先上传文档到知识库。"

    def test_error_is_terminal_and_partial_answer_not_saved(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        """模型中途失败：已输出的 token 保留，但发 error 终结且不保存部分回答。"""
        fake_llm(tokens=["部分回答"], stream_error=RuntimeError("模型服务中断"))
        _patch_prepare_context(monkeypatch)

        resp = client.post("/api/qa/stream/http", json={
            "question": "会失败的问题", "kb_id": sample_kb["kb_id"],
        }, headers=auth_headers)

        events = _parse_ndjson(resp.text)
        types = [e["event"] for e in events]
        assert types == ["citations", "token", "error"]
        assert "done" not in types
        assert "模型服务中断" in events[-1]["data"]["message"]

        # 部分回答不能落库，否则用户会看到半截答案被当成完整回答
        assert database.get_session_messages(SESSION_ID) == []


class TestSseStream:
    """SSE 序列化"""

    def test_sse_serializes_events(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        fake_llm(tokens=["SSE"])
        _patch_prepare_context(monkeypatch)

        resp = client.post("/api/qa/stream/sse", json={
            "question": "SSE 问题", "kb_id": sample_kb["kb_id"],
        }, headers=auth_headers)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "event: citations" in resp.text
        assert "event: token" in resp.text
        assert "event: done" in resp.text


class TestWebSocketStream:
    """WebSocket 认证与事件流"""

    def test_unauthenticated_rejected(self, client):
        """未认证连接只能收到 error（不做检索）。"""
        with client.websocket_connect("/api/qa/stream/ws") as ws:
            ws.send_json({"question": "任何问题", "kb_id": "any-kb"})
            event = ws.receive_json()

        assert event["event"] == "error"
        assert "未认证" in event["data"]["message"]

    def test_api_key_auth_streams_tokens(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        """消息体 API Key 认证路径：citations → token → done。"""
        fake_llm(tokens=["R", "A", "G"])
        _patch_prepare_context(monkeypatch)

        with client.websocket_connect("/api/qa/stream/ws") as ws:
            ws.send_json({
                "question": "什么是 RAG",
                "kb_id": sample_kb["kb_id"],
                "token": auth_headers["X-API-Key"],
            })
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["event"] in ("done", "error"):
                    break

        types = [e["event"] for e in events]
        assert types[0] == "citations"
        assert types[-1] == "done"
        assert "".join(e["data"]["content"] for e in events if e["event"] == "token") == "RAG"


class TestCancellation:
    """取消语义：不保存迟到/部分回答"""

    def test_cancel_before_start_saves_nothing(self, client, fake_llm):
        from rag_stream import stream_tokens_generator

        fake_llm(tokens=["不应输出"])
        stop_event = threading.Event()
        stop_event.set()

        events = list(stream_tokens_generator(["msg"], SESSION_ID, [], stop_event=stop_event))

        assert events == []
        assert database.get_session_messages(SESSION_ID) == []

    def test_cancel_midway_stops_without_saving(self, client, fake_llm):
        from rag_stream import stream_tokens_generator

        fake_llm(tokens=["第一段", "第二段", "第三段"])
        stop_event = threading.Event()
        stream = stream_tokens_generator(["msg"], SESSION_ID, [], stop_event=stop_event)

        first = next(stream)
        assert first["event"] == "token"

        stop_event.set()
        assert list(stream) == []
        assert database.get_session_messages(SESSION_ID) == []


class TestPipelineParameterPriority:
    """参数优先级：显式入参 > pipeline_config 默认值"""

    def test_explicit_params_override_pipeline_config(self, monkeypatch, fake_bge):
        import milvus_ops
        import rag_pipeline

        monkeypatch.setattr(rag_pipeline, "_load_pipeline_config", lambda: {
            "enable_rerank": True,
            "enable_index": True,
            "dense_weight": 0.5,
            "sparse_weight": 0.5,
            "retrieval_top_k": 1,   # 配置只取 1 条，显式 top_k=3 应覆盖
            "candidate_k": 20,
        })
        fake_bge()
        hits = [
            {
                "pk": f"chunk-{i}", "source": "a.txt", "chunk_index": i,
                "text": f"内容{i}", "dense_score": 0.5 + i * 0.1, "sparse_score": 0.0,
            }
            for i in range(3)
        ]
        monkeypatch.setattr(milvus_ops, "search_vectors", lambda **kwargs: hits)

        def _unexpected_index_search(*args, **kwargs):
            raise AssertionError("enable_index=False 时不应检索索引问题")

        monkeypatch.setattr(milvus_ops, "search_index_questions", _unexpected_index_search)

        result = rag_pipeline.recall_test(
            kb_id="kb-1", query="问题", retrieval_mode="dense_only",
            top_k=3, enable_rerank=False, enable_index=False, debug=True,
        )

        assert len(result["items"]) == 3
        assert result["debug"]["pipeline"]["enable_index"] is False
        assert result["debug"]["pipeline"]["rerank_applied"] is False
        assert result["items"][0]["content"] == "内容2"   # dense 分最高者排首位
        assert result["items"][0]["rerank_score"] is None
