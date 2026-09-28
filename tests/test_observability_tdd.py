"""
全链路可观测性测试（Advanced RAG 迭代 · 第 4 期）。

测试范围：
  - 记录器：span 起止/耗时/状态、异常标记、降级可见、未开启追踪时零成本
  - 脱敏：API Key / 签名密钥 / Redis 密码 / Cookie 不落库
  - 容错：观测写库失败**不能影响主流程**
  - 问答链路：trace_id 回传、阶段齐全（接收→改写→向量化→召回→融合→Rerank→上下文→生成）
  - 流式：citations 事件带 trace_id、done 事件带首 Token 与总耗时
  - 文档链路：文件保存/解析/切分/Embedding/向量写入/数据保存/索引完成
  - 面板聚合、列表过滤、手动清理（dry-run + 管理员限定）
  - 健康检查：存活与就绪分离，且明确指出不可用的依赖
"""
import json

import pytest

import database
import observability


# ── 记录器与脱敏 ────────────────────────────────────────

class TestMasking:
    def test_masks_keys_and_passwords(self):
        # 一律使用明显的样例值（保留域名/IP 用文档保留段），避免把真实内网地址与口令写进仓库
        text = ('X-API-Key: ak_0123456789abcdef  authorization="Bearer sk-abcdefgh12345678" '
                'redis://:pwd-for-test@redis.example.internal:6379/0 AUTH_SECRET_KEY=deadbeefcafe1234 '
                'Cookie: rag_session=u_abc.123.sig')
        masked = observability.mask_secrets(text)
        assert "ak_0123456789abcdef" not in masked
        assert "sk-abcdefgh12345678" not in masked
        assert "pwd-for-test" not in masked
        assert "deadbeefcafe1234" not in masked
        assert "u_abc.123.sig" not in masked
        assert "***" in masked

    def test_sanitize_walks_nested_structures(self):
        payload = {"config": {"LLM_API_KEY": "ak_1234567890abcdef"}, "list": ["rag_session=secret", 42]}
        cleaned = observability.sanitize(payload)
        assert "ak_1234567890abcdef" not in json.dumps(cleaned, ensure_ascii=False)
        assert "secret" not in json.dumps(cleaned, ensure_ascii=False)
        assert cleaned["list"][1] == 42


class TestRecorder:
    def test_records_spans_and_finishes(self, client):
        recorder = observability.TraceRecorder.start("qa", kb_id="kb-x", question="问题")
        with recorder.span(observability.STAGE_RECEIVE, input_count=1) as span:
            span.set_output(1, transport="test")
        recorder.mark(observability.STAGE_FUSE, duration_ms=7, input_count=3, output_count=2)
        trace_id = recorder.finish(status=observability.STATUS_OK, first_token_ms=120)

        trace = observability.get_trace(trace_id)
        assert trace["status"] == "ok"
        assert trace["first_token_ms"] == 120
        assert trace["total_ms"] is not None
        assert [span["stage"] for span in trace["spans"]] == [observability.STAGE_RECEIVE,
                                                              observability.STAGE_FUSE]
        assert trace["spans"][0]["output_count"] == 1
        assert trace["spans"][1]["duration_ms"] == pytest.approx(7, abs=2)

    def test_exception_marks_span_failed_and_reraises(self, client):
        recorder = observability.TraceRecorder.start("qa", question="问题")
        with pytest.raises(RuntimeError):
            with recorder.span(observability.STAGE_EMBED):
                raise RuntimeError("BGE 超时")
        trace = observability.get_trace(recorder.finish(status=observability.STATUS_FAILED,
                                                        error="BGE 超时"))
        span = trace["spans"][0]
        assert span["status"] == observability.STATUS_FAILED
        assert "BGE 超时" in span["error"]
        assert trace["status"] == "failed"

    def test_degradation_is_visible(self, client):
        recorder = observability.TraceRecorder.start("qa", question="问题")
        recorder.degrade("问题改写", "模型服务 502", "使用原问题检索")
        trace = observability.get_trace(recorder.finish(status=observability.STATUS_OK))
        assert trace["degrade_count"] == 1
        assert trace["detail"]["degradations"][0]["fallback"] == "使用原问题检索"
        assert trace["spans"][0]["status"] == observability.STATUS_DEGRADED

    def test_result_fields_are_masked(self, client):
        recorder = observability.TraceRecorder.start("qa", question="带密钥的问题 ak_0123456789abcdef")
        recorder.set_result(rewritten_question="改写 ak_0123456789abcdef",
                            expansions=["扩展 ak_0123456789abcdef"], hyde_answer="假设")
        trace = observability.get_trace(recorder.finish(status=observability.STATUS_OK))
        dumped = json.dumps(trace, ensure_ascii=False)
        assert "ak_0123456789abcdef" not in dumped

    def test_observability_failure_does_not_break_caller(self, client, monkeypatch):
        """观测写库失败只告警：不能把异常抛给主流程。"""
        def _broken():
            raise RuntimeError("数据库不可用")

        monkeypatch.setattr(observability.database, "get_connection", _broken)
        recorder = observability.TraceRecorder.start("qa", question="问题")
        with recorder.span(observability.STAGE_RECEIVE):
            pass
        assert recorder.finish(status=observability.STATUS_OK)      # 不抛异常
        monkeypatch.undo()

    def test_span_without_active_trace_is_noop(self, client):
        with observability.span(observability.STAGE_EMBED) as span:
            span.set_output(1)
        assert observability.current() is None
        assert observability.list_traces()["total"] == 0

    def test_record_span_appends_to_existing_trace(self, client):
        recorder = observability.TraceRecorder.for_task("task-123", "upload", question="a.txt")
        observability.record_span("task-123", observability.STAGE_FILE_SAVE, duration_ms=5,
                                 output_count=100, filename="a.txt")
        recorder.mark(observability.STAGE_PARSE, duration_ms=10)
        trace = observability.get_trace(recorder.finish(status=observability.STATUS_OK))
        stages = [span["stage"] for span in trace["spans"]]
        assert stages == [observability.STAGE_FILE_SAVE, observability.STAGE_PARSE]


# ── 问答链路（端到端）───────────────────────────────────

def _kb_with_chunks(client, headers, fake_bge, name, strategy=None):
    kb = client.post("/api/kb", json={"name": name, "chunk_size": 120, "chunk_overlap": 0},
                     headers=headers).json()["data"]
    if strategy:
        client.put(f"/api/kb/{kb['kb_id']}/retrieval-strategy", json={"strategy": strategy},
                   headers=headers)
    fake_bge()
    upload = client.post("/api/documents/upload", data={"kb_id": kb["kb_id"]},
                         files={"file": ("obs.txt", ("可观测性验证文档，包含若干句子。" * 20).encode(),
                                         "text/plain")},
                         headers=headers)
    import tasks
    tasks.process_upload_task(upload.json()["data"]["task_id"])
    return kb, upload.json()["data"]["doc_id"]


def _mock_search(monkeypatch, chunk_ids):
    import milvus_ops
    hits = [{"pk": cid, "source": "obs.txt", "chunk_index": index, "text": f"内容{cid}",
             "dense_score": 0.9 - index * 0.05, "sparse_score": 0.0}
            for index, cid in enumerate(chunk_ids)]
    monkeypatch.setattr(milvus_ops, "search_vectors", lambda **kwargs: hits)


class TestQaTrace:
    def test_qa_returns_trace_id_and_records_stages(self, client, auth_headers, fake_bge, fake_llm, monkeypatch):
        kb, doc_id = _kb_with_chunks(client, auth_headers, fake_bge, "观测问答库")
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]
        _mock_search(monkeypatch, children[:2])
        fake_llm(answer="最终回答")

        resp = client.post("/api/qa/query", json={"question": "这个文档讲了什么", "kb_id": kb["kb_id"]},
                           headers=auth_headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        trace_id = data["trace_id"]
        assert trace_id and data["answer"] == "最终回答"

        trace = client.get(f"/api/observability/traces/{trace_id}", headers=auth_headers).json()["data"]
        stages = [span["stage"] for span in trace["spans"]]
        assert observability.STAGE_RECEIVE in stages
        assert observability.STAGE_EMBED in stages
        assert observability.STAGE_RETRIEVE in stages
        assert observability.STAGE_FUSE in stages
        assert observability.STAGE_CONTEXT in stages
        assert trace["status"] == "ok"
        assert trace["config_snapshot"]["mode"] == "basic"
        assert trace["detail"]["top_items"]                        # 最终上下文摘要
        for span in trace["spans"]:
            assert span["duration_ms"] >= 0 and span["status"] in ("ok", "degraded", "failed")

    def test_advanced_trace_exposes_rewrite_expansions_and_hyde(self, client, auth_headers, fake_bge,
                                                               fake_llm, monkeypatch):
        kb, doc_id = _kb_with_chunks(client, auth_headers, fake_bge, "观测高级库", strategy={
            "mode": "advanced", "enable_rewrite": True, "enable_multi_query": True,
            "multi_query_count": 2, "enable_hyde": True,
        })
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]
        fake_llm(answer="最终回答", answers=[
            ("改写后的问题", "独立化后的问题"),
            ("检索问题", "扩展问题一：住宿费报销上限是多少\n扩展问题二：差旅住宿标准有哪些"),
            ("假设性回答片段", "这是假设答案"),
        ])
        calls = []
        import milvus_ops

        def _fake(**kwargs):
            calls.append(kwargs)
            return [{"pk": children[0], "source": "obs.txt", "chunk_index": 0,
                     "text": "内容", "dense_score": 0.9, "sparse_score": 0.0}]
        monkeypatch.setattr(milvus_ops, "search_vectors", _fake)

        # 先写入历史，便于问题改写发挥作用
        first = client.post("/api/qa/query", json={"question": "文档讲了什么", "kb_id": kb["kb_id"]},
                            headers=auth_headers)
        session_id = first.json()["data"]["session_id"]
        resp = client.post("/api/qa/query", json={
            "question": "那它还有什么特点", "kb_id": kb["kb_id"], "session_id": session_id,
        }, headers=auth_headers)
        trace_id = resp.json()["data"]["trace_id"]

        trace = client.get(f"/api/observability/traces/{trace_id}", headers=auth_headers).json()["data"]
        assert trace["rewritten_question"] == "独立化后的问题"
        assert trace["expansions"] == ["扩展问题一：住宿费报销上限是多少", "扩展问题二：差旅住宿标准有哪些"]
        assert trace["hyde_answer"] == "这是假设答案"
        assert trace["config_snapshot"]["mode"] == "advanced"
        assert trace["config_snapshot"]["fusion_method"] == "rrf"
        # HyDE 只用于检索：最终回答仍是 LLM 针对参考内容的输出
        assert resp.json()["data"]["answer"] == "最终回答"
        assert "这是假设答案" not in resp.json()["data"]["answer"]
        # 多路召回：原问题/改写/两条扩展/HyDE 各检索一次
        assert len(calls) >= 5

    def test_advanced_degradation_visible_in_trace(self, client, auth_headers, fake_bge, fake_llm, monkeypatch):
        kb, doc_id = _kb_with_chunks(client, auth_headers, fake_bge, "观测降级库", strategy={
            "mode": "advanced", "enable_rewrite": True, "enable_multi_query": True, "enable_hyde": True,
        })
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]
        _mock_search(monkeypatch, children[:1])
        fake_llm(invoke_error=RuntimeError("模型服务 502"))

        resp = client.post("/api/qa/query", json={"question": "问题", "kb_id": kb["kb_id"]},
                           headers=auth_headers)
        assert resp.status_code == 200          # 降级后仍能返回结果
        trace_id = resp.json()["data"]["trace_id"]
        trace = client.get(f"/api/observability/traces/{trace_id}", headers=auth_headers).json()["data"]
        stages = {item["stage"] for item in trace["detail"]["degradations"]}
        # 无对话历史时"改写"是正常跳过（不算降级），因此这里只断言真的失败的两项
        assert {"multi_query", "hyde"} <= stages
        assert trace["degrade_count"] >= 2
        degraded_spans = [span for span in trace["spans"] if span["status"] == "degraded"]
        assert degraded_spans

    def test_http_stream_trace_and_time_fields(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        import routes.qa_stream_routes as qa_stream_routes

        fake_llm(tokens=["你", "好"])
        monkeypatch.setattr(qa_stream_routes, "prepare_context", lambda *a, **k: {
            "session_id": "sess-obs", "messages": ["m"], "citations": [],
            "strategy_snapshot": {"mode": "basic"}, "query_plan": None, "degradations": [],
        })
        resp = client.post("/api/qa/stream/http",
                           json={"question": "流式问题", "kb_id": sample_kb["kb_id"]},
                           headers=auth_headers)
        events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
        citations_evt = events[0]
        done_evt = events[-1]
        assert citations_evt["event"] == "citations"
        trace_id = citations_evt["data"]["trace_id"]
        assert trace_id
        assert done_evt["data"]["trace_id"] == trace_id
        assert done_evt["data"]["first_token_ms"] is not None
        assert done_evt["data"]["total_ms"] >= done_evt["data"]["first_token_ms"]

        trace = client.get(f"/api/observability/traces/{trace_id}", headers=auth_headers).json()["data"]
        assert trace["status"] == "ok"
        assert trace["first_token_ms"] is not None
        stages = [span["stage"] for span in trace["spans"]]
        assert observability.STAGE_GENERATE in stages
        assert observability.STAGE_STREAM_DONE in stages


# ── 文档链路 ────────────────────────────────────────────

class TestUploadTrace:
    def test_upload_task_trace_has_all_stages(self, client, auth_headers, fake_bge):
        kb, doc_id = _kb_with_chunks(client, auth_headers, fake_bge, "观测文档链路库")
        task = database.get_tasks_by_doc_id(doc_id)[0]

        resp = client.get(f"/api/observability/tasks/{task['task_id']}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        stages = [span["stage"] for span in data["spans"]]
        for expected in (observability.STAGE_FILE_SAVE, observability.STAGE_PARSE,
                         observability.STAGE_SPLIT, observability.STAGE_EMBED_DOC,
                         observability.STAGE_VECTOR_WRITE, observability.STAGE_CHUNK_SAVE,
                         observability.STAGE_INDEX_DONE):
            assert expected in stages, f"缺少阶段 {expected}"
        assert data["task"]["current_stage"] == "completed"
        assert data["trace"]["status"] == "ok"

    def test_task_trace_404(self, client, auth_headers):
        resp = client.get("/api/observability/tasks/no-such-task", headers=auth_headers)
        assert resp.status_code == 404


# ── 面板 / 列表 / 清理 ──────────────────────────────────

class TestOverviewAndCleanup:
    def test_overview_aggregates(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        import routes.qa_stream_routes as qa_stream_routes

        fake_llm(tokens=["答"])
        monkeypatch.setattr(qa_stream_routes, "prepare_context", lambda *a, **k: {
            "session_id": "sess-agg", "messages": ["m"], "citations": [],
            "strategy_snapshot": {"mode": "basic"}, "query_plan": None, "degradations": [],
        })
        client.post("/api/qa/stream/http", json={"question": "问题", "kb_id": sample_kb["kb_id"]},
                    headers=auth_headers)

        resp = client.get("/api/observability/overview?with_dependencies=false", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["qa"]["requests"] >= 1
        assert data["qa"]["success"] >= 1
        assert data["qa"]["success_rate"] > 0
        assert "avg_first_token_ms" in data["qa"]
        assert "no_recall_ratio" in data["qa"]
        assert set(data["features"]) == {"rewrite", "multi_query", "hyde"}
        assert "rerank" in data and "document_tasks" in data
        assert data["storage"]["traces"] >= 1

    def test_traces_list_filters_by_biz_type(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        import routes.qa_stream_routes as qa_stream_routes

        fake_llm(tokens=["答"])
        monkeypatch.setattr(qa_stream_routes, "prepare_context", lambda *a, **k: {
            "session_id": "sess-list", "messages": ["m"], "citations": [],
            "strategy_snapshot": {}, "query_plan": None, "degradations": [],
        })
        client.post("/api/qa/stream/http", json={"question": "问题", "kb_id": sample_kb["kb_id"]},
                    headers=auth_headers)

        all_traces = client.get("/api/observability/traces", headers=auth_headers).json()["data"]
        assert all_traces["total"] >= 1
        qa_only = client.get("/api/observability/traces?biz_type=qa", headers=auth_headers).json()["data"]
        assert qa_only["total"] >= 1
        none = client.get("/api/observability/traces?biz_type=index_build", headers=auth_headers).json()["data"]
        assert none["total"] == 0

    def test_cleanup_dry_run_then_delete(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        import routes.qa_stream_routes as qa_stream_routes

        fake_llm(tokens=["答"])
        monkeypatch.setattr(qa_stream_routes, "prepare_context", lambda *a, **k: {
            "session_id": "sess-clean", "messages": ["m"], "citations": [],
            "strategy_snapshot": {}, "query_plan": None, "degradations": [],
        })
        client.post("/api/qa/stream/http", json={"question": "问题", "kb_id": sample_kb["kb_id"]},
                    headers=auth_headers)

        before = observability.list_traces()["total"]
        assert before >= 1

        dry = client.post("/api/observability/traces/cleanup",
                          json={"mode": "all", "dry_run": True}, headers=auth_headers).json()["data"]
        assert dry["matched_traces"] == before
        assert dry["deleted_traces"] == 0
        assert observability.list_traces()["total"] == before       # dry-run 未删除

        real = client.post("/api/observability/traces/cleanup",
                           json={"mode": "all", "dry_run": False}, headers=auth_headers).json()["data"]
        assert real["deleted_traces"] == before
        assert observability.list_traces()["total"] == 0
        # 清理只删观测数据，不影响知识库
        assert database.get_kb(sample_kb["kb_id"]) is not None

    def test_cleanup_requires_admin(self, client, sample_kb):
        registered = client.post("/api/auth/register", json={"username": "obs_user", "password": "test1234"})
        headers = {"X-API-Key": registered.json()["data"]["api_key"]}
        resp = client.post("/api/observability/traces/cleanup",
                           json={"mode": "all", "dry_run": True}, headers=headers)
        assert resp.status_code == 403

    def test_non_admin_only_sees_own_kb_traces(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        import routes.qa_stream_routes as qa_stream_routes

        fake_llm(tokens=["答"])
        monkeypatch.setattr(qa_stream_routes, "prepare_context", lambda *a, **k: {
            "session_id": "sess-scope", "messages": ["m"], "citations": [],
            "strategy_snapshot": {}, "query_plan": None, "degradations": [],
        })
        client.post("/api/qa/stream/http", json={"question": "问题", "kb_id": sample_kb["kb_id"]},
                    headers=auth_headers)

        registered = client.post("/api/auth/register", json={"username": "obs_scope", "password": "test1234"})
        user_headers = {"X-API-Key": registered.json()["data"]["api_key"]}
        scoped = client.get("/api/observability/overview?with_dependencies=false",
                            headers=user_headers).json()["data"]
        assert scoped["qa"]["requests"] == 0
        assert scoped["scope"]["note"]

    # ── 数据维护：按范围清理 / 导出 ──────────────────────

    def _make_traces(self, client, auth_headers, sample_kb, fake_llm, monkeypatch, count=2):
        """造 count 条问答 trace，返回 trace_id 列表（新的在前）。"""
        import routes.qa_stream_routes as qa_stream_routes

        fake_llm(tokens=["答"])
        for index in range(count):
            monkeypatch.setattr(qa_stream_routes, "prepare_context", lambda *a, **k: {
                "session_id": "sess-maint-" + str(index), "messages": ["m"], "citations": [],
                "strategy_snapshot": {}, "query_plan": None, "degradations": [],
            })
            client.post("/api/qa/stream/http",
                        json={"question": "维护测试问题 " + str(index), "kb_id": sample_kb["kb_id"]},
                        headers=auth_headers)
        items = client.get("/api/observability/traces?limit=10", headers=auth_headers).json()["data"]["items"]
        return [item["trace_id"] for item in items]

    def test_cleanup_by_trace_ids_only_removes_those(self, client, auth_headers, sample_kb,
                                                     fake_llm, monkeypatch):
        """按 trace_id 精确清理：只删指定那条，其余保留（对应"排查完某次异常再删掉它"）。"""
        trace_ids = self._make_traces(client, auth_headers, sample_kb, fake_llm, monkeypatch, count=2)
        assert len(trace_ids) >= 2
        target, keep = trace_ids[0], trace_ids[1]

        dry = client.post("/api/observability/traces/cleanup",
                          json={"mode": "trace_ids", "trace_ids": [target], "dry_run": True},
                          headers=auth_headers).json()["data"]
        assert dry["matched_traces"] == 1
        assert dry["deleted_traces"] == 0

        real = client.post("/api/observability/traces/cleanup",
                           json={"mode": "trace_ids", "trace_ids": [target], "dry_run": False},
                           headers=auth_headers).json()["data"]
        assert real["deleted_traces"] == 1
        assert observability.get_trace(target) is None
        assert observability.get_trace(keep) is not None            # 未被误删

    def test_cleanup_older_than_keeps_recent(self, client, auth_headers, sample_kb, fake_llm, monkeypatch):
        """保留最近 N 天：刚写入的数据必须留着。"""
        self._make_traces(client, auth_headers, sample_kb, fake_llm, monkeypatch, count=1)
        before = observability.list_traces()["total"]
        assert before >= 1

        result = client.post("/api/observability/traces/cleanup",
                             json={"mode": "older_than", "days": 7, "dry_run": False},
                             headers=auth_headers).json()["data"]
        assert result["deleted_traces"] == 0
        assert observability.list_traces()["total"] == before

    def test_cleanup_trace_ids_requires_ids(self, client, auth_headers):
        resp = client.post("/api/observability/traces/cleanup",
                           json={"mode": "trace_ids", "trace_ids": [], "dry_run": True},
                           headers=auth_headers)
        assert resp.status_code == 400
        assert "trace" in resp.json()["detail"]["message"].lower()

    def test_cleanup_invalid_mode_returns_400(self, client, auth_headers):
        resp = client.post("/api/observability/traces/cleanup",
                           json={"mode": "everything", "dry_run": True}, headers=auth_headers)
        assert resp.status_code == 400
        assert "mode 仅支持" in resp.json()["detail"]["message"]

    def test_export_returns_traces_with_spans(self, client, auth_headers, sample_kb,
                                              fake_llm, monkeypatch):
        """导出用于"先备份再清理"：含 trace 与其 spans，带下载文件名。"""
        trace_ids = self._make_traces(client, auth_headers, sample_kb, fake_llm, monkeypatch, count=1)
        assert trace_ids

        resp = client.get("/api/observability/traces/export?limit=100", headers=auth_headers)
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        data = resp.json()["data"]
        assert data["traces"] and data["spans"]
        assert data["total_traces_in_store"] >= len(data["traces"])
        assert "truncated" in data
        # 导出内容可直接反序列化使用（detail/config_snapshot 已解析成对象）
        assert isinstance(data["traces"][0]["config_snapshot"], dict)

    def test_export_requires_admin(self, client):
        registered = client.post("/api/auth/register", json={"username": "obs_export", "password": "test1234"})
        headers = {"X-API-Key": registered.json()["data"]["api_key"]}
        resp = client.get("/api/observability/traces/export", headers=headers)
        assert resp.status_code == 403

    def test_list_traces_supports_time_window_and_paging(self, client, auth_headers, sample_kb,
                                                        fake_llm, monkeypatch):
        """列表支持时间快捷范围 + 分页（前端"像浏览器历史那样筛选/翻页"依赖它）。"""
        self._make_traces(client, auth_headers, sample_kb, fake_llm, monkeypatch, count=3)

        page1 = client.get("/api/observability/traces?limit=2&offset=0", headers=auth_headers).json()["data"]
        page2 = client.get("/api/observability/traces?limit=2&offset=2", headers=auth_headers).json()["data"]
        assert page1["total"] == page2["total"] >= 3
        assert len(page1["items"]) == 2
        # 分页不重叠
        assert not ({i["trace_id"] for i in page1["items"]} & {i["trace_id"] for i in page2["items"]})

        # 时间范围："最近 1 小时"应包含刚写入的记录；换个不存在的类型则为 0
        recent = client.get("/api/observability/traces?hours=1&limit=100", headers=auth_headers).json()["data"]
        assert recent["total"] >= 3
        none = client.get("/api/observability/traces?hours=1&biz_type=index_build",
                          headers=auth_headers).json()["data"]
        assert none["total"] == 0

    def test_cleanup_range_deletes_recent_window(self, client, auth_headers, sample_kb,
                                                 fake_llm, monkeypatch):
        """按时间范围删除（最近 N 小时）——对应"删掉最近一小时的记录"。"""
        self._make_traces(client, auth_headers, sample_kb, fake_llm, monkeypatch, count=2)
        before = observability.list_traces()["total"]

        dry = client.post("/api/observability/traces/cleanup",
                          json={"mode": "range", "hours": 1, "dry_run": True},
                          headers=auth_headers).json()["data"]
        assert dry["matched_traces"] == before
        assert observability.list_traces()["total"] == before        # dry-run 未删

        real = client.post("/api/observability/traces/cleanup",
                           json={"mode": "range", "hours": 1, "dry_run": False},
                           headers=auth_headers).json()["data"]
        assert real["deleted_traces"] == before
        assert real["remaining_traces"] == 0

    def test_cleanup_range_requires_hours(self, client, auth_headers):
        """range 模式必须给 hours，否则 400（避免"范围删除"被误当全清）。"""
        resp = client.post("/api/observability/traces/cleanup",
                           json={"mode": "range", "dry_run": True}, headers=auth_headers)
        assert resp.status_code == 400
        assert "hours" in resp.json()["detail"]["message"]

    def test_cleanup_reports_remaining_counts(self, client, auth_headers, sample_kb,
                                              fake_llm, monkeypatch):
        """清理后必须回报"剩余多少"——否则用户无法确认到底删掉没有（前端据此显示）。"""
        trace_ids = self._make_traces(client, auth_headers, sample_kb, fake_llm, monkeypatch, count=2)
        assert len(trace_ids) >= 2

        result = client.post("/api/observability/traces/cleanup",
                             json={"mode": "trace_ids", "trace_ids": [trace_ids[0]], "dry_run": False},
                             headers=auth_headers).json()["data"]
        assert result["deleted_traces"] == 1
        assert result["remaining_traces"] == observability.list_traces()["total"]
        assert result["remaining_traces"] >= 1

    def test_overview_storage_has_time_bounds(self, client, auth_headers, sample_kb,
                                              fake_llm, monkeypatch):
        """面板要能看到占用条数与时间范围，否则用户不知道要不要清理。"""
        self._make_traces(client, auth_headers, sample_kb, fake_llm, monkeypatch, count=1)
        storage = client.get("/api/observability/overview?with_dependencies=false",
                             headers=auth_headers).json()["data"]["storage"]
        assert storage["traces"] >= 1
        assert storage["spans"] >= 1
        assert storage["earliest"] and storage["latest"]


# ── 健康检查 ────────────────────────────────────────────

class TestHealth:
    def test_live_always_ok(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_ready_reports_each_dependency(self, client):
        """离线环境下：SQLite 正常，其余外部依赖被明确指出来（而不是笼统的系统异常）。"""
        resp = client.get("/health/ready")
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert body["dependencies"]["sqlite"]["status"] == "ok"
        assert isinstance(body["unhealthy"], list)
        for name in ("redis", "milvus", "bge_embedding", "llm"):
            item = body["dependencies"][name]
            assert item["status"] in ("ok", "failed")
            if item["status"] == "failed":
                assert item["error"]                       # 必须给出具体原因
        if body["unhealthy"]:
            assert resp.status_code == 503
            assert set(body["unhealthy"]) <= set(body["dependencies"])
