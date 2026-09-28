"""
Advanced RAG 与检索策略测试（Advanced RAG 迭代 · 第 3 期）。

测试范围：
  - 策略层：默认值/字段校验/优先级（请求 > 知识库 > 全局 > 内置）/指纹
  - 策略接口：查看、修改、恢复默认、非法参数 400、知识库之间互不影响
  - 前置能力：问题改写 / 多查询扩展 / HyDE —— 各自的正常路径与**降级路径**
  - 融合：RRF、路径内 min-max 加权、去重、每路配额与总候选上限、hit_paths 记录
  - 端到端：高级模式下多路召回 → 融合 → Rerank → 父子块展开；基础模式行为不变
  - 上下文预算：max_context_chars 控制最终上下文长度
"""
import pytest

import advanced_rag
import chunking
import database
import fusion
import rag_pipeline
import strategy as strategy_module

# ── 策略层（纯函数） ────────────────────────────────────

class TestStrategyConfig:
    def test_default_strategy_is_valid(self):
        config = strategy_module.validate_strategy(None)
        assert config["mode"] == "basic"
        assert config["fusion_method"] == "rrf"
        assert config["rerank_top_n"] == 5

    @pytest.mark.parametrize("patch, keyword", [
        ({"mode": "turbo"}, "检索模式仅支持"),
        ({"fusion_method": "magic"}, "融合算法仅支持"),
        ({"multi_query_count": 0}, "multi_query_count"),
        ({"multi_query_count": 99}, "multi_query_count"),
        ({"dense_weight": 1.5}, "dense_weight"),
        ({"dense_weight": 0, "sparse_weight": 0}, "不能同时为 0"),
        ({"rerank_top_n": 50, "total_candidate_cap": 10}, "不能大于 total_candidate_cap"),
        ({"per_path_k": 100, "total_candidate_cap": 5}, "per_path_k 过大"),
    ])
    def test_invalid_strategy_rejected_with_reason(self, patch, keyword):
        with pytest.raises(strategy_module.StrategyConfigError) as exc:
            strategy_module.validate_strategy(patch)
        assert keyword in str(exc.value)

    def test_priority_request_over_kb_over_global(self):
        """优先级：请求临时覆盖 > 知识库策略 > 全局 pipeline_config > 内置默认。"""
        kb_strategy = {"fusion_method": "weighted", "rerank_top_n": 3, "enable_hyde": True}
        globals_ = {"enable_rerank": False, "retrieval_top_k": 8, "dense_weight": 0.7}

        resolved = strategy_module.resolve(kb_strategy, None, globals_)
        assert resolved["fusion_method"] == "weighted"   # 知识库覆盖全局
        assert resolved["rerank_top_n"] == 3
        assert resolved["enable_rerank"] is False        # 知识库没设 → 用全局
        assert resolved["dense_weight"] == 0.7

        overridden = strategy_module.resolve(kb_strategy, {"fusion_method": "rrf", "rerank_top_n": 9}, globals_)
        assert overridden["fusion_method"] == "rrf"      # 请求覆盖知识库
        assert overridden["rerank_top_n"] == 9

    def test_global_retrieval_top_k_maps_to_rerank_top_n(self):
        resolved = strategy_module.resolve(None, None, {"retrieval_top_k": 4})
        assert resolved["rerank_top_n"] == 4

    def test_strategy_hash_stable_and_sensitive(self):
        a = strategy_module.validate_strategy({"fusion_method": "rrf"})
        b = strategy_module.validate_strategy({"fusion_method": "rrf"})
        c = strategy_module.validate_strategy({"fusion_method": "weighted"})
        assert strategy_module.strategy_hash(a) == strategy_module.strategy_hash(b)
        assert strategy_module.strategy_hash(a) != strategy_module.strategy_hash(c)


class TestStrategyAPI:
    def test_get_default_strategy(self, client, auth_headers, sample_kb):
        resp = client.get(f"/api/kb/{sample_kb['kb_id']}/retrieval-strategy", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["customized"] is False
        assert data["strategy"]["mode"] == "basic"
        assert "rrf" in data["supported"]["fusion_methods"]

    def test_update_get_and_reset(self, client, auth_headers, sample_kb):
        kb_id = sample_kb["kb_id"]
        resp = client.put(f"/api/kb/{kb_id}/retrieval-strategy", json={"strategy": {
            "mode": "advanced", "enable_multi_query": True, "multi_query_count": 4,
            "fusion_method": "weighted", "rerank_top_n": 4,
        }}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["data"]["strategy"]["fusion_method"] == "weighted"

        fetched = client.get(f"/api/kb/{kb_id}/retrieval-strategy", headers=auth_headers)
        data = fetched.json()["data"]
        assert data["customized"] is True
        assert data["strategy"]["multi_query_count"] == 4

        reset = client.post(f"/api/kb/{kb_id}/retrieval-strategy/reset", headers=auth_headers)
        assert reset.status_code == 200
        assert reset.json()["data"]["strategy"]["mode"] == "basic"
        again = client.get(f"/api/kb/{kb_id}/retrieval-strategy", headers=auth_headers)
        assert again.json()["data"]["customized"] is False

    def test_invalid_strategy_returns_400(self, client, auth_headers, sample_kb):
        resp = client.put(f"/api/kb/{sample_kb['kb_id']}/retrieval-strategy",
                          json={"strategy": {"mode": "turbo"}}, headers=auth_headers)
        assert resp.status_code == 400
        assert "检索模式仅支持" in resp.json()["detail"]["message"]

    def test_strategy_isolated_between_kbs(self, client, auth_headers):
        first = client.post("/api/kb", json={"name": "策略库A"}, headers=auth_headers).json()["data"]
        second = client.post("/api/kb", json={"name": "策略库B"}, headers=auth_headers).json()["data"]
        client.put(f"/api/kb/{first['kb_id']}/retrieval-strategy",
                   json={"strategy": {"mode": "advanced"}}, headers=auth_headers)

        assert client.get(f"/api/kb/{first['kb_id']}/retrieval-strategy",
                          headers=auth_headers).json()["data"]["strategy"]["mode"] == "advanced"
        assert client.get(f"/api/kb/{second['kb_id']}/retrieval-strategy",
                          headers=auth_headers).json()["data"]["strategy"]["mode"] == "basic"


# ── 前置能力（改写 / 多查询 / HyDE） ────────────────────

HISTORY = [
    {"role": "user", "content": "公司的差旅报销标准是什么？"},
    {"role": "assistant", "content": "住宿费与交通费按职级限额报销。"},
]


class TestQueryRewrite:
    def test_rewrite_uses_history(self, fake_llm):
        fake_llm(answers=[("改写后的问题", "公司的差旅报销标准中，住宿费的报销上限是多少？")])
        rewritten, error = advanced_rag.rewrite_question("那住宿费呢？", HISTORY)
        assert error is None
        assert rewritten == "公司的差旅报销标准中，住宿费的报销上限是多少？"

    def test_no_history_skips_llm(self, fake_llm):
        llm = fake_llm(answer="不该被调用")
        rewritten, error = advanced_rag.rewrite_question("独立问题", [])
        assert (rewritten, error) == ("", None)
        assert llm.invoke_calls == []          # 无历史时直接使用原问题，不消耗调用

    def test_rewrite_failure_degrades_to_original(self, fake_llm):
        fake_llm(invoke_error=RuntimeError("模型不可用"))
        rewritten, error = advanced_rag.rewrite_question("那住宿费呢？", HISTORY)
        assert rewritten == ""
        assert "模型不可用" in error

    def test_rewrite_strips_numbering_and_quotes(self, fake_llm):
        fake_llm(answers=[("改写后的问题", '1. "差旅报销标准中住宿费的上限是多少？"')])
        rewritten, _ = advanced_rag.rewrite_question("那住宿费呢？", HISTORY)
        assert rewritten == "差旅报销标准中住宿费的上限是多少？"


class TestMultiQueryExpansion:
    def test_expansion_parses_numbered_lines_and_caps_count(self, fake_llm):
        fake_llm(answers=[("检索问题", "\n".join([
            "1. 差旅住宿费的报销上限是多少",
            "2. 住宿费报销标准",
            "3. 差旅标准 住宿 限额",
            "4. 逗号，",                 # 过短，应被过滤
            "5. 差旅住宿费的报销上限是多少",  # 重复，应被去重
        ]))])
        queries, error = advanced_rag.expand_queries("住宿费报销多少", 3)
        assert error is None
        assert queries == ["差旅住宿费的报销上限是多少", "住宿费报销标准", "差旅标准 住宿 限额"]

    def test_expansion_excludes_original_question(self, fake_llm):
        fake_llm(answers=[("检索问题", "住宿费报销多少\n差旅住宿限额")])
        queries, _ = advanced_rag.expand_queries("住宿费报销多少", 3)
        assert "住宿费报销多少" not in queries
        assert "差旅住宿限额" in queries

    def test_expansion_failure_degrades(self, fake_llm):
        fake_llm(invoke_error=RuntimeError("超时"))
        queries, error = advanced_rag.expand_queries("问题", 3)
        assert queries == [] and "超时" in error


class TestHyde:
    def test_hyde_is_truncated_at_sentence_boundary(self, fake_llm):
        long_answer = "假设答案第一句。" * 40
        fake_llm(answers=[("假设性回答片段", long_answer)])
        answer, error = advanced_rag.generate_hypothetical_answer("问题", max_chars=60)
        assert error is None
        assert len(answer) <= 60
        assert answer.endswith("。")           # 在句末截断，不截半句

    def test_hyde_failure_skips_path(self, fake_llm):
        fake_llm(invoke_error=RuntimeError("502"))
        answer, error = advanced_rag.generate_hypothetical_answer("问题", 100)
        assert answer == "" and "502" in error


class TestBuildPlan:
    def test_all_switches_off_means_original_only(self, fake_llm):
        llm = fake_llm(answer="不该被调用")
        plan = advanced_rag.build_plan("问题", history=HISTORY, config={"mode": "advanced"})
        assert plan.search_queries() == [("original", "问题")]
        assert llm.invoke_calls == []

    def test_plan_collects_all_paths(self, fake_llm):
        fake_llm(answers=[
            ("改写后的问题", "改写后的独立问题"),
            ("检索问题", "扩展问题一\n扩展问题二"),
            ("假设性回答片段", "假设答案内容"),
        ])
        plan = advanced_rag.build_plan("那住宿费呢", history=HISTORY, config={
            "mode": "advanced", "enable_rewrite": True,
            "enable_multi_query": True, "multi_query_count": 2, "enable_hyde": True,
        })
        paths = [path for path, _ in plan.search_queries()]
        assert paths == ["original", "rewritten", "multi_query_1", "multi_query_2", "hyde"]
        assert plan.rewritten == "改写后的独立问题"
        assert plan.hyde_answer == "假设答案内容"
        assert plan.degradations == []

    def test_plan_records_degradations(self, fake_llm):
        fake_llm(invoke_error=RuntimeError("全部失败"))
        plan = advanced_rag.build_plan("问题", history=HISTORY, config={
            "mode": "advanced", "enable_rewrite": True,
            "enable_multi_query": True, "enable_hyde": True,
        })
        stages = {item["stage"] for item in plan.degradations}
        assert stages == {"rewrite", "multi_query", "hyde"}
        assert all(item["fallback"] for item in plan.degradations)
        assert plan.search_queries() == [("original", "问题")]   # 仍能用原问题检索


# ── 融合 ────────────────────────────────────────────────

def _hits(chunk_ids, base=1.0, step=0.1):
    """融合层输入（已在本路径内排好序，带 score）。"""
    return [{"chunk_id": cid, "source": "a.txt", "chunk_index": i,
             "content": f"内容{cid}", "score": base - i * step}
            for i, cid in enumerate(chunk_ids)]


def _search_hits(chunk_ids, base=0.9, step=0.05):
    """Milvus search_vectors 的返回结构（dense/sparse 原始分）。"""
    return [{"pk": cid, "source": "a.txt", "chunk_index": i, "text": f"内容{cid}",
             "dense_score": base - i * step, "sparse_score": 0.0}
            for i, cid in enumerate(chunk_ids)]


class TestFusion:
    def test_rrf_prefers_hits_from_multiple_paths(self):
        nodes, debug = fusion.fuse({
            "original": _hits(["a0", "a1", "a2"]),
            "hyde": _hits(["a1", "b1"]),
        }, method="rrf")
        assert nodes[0]["chunk_id"] == "a1"                    # 两条路径都命中 → 排第一
        assert nodes[0]["hit_paths"] == ["original", "hyde"]
        assert debug["dedup_removed"] == 1                     # a1 的重复命中被合并
        assert debug["path_stats"]["hyde"]["hits"] == 2

    def test_weighted_normalizes_within_path(self):
        nodes, _ = fusion.fuse({
            "p1": _hits(["x0", "x1"], base=0.9, step=0.1),     # 0.9 / 0.8
            "p2": _hits(["y0", "y1"], base=0.3, step=0.2),     # 0.3 / 0.1
        }, method="weighted")
        # 两条路径各自的最高分都归一化为 1.0，不会被"原始量纲"直接比较
        top = {node["chunk_id"]: node["fused_score"] for node in nodes}
        assert top["x0"] == pytest.approx(1.0)
        assert top["y0"] == pytest.approx(1.0)
        assert top["x1"] == pytest.approx(0.0)
        assert top["y1"] == pytest.approx(0.0)

    def test_equal_scores_treated_as_full_match(self):
        assert fusion._min_max([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]

    def test_per_path_quota_and_total_cap(self):
        nodes, debug = fusion.fuse({
            "p1": _hits([f"p1_{i}" for i in range(10)]),
            "p2": _hits([f"p2_{i}" for i in range(10)]),
        }, method="rrf", per_path_k=2, total_cap=3)
        assert len(nodes) == 3
        assert debug["path_stats"]["p1"]["used"] == 2
        assert debug["path_stats"]["p2"]["used"] == 2
        assert debug["kept_candidates"] == 3

    def test_empty_path_is_skipped(self):
        nodes, debug = fusion.fuse({"original": _hits(["a0"]), "hyde": []}, method="rrf")
        assert [node["chunk_id"] for node in nodes] == ["a0"]
        assert debug["path_stats"]["hyde"]["used"] == 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            fusion.fuse({"p": _hits(["a"])}, method="unknown")


# ── 端到端：高级模式召回 ────────────────────────────────

def _create_kb(client, headers, name, **extra):
    payload = {"name": name, "chunk_size": 120, "chunk_overlap": 0}
    payload.update(extra)
    resp = client.post("/api/kb", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]


def _upload_and_index(client, headers, kb_id, fake_bge, filename="adv.txt"):
    fake_bge()
    resp = client.post("/api/documents/upload", data={"kb_id": kb_id},
                       files={"file": (filename, ("容器化部署由 Web、Worker 和 Redis 三个容器协同完成。" * 20).encode(), "text/plain")},
                       headers=headers)
    assert resp.status_code == 200, resp.text
    doc_id = resp.json()["data"]["doc_id"]
    import tasks
    result = tasks.process_upload_task(resp.json()["data"]["task_id"])
    assert result["status"] == "completed", result
    return doc_id


def _mock_search(monkeypatch, hits_per_call):
    """按调用顺序返回不同命中，便于验证"多路召回 + 融合"。"""
    import milvus_ops
    calls = []

    def _fake(**kwargs):
        calls.append(kwargs)
        index = len(calls) - 1
        return hits_per_call[index] if index < len(hits_per_call) else []

    monkeypatch.setattr(milvus_ops, "search_vectors", _fake)
    return calls


class TestAdvancedRecall:
    def test_multi_path_recall_fuses_and_records_paths(self, client, auth_headers, fake_bge, fake_llm, monkeypatch):
        kb = _create_kb(client, auth_headers, "高级检索库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        client.put(f"/api/kb/{kb['kb_id']}/retrieval-strategy", json={"strategy": {
            "mode": "advanced", "enable_multi_query": True, "multi_query_count": 2,
            "enable_hyde": True, "fusion_method": "rrf", "per_path_k": 5, "total_candidate_cap": 20,
        }}, headers=auth_headers)
        fake_llm(answers=[("检索问题", "扩展问题一\n扩展问题二"), ("假设性回答片段", "假设答案：三个容器协同。")])

        # 原问题路径命中 chunk0/chunk1；扩展一命中 chunk1；扩展二命中 chunk2；HyDE 命中 chunk3
        calls = _mock_search(monkeypatch, [
            _search_hits(children[:2]),
            _search_hits([children[1]]),
            _search_hits([children[2]]),
            _search_hits([children[3]]),
        ])

        result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="有哪些容器", retrieval_mode="hybrid",
                                         debug=True)
        assert len(calls) == 4                       # original + multi_query_1 + multi_query_2 + hyde
        paths = set(result["debug"]["path_hits"])
        assert {"original", "multi_query_1", "multi_query_2", "hyde"} <= paths
        assert result["debug"]["pipeline"]["mode"] == "advanced"
        assert result["debug"]["fusion"]["method"] == "rrf"

        hit_paths = {path for item in result["items"] for path in item["hit_paths"]}
        assert {"original", "multi_query_1", "multi_query_2", "hyde"} <= hit_paths
        # chunk1 被两条路径命中 → RRF 下应排在首位
        assert result["items"][0]["chunk_id"] == children[1]
        assert result["items"][0]["hit_source"] == "multi_path"
        assert len({item["chunk_id"] for item in result["items"]}) == len(result["items"])  # 已去重

    def test_advanced_mode_degrades_to_original_path(self, client, auth_headers, fake_bge, fake_llm, monkeypatch):
        """改写/多查询/HyDE 全部失败时，仍用原问题完成检索（降级不阻断）。"""
        kb = _create_kb(client, auth_headers, "高级降级库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        # 关掉索引问题路径：本用例只考察"前置能力降级"，而离线替身 search_index_questions 返回 None
        # 会额外产生一条索引问题降级（见 TestRetrievalDegradations），会把断言搅浑。
        client.put(f"/api/kb/{kb['kb_id']}/retrieval-strategy", json={"strategy": {
            "mode": "advanced", "enable_rewrite": True, "enable_multi_query": True, "enable_hyde": True,
            "enable_index": False,
        }}, headers=auth_headers)
        fake_llm(invoke_error=RuntimeError("模型服务 502"))

        calls = _mock_search(monkeypatch, [_search_hits(children[:2])])
        result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="问题", retrieval_mode="dense_only",
                                         history=HISTORY, debug=True)

        assert len(calls) == 1                              # 只剩 original 路径
        assert result["total"] >= 1
        # 三项能力必须全部降级（这里用子集断言：离线替身的 Rerank 返回空，会额外产生一条 Rerank 降级）
        assert {"rewrite", "multi_query", "hyde"} <= {item["stage"] for item in result["degradations"]}
        assert all(item["fallback"] for item in result["degradations"])
        assert result["debug"]["query_plan"]["search_paths"] == ["original"]

    def test_fusion_method_switchable_via_strategy(self, client, auth_headers, fake_bge, fake_llm, monkeypatch):
        kb = _create_kb(client, auth_headers, "融合切换库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        client.put(f"/api/kb/{kb['kb_id']}/retrieval-strategy", json={"strategy": {
            "mode": "advanced", "enable_hyde": True, "fusion_method": "weighted",
        }}, headers=auth_headers)
        fake_llm(answers=[("假设性回答片段", "假设答案")])

        _mock_search(monkeypatch, [_search_hits(children[:2]), _search_hits(children[:2])])
        result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="问题", retrieval_mode="hybrid", debug=True)
        assert result["config_snapshot"]["fusion_method"] == "weighted"
        assert result["debug"]["fusion"]["method"] == "weighted"

        # 请求级临时覆盖优先于知识库策略
        _mock_search(monkeypatch, [_search_hits(children[:2]), _search_hits(children[:2])])
        overridden = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="问题", retrieval_mode="hybrid",
                                             strategy={"fusion_method": "rrf"}, debug=True)
        assert overridden["config_snapshot"]["fusion_method"] == "rrf"

    def test_basic_mode_unaffected_by_strategy_module(self, client, auth_headers, fake_bge, monkeypatch):
        kb = _create_kb(client, auth_headers, "基础模式库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        _mock_search(monkeypatch, [_search_hits(children[:2])])
        result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="问题", retrieval_mode="dense_only",
                                         enable_rerank=False, debug=True)
        assert result["debug"]["pipeline"]["mode"] == "basic"
        assert result["total"] >= 1
        assert "query_plan" not in result          # 基础模式不产生查询计划

    def test_recall_api_accepts_temporary_override(self, client, auth_headers, fake_bge, monkeypatch):
        kb = _create_kb(client, auth_headers, "召回覆盖库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]
        _mock_search(monkeypatch, [_search_hits(children[:1])] * 4)

        resp = client.post(f"/api/kb/{kb['kb_id']}/recall-test", json={
            "query": "容器", "retrieval_mode": "dense_only", "debug": True,
            "fusion_method": "weighted", "rerank_top_n": 2,
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        # 只做临时覆盖，不写库
        assert client.get(f"/api/kb/{kb['kb_id']}/retrieval-strategy",
                          headers=auth_headers).json()["data"]["customized"] is False
        assert data["config_snapshot"]["fusion_method"] == "weighted"


# ── 上下文预算 ──────────────────────────────────────────

class TestContextBudget:
    def test_max_context_chars_limits_context_and_citations(self, client, auth_headers, fake_bge, fake_llm, monkeypatch):
        kb = _create_kb(client, auth_headers, "上下文预算库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        client.put(f"/api/kb/{kb['kb_id']}/retrieval-strategy",
                   json={"strategy": {"max_context_chars": 200, "rerank_top_n": 5}},
                   headers=auth_headers)
        fake_llm(answer="最终回答")
        _mock_search(monkeypatch, [_search_hits(children[:5], base=0.9, step=0.05)])

        ctx = rag_pipeline.prepare_context(kb["kb_id"], "问题", None)
        assert ctx["messages"] is not None
        prompt = ctx["messages"][-1].content
        context_part = prompt.split("参考内容：\n", 1)[1].split("\n\n用户问题：", 1)[0]
        assert len(context_part) <= 200                    # 上下文被预算截断
        assert len(ctx["citations"]) <= 5
        assert ctx["strategy_snapshot"]["max_context_chars"] == 200


class TestModeCoercion:
    """运行模式与高级能力的联动（修复"勾了改写却没反应"）。"""

    def test_enabled_ability_implies_advanced_mode(self):
        """历史脏数据（mode=basic 却开了能力）→ 兜底提升为高级模式，并**显式标记** mode_adjusted。"""
        config = strategy_module.resolve(kb_strategy={"mode": "basic", "enable_rewrite": True})
        assert config["mode"] == "advanced"
        assert config["mode_adjusted"] is True

    def test_basic_mode_stays_basic_without_abilities(self):
        config = strategy_module.resolve(kb_strategy={"mode": "basic"})
        assert config["mode"] == "basic"
        assert "mode_adjusted" not in config

    def test_explicit_request_mode_wins(self):
        """请求级显式指定 basic（召回测试做基础/高级对比）时，即使策略勾了能力也尊重调用方。"""
        config = strategy_module.resolve(
            kb_strategy={"mode": "basic", "enable_rewrite": True},
            override={"mode": "basic"},
        )
        assert config["mode"] == "basic"
        assert "mode_adjusted" not in config

    def test_put_strategy_rejects_basic_with_abilities(self, client, auth_headers, sample_kb):
        """生产级：保存时严格校验，禁止产生"开了高级能力却停在基础模式"的矛盾配置。"""
        resp = client.put(f"/api/kb/{sample_kb['kb_id']}/retrieval-strategy",
                          json={"strategy": {"mode": "basic", "enable_rewrite": True}},
                          headers=auth_headers)
        assert resp.status_code == 400
        assert "运行模式必须是 advanced" in resp.json()["detail"]["message"]

    def test_put_strategy_accepts_advanced_with_abilities(self, client, auth_headers, sample_kb):
        resp = client.put(f"/api/kb/{sample_kb['kb_id']}/retrieval-strategy",
                          json={"strategy": {"mode": "advanced", "enable_hyde": True}},
                          headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["data"]["strategy"]["mode"] == "advanced"

    def test_recall_test_accepts_history_and_returns_rewritten(self, client, auth_headers, fake_bge,
                                                              fake_llm, monkeypatch):
        """召回测试可手填"上一轮问题"，改写结果会出现在 query_plan 里（验收场景 3 的证据）。"""
        kb = _create_kb(client, auth_headers, "召回改写库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        fake_llm(answers=[("改写后的问题", "公司的差旅报销标准中，住宿费的报销上限是多少？")])
        _mock_search(monkeypatch, [_search_hits(children[:1])])

        resp = client.post(f"/api/kb/{kb['kb_id']}/recall-test", json={
            "query": "那住宿费呢", "retrieval_mode": "dense_only", "mode": "advanced",
            "enable_rewrite": True, "history": ["公司的差旅报销标准是什么？"],
        }, headers=auth_headers)
        assert resp.status_code == 200, resp.text
        plan = resp.json()["data"]["query_plan"]
        assert plan["rewritten"] == "公司的差旅报销标准中，住宿费的报销上限是多少？"
        assert "rewritten" in plan["search_paths"]


# ── 检索路径级降级必须可见（需求六 / 验收场景 8）────────────
#
# 【背景】改前只有"改写/多查询/HyDE 失败"会进执行详情的「降级记录」：
#   索引问题检索失败在基础模式里只 print 一行日志（连 degraded 阶段都没有），
#   高级模式虽然记了 degraded 阶段，但没写进 detail.degradations → 「降级记录」为空。
# 下面三条用例把这两类降级都钉死：返回值 + 阶段状态 + 降级记录三处都要能看到。

def _boom_index_search(*args, **kwargs):
    raise RuntimeError("collection not found: rag_tutorial_index_questions")


def _trace_for(question="问题"):
    """在追踪上下文里执行被测代码（离线用例：手工开一条 trace，便于断言降级记录）。"""
    import observability
    return observability.active_trace(observability.TraceRecorder.start("qa", question=question))


class TestRetrievalDegradations:
    def test_basic_mode_index_question_failure_is_visible(self, client, auth_headers, fake_bge, monkeypatch):
        """基础模式：索引问题检索失败 → 仍返回结果，且降级在返回值/阶段/降级记录里都可见。"""
        import milvus_ops
        import observability

        kb = _create_kb(client, auth_headers, "基础降级库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        _mock_search(monkeypatch, [_search_hits(children[:2])])
        monkeypatch.setattr(milvus_ops, "search_index_questions", _boom_index_search)

        with _trace_for() as recorder:
            result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="问题",
                                              retrieval_mode="dense_only", enable_rerank=False,
                                              enable_index=True, debug=True)

        assert result["total"] >= 1                                   # 降级不阻断：普通召回照常出结果
        assert [item["stage"] for item in result["degradations"]] == [observability.STAGE_RETRIEVE]
        assert "索引问题检索失败" in result["degradations"][0]["reason"]
        assert result["degradations"][0]["fallback"]

        trace = observability.get_trace(recorder.finish(status=observability.STATUS_OK))
        degraded = [span for span in trace["spans"] if span["status"] == observability.STATUS_DEGRADED]
        assert any(span["stage"] == observability.STAGE_RETRIEVE for span in degraded)
        assert trace["degrade_count"] == 1                            # ← 「降级记录」区会显示这一条

    def test_advanced_mode_index_question_failure_is_visible(self, client, auth_headers, fake_bge, monkeypatch):
        """高级模式：索引问题路径失败 → 该路径 0 命中，其它路径正常，降级同样可见。"""
        import milvus_ops
        import observability

        kb = _create_kb(client, auth_headers, "高级降级库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]
        client.put(f"/api/kb/{kb['kb_id']}/retrieval-strategy",
                   json={"strategy": {"mode": "advanced"}}, headers=auth_headers)

        _mock_search(monkeypatch, [_search_hits(children[:2])])
        monkeypatch.setattr(milvus_ops, "search_index_questions", _boom_index_search)

        with _trace_for() as recorder:
            result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="问题",
                                              retrieval_mode="dense_only", debug=True)

        assert result["total"] >= 1
        assert result["debug"]["path_hits"]["index_question"] == 0     # 只有这一路被跳过
        assert any(item["stage"] == observability.STAGE_RETRIEVE for item in result["degradations"])
        assert "索引问题检索失败" in result["degradations"][0]["reason"]

        trace = observability.get_trace(recorder.finish(status=observability.STATUS_OK))
        assert trace["degrade_count"] >= 1

    def test_rerank_failure_is_recorded_as_degradation(self, client, auth_headers, fake_bge, monkeypatch):
        """Rerank 不可用（返回空 = 服务异常/无结果）→ 退化为融合序，且写进降级记录。"""
        import observability

        kb = _create_kb(client, auth_headers, "重排降级库")
        doc_id = _upload_and_index(client, auth_headers, kb["kb_id"], fake_bge)
        children = [child["chunk_id"] for child in database.get_chunks_for_doc(doc_id, role="child")]

        # fake_bge() 默认 rerank 返回空列表 → 等价于"重排服务不可用"
        fake_bge(rerank_result=[])
        _mock_search(monkeypatch, [_search_hits(children[:2])])
        monkeypatch.setattr("milvus_ops.search_index_questions", _boom_index_search)

        with _trace_for() as recorder:
            result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="问题",
                                              retrieval_mode="dense_only", enable_rerank=True,
                                              enable_index=False, debug=True)

        assert result["total"] >= 1                                    # 没有重排也能出结果（用融合分排序）
        stages = [item["stage"] for item in result["degradations"]]
        assert observability.STAGE_RERANK in stages
        assert all(item["fallback"] for item in result["degradations"])

        trace = observability.get_trace(recorder.finish(status=observability.STATUS_OK))
        assert any(span["stage"] == observability.STAGE_RERANK
                   and span["status"] == observability.STATUS_DEGRADED for span in trace["spans"])
