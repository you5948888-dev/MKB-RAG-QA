"""
父子分块与重新索引测试（Advanced RAG 迭代 · 第 2 期）。

测试范围：
  - 引擎层：父块/子块切分与层级关系、跨层级参数校验、通用分块不受影响
  - 入库层：父块只落 SQLite、**只有子块进向量库**、计数口径、删除清理
  - 检索层：子块命中 → 回查父块完整上下文 → 同一父块去重 → 引用可定位父块/子块
  - 一致性：配置变更拦截（409）、指纹过期标记、重新索引闭环（重建后不再过期）

设计前提（已冻结）：父子分块**只把子块嵌入 Milvus**，父块正文只存 SQLite，
检索命中子块后按 parent_chunk_id 回查父块组装上下文。
"""
import json

import pytest

import chunking
import database

PARA_1 = "第一段内容。" * 25      # 150 字
PARA_2 = "第二段内容。" * 25
PARA_3 = "第三段内容。" * 25
CONTENT = (PARA_1 + "\n\n" + PARA_2 + "\n\n" + PARA_3).encode("utf-8")

PC_CONFIG = {
    "strategy": "parent_child",
    "parent": {"separators": ["\n\n"], "chunk_size": 200, "keep_separator": True,
               "chunk_overlap": 0, "filter_whitespace": True},
    "child": {"separators": ["。", ""], "chunk_size": 80, "chunk_overlap": 0,
              "min_length": 0, "short_chunk_policy": "keep", "filter_whitespace": True},
}


# ── 引擎层 ──────────────────────────────────────────────

class TestParentChildEngine:
    """父子切分与层级关系"""

    def test_builds_parent_child_hierarchy(self):
        built = chunking.build_chunks(PARA_1 + "\n\n" + PARA_2 + "\n\n" + PARA_3, PC_CONFIG)
        assert built["strategy"] == "parent_child"
        assert len(built["parents"]) == 3
        assert len(built["children"]) == 6  # 每个 150 字父块 → 2 个 80 字以内子块

        for child in built["children"]:
            assert 0 <= child["parent_index"] < len(built["parents"])
        # 同一父块内 sense_index 从 0 连续递增
        first_parent_children = [c for c in built["children"] if c["parent_index"] == 0]
        assert [c["sense_index"] for c in first_parent_children] == [0, 1]
        assert all(c["length"] <= 80 for c in built["children"])

    def test_group_parents_nests_children(self):
        built = chunking.build_chunks(PARA_1 + "\n\n" + PARA_2, PC_CONFIG)
        grouped = chunking.group_parents(built)
        assert len(grouped) == 2
        assert all(len(parent["children"]) == 2 for parent in grouped)
        assert grouped[0]["children"][0]["content"] in grouped[0]["content"]

    def test_general_strategy_has_no_parents(self):
        built = chunking.build_chunks("abcdefghij", {"chunk_size": 4, "chunk_overlap": 0})
        assert built["strategy"] == "general"
        assert built["parents"] == []
        assert chunking.group_parents(built) == []
        assert all(child["parent_index"] == -1 for child in built["children"])

    def test_parent_must_be_larger_than_child(self):
        with pytest.raises(chunking.ChunkConfigError) as exc:
            chunking.validate_config({
                "strategy": "parent_child",
                "parent": {"chunk_size": 200},
                "child": {"chunk_size": 300, "chunk_overlap": 10},
            })
        assert "父块最大长度" in str(exc.value)

    def test_child_params_inherit_flat_fields(self):
        """只传扁平 chunk_size/chunk_overlap 时，应作为子块参数兜底。"""
        config = chunking.validate_config({"strategy": "parent_child", "chunk_size": 120, "chunk_overlap": 20})
        assert config["child"]["chunk_size"] == 120
        assert config["child"]["chunk_overlap"] == 20
        assert config["parent"]["chunk_size"] == chunking.DEFAULT_PARENT_CONFIG["chunk_size"]

    def test_parent_child_hash_differs_from_general(self):
        assert chunking.config_hash(chunking.validate_config(PC_CONFIG)) != \
               chunking.config_hash(chunking.validate_config({"chunk_size": 80, "chunk_overlap": 0}))


# ── 入库层 ──────────────────────────────────────────────

def _create_kb(client, headers, name, **extra):
    payload = {"name": name, "description": "父子分块测试"}
    payload.update(extra)
    resp = client.post("/api/kb", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]


def _upload(client, headers, kb_id, filename="pc.txt", content=CONTENT):
    return client.post("/api/documents/upload", data={"kb_id": kb_id},
                       files={"file": (filename, content, "text/plain")}, headers=headers)


def _run_worker(task_id):
    import tasks
    return tasks.process_upload_task(task_id)


class TestParentChildIngest:
    """父块落库、子块入向量库、计数与清理"""

    def test_ingest_saves_parents_and_children(self, client, auth_headers, fake_bge):
        fake_bge()
        kb = _create_kb(client, auth_headers, "父子入库库", chunk_strategy="parent_child",
                        chunk_config=PC_CONFIG)
        upload = _upload(client, auth_headers, kb["kb_id"])
        assert upload.status_code == 200, upload.text
        doc_id = upload.json()["data"]["doc_id"]
        task_id = upload.json()["data"]["task_id"]

        result = _run_worker(task_id)
        assert result["status"] == "completed", result
        assert result["parent_chunks"] == 3
        assert result["total_chunks"] == 6

        parents = database.get_chunks_for_doc(doc_id, role="parent")
        children = database.get_chunks_for_doc(doc_id, role="child")
        assert len(parents) == 3 and len(children) == 6
        # 父块 chunk_id 形如 {doc_id}_p0，子块形如 {doc_id}_c0 并指向父块
        assert parents[0]["chunk_id"] == f"{doc_id}_p0"
        assert children[0]["chunk_id"] == f"{doc_id}_c0"
        assert children[0]["parent_chunk_id"] == f"{doc_id}_p0"

        doc = database.get_document(doc_id)
        assert doc["parent_count"] == 3
        assert doc["child_count"] == 6
        assert doc["chunk_count"] == 6                      # 可检索块口径 = 子块数
        assert doc["chunk_config_hash"] == chunking.config_hash(chunking.validate_config(PC_CONFIG))

        task = database.get_upload_task(task_id)
        assert task["parent_count"] == 3 and task["child_count"] == 6
        assert task["embedded_count"] == 6
        assert task["current_stage"] == "completed"

    def test_only_children_are_embedded(self, client, auth_headers, fake_bge, monkeypatch):
        """父块绝对不能进向量库——这是"不改 Milvus schema"的前提。"""
        import milvus_ops
        captured = []
        monkeypatch.setattr(milvus_ops, "upsert_vectors", lambda rows: captured.extend(rows))

        fake_bge()
        kb = _create_kb(client, auth_headers, "只嵌子块库", chunk_strategy="parent_child",
                        chunk_config=PC_CONFIG)
        upload = _upload(client, auth_headers, kb["kb_id"])
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])

        parent_ids = {p["chunk_id"] for p in database.get_chunks_for_doc(doc_id, role="parent")}
        child_ids = {c["chunk_id"] for c in database.get_chunks_for_doc(doc_id, role="child")}
        embedded_pks = {row["pk"] for row in captured}
        assert embedded_pks == child_ids
        assert not (embedded_pks & parent_ids)

    def test_kb_chunk_count_excludes_parents(self, client, auth_headers, fake_bge):
        fake_bge()
        kb = _create_kb(client, auth_headers, "计数口径库", chunk_strategy="parent_child",
                        chunk_config=PC_CONFIG)
        upload = _upload(client, auth_headers, kb["kb_id"])
        _run_worker(upload.json()["data"]["task_id"])
        assert database.get_kb(kb["kb_id"])["chunk_count"] == 6
        assert database.get_kb(kb["kb_id"])["doc_count"] == 1

    def test_delete_document_removes_parents_and_children(self, client, auth_headers, fake_bge):
        fake_bge()
        kb = _create_kb(client, auth_headers, "删除清理库", chunk_strategy="parent_child",
                        chunk_config=PC_CONFIG)
        upload = _upload(client, auth_headers, kb["kb_id"])
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])
        assert database.get_chunks_for_doc(doc_id)

        resp = client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert database.get_chunks_for_doc(doc_id) == []          # 父块 + 子块都被清理
        assert database.get_document(doc_id) is None

    def test_index_question_candidates_exclude_parents(self, client, auth_headers, fake_bge):
        """索引问题只能绑定可检索叶子块，不能绑到父块上。"""
        fake_bge()
        kb = _create_kb(client, auth_headers, "索引问题候选库", chunk_strategy="parent_child",
                        chunk_config=PC_CONFIG)
        upload = _upload(client, auth_headers, kb["kb_id"])
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])

        candidates = database.get_chunks_for_index(kb["kb_id"])
        ids = {item["chunk_id"] for item in candidates}
        assert ids == {c["chunk_id"] for c in database.get_chunks_for_doc(doc_id, role="child")}


# ── 检索层 ──────────────────────────────────────────────

def _child_hits(doc_id, kb_id, source, limit=None):
    """构造 Milvus 检索命中（只返回子块，贴合"只有子块入库"的实现）。"""
    children = database.get_chunks_for_doc(doc_id, role="child")
    hits = []
    for index, child in enumerate(children[:limit]):
        hits.append({
            "pk": child["chunk_id"],
            "source": source,
            "chunk_index": child["chunk_index"],
            "text": child["content"],
            "dense_score": 0.9 - index * 0.05,
            "sparse_score": 0.0,
        })
    return hits


class TestParentChildRetrieval:
    """子块定位、父块上下文组装、同父块去重"""

    def test_recall_expands_to_parent_context_and_dedupes(self, client, auth_headers, fake_bge, monkeypatch):
        import milvus_ops
        import rag_pipeline

        fake_bge()
        kb = _create_kb(client, auth_headers, "父子检索库", chunk_strategy="parent_child",
                        chunk_config=PC_CONFIG)
        upload = _upload(client, auth_headers, kb["kb_id"])
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])

        hits = _child_hits(doc_id, kb["kb_id"], "pc.txt")   # 6 个子块，分属 3 个父块
        monkeypatch.setattr(milvus_ops, "search_vectors", lambda **kwargs: hits)

        result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="第一段内容", retrieval_mode="dense_only",
                                          top_k=10, enable_rerank=False, enable_index=False)
        assert result["total"] == 3                          # 3 个父块 → 去重后 3 条
        parents = database.get_chunks_for_doc(doc_id, role="parent")
        parent_contents = {p["content"] for p in parents}
        for item in result["items"]:
            assert item["content"] in parent_contents        # 给大模型的是父块完整上下文
            assert item["child_content"] in item["content"]  # 命中的子块包含在父块里
            assert item["parent_chunk_id"].startswith(f"{doc_id}_p")
            assert item["hit_child_count"] == 2              # 同一父块命中 2 个子块
            assert len(item["merged_children"]) == 2
        # 去重后每个父块只出现一次
        assert len({item["parent_chunk_id"] for item in result["items"]}) == 3

    def test_general_recall_keeps_child_content(self, client, auth_headers, fake_bge, monkeypatch):
        """通用分块文档不受父子逻辑影响（回归保护）。"""
        import milvus_ops
        import rag_pipeline

        fake_bge()
        kb = _create_kb(client, auth_headers, "通用检索库", chunk_size=100, chunk_overlap=0)
        upload = _upload(client, auth_headers, kb["kb_id"], filename="g.txt",
                         content=("通用分块内容。" * 30).encode("utf-8"))
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])

        children = database.get_chunks_for_doc(doc_id, role="child")
        hits = [{
            "pk": children[0]["chunk_id"], "source": "g.txt",
            "chunk_index": children[0]["chunk_index"], "text": children[0]["content"],
            "dense_score": 0.8, "sparse_score": 0.0,
        }]
        monkeypatch.setattr(milvus_ops, "search_vectors", lambda **kwargs: hits)

        result = rag_pipeline.recall_test(kb_id=kb["kb_id"], query="内容", retrieval_mode="dense_only",
                                          top_k=5, enable_rerank=False, enable_index=False)
        assert result["total"] == 1
        assert result["items"][0]["content"] == children[0]["content"]
        assert not result["items"][0].get("parent_chunk_id")

    def test_prepare_context_citations_locate_parent_and_child(self, client, auth_headers, fake_bge, monkeypatch):
        import milvus_ops
        import rag_pipeline

        fake_bge()
        kb = _create_kb(client, auth_headers, "父子引用库", chunk_strategy="parent_child",
                        chunk_config=PC_CONFIG)
        upload = _upload(client, auth_headers, kb["kb_id"])
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])

        hits = _child_hits(doc_id, kb["kb_id"], "pc.txt", limit=2)   # 都命中父块 0
        monkeypatch.setattr(milvus_ops, "search_vectors", lambda **kwargs: hits)

        ctx = rag_pipeline.prepare_context(kb["kb_id"], "第一段讲了什么", None)
        assert ctx["messages"] is not None
        assert len(ctx["citations"]) == 1                            # 同一父块只给一份上下文

        citation = ctx["citations"][0]
        assert citation["source"] == "pc.txt"
        assert citation["parent_index"] == 0
        assert citation["child_chunk_id"].startswith(f"{doc_id}_c")
        assert citation["child_content"] in citation["content"]
        assert citation["hit_child_count"] == 2

        prompt = ctx["messages"][-1].content
        assert "父块 #0" in prompt
        assert citation["content"] in prompt                         # 大模型收到父块完整上下文


# ── 配置变更与重新索引 ──────────────────────────────────

class TestReindexFlow:
    """需求 2.3：配置变更不得与旧索引混用，重新索引后恢复一致"""

    def test_switch_config_requires_explicit_confirm(self, client, auth_headers, sample_kb):
        """有文档时先 409（附旧/新配置），确认后才允许切换。"""
        upload = _upload(client, auth_headers, sample_kb["kb_id"])
        task_id = upload.json()["data"]["task_id"]

        resp = client.put(f"/api/kb/{sample_kb['kb_id']}/chunk-config",
                          json={"chunk_config": PC_CONFIG}, headers=auth_headers)
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["requires_reindex"] is True
        assert detail["doc_count"] == 1
        assert detail["old_config"]["strategy"] == "general"
        assert detail["new_config"]["strategy"] == "parent_child"

        confirmed = client.put(f"/api/kb/{sample_kb['kb_id']}/chunk-config",
                               json={"chunk_config": PC_CONFIG, "switch_and_reindex": True},
                               headers=auth_headers)
        assert confirmed.status_code == 200
        data = confirmed.json()["data"]
        assert data["chunk_config"]["strategy"] == "parent_child"
        # 这篇文档还没被 Worker 领取：直接把排队任务的分块快照换成新配置，无需重新索引
        assert data["docs_retargeted"] == 1
        assert data["requires_reindex"] is False
        assert json.loads(database.get_upload_task(task_id)["chunk_config"])["strategy"] == "parent_child"

    def test_switch_config_after_ingest_requires_reindex(self, client, auth_headers, sample_kb, fake_bge):
        """已经按旧配置入库过的文档：切换后必须重新索引才算一致。"""
        fake_bge()
        upload = _upload(client, auth_headers, sample_kb["kb_id"])
        _run_worker(upload.json()["data"]["task_id"])

        confirmed = client.put(f"/api/kb/{sample_kb['kb_id']}/chunk-config",
                               json={"chunk_config": PC_CONFIG, "switch_and_reindex": True},
                               headers=auth_headers)
        data = confirmed.json()["data"]
        assert data["requires_reindex"] is True
        assert data["docs_to_reindex"] == 1
        assert data["docs_retargeted"] == 0

    def test_index_status_flags_outdated_documents(self, client, auth_headers, sample_kb, fake_bge):
        fake_bge()
        upload = _upload(client, auth_headers, sample_kb["kb_id"], filename="old.txt")
        _run_worker(upload.json()["data"]["task_id"])

        before = client.get(f"/api/kb/{sample_kb['kb_id']}/index-status", headers=auth_headers)
        assert before.json()["data"]["outdated_count"] == 0

        client.put(f"/api/kb/{sample_kb['kb_id']}/chunk-config",
                   json={"chunk_config": PC_CONFIG, "switch_and_reindex": True}, headers=auth_headers)

        after = client.get(f"/api/kb/{sample_kb['kb_id']}/index-status", headers=auth_headers)
        data = after.json()["data"]
        assert data["outdated_count"] == 1
        assert data["items"][0]["config_outdated"] is True
        assert data["items"][0]["doc_config_hash"]

        docs = client.get(f"/api/documents?kb_id={sample_kb['kb_id']}", headers=auth_headers)
        assert docs.json()["data"]["items"][0]["config_outdated"] is True

    def test_reindex_rebuilds_and_clears_outdated(self, client, auth_headers, fake_bge):
        fake_bge()
        kb = _create_kb(client, auth_headers, "重建索引库", chunk_size=100, chunk_overlap=0)
        upload = _upload(client, auth_headers, kb["kb_id"])
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])
        # 重建前：通用分块，没有父块，子块也没有父块指向
        assert database.get_chunks_for_doc(doc_id, role="parent") == []
        before = database.get_chunks_for_doc(doc_id, role="child")
        assert before and all(child["parent_chunk_id"] == "" for child in before)

        client.put(f"/api/kb/{kb['kb_id']}/chunk-config",
                   json={"chunk_config": PC_CONFIG, "switch_and_reindex": True}, headers=auth_headers)

        resp = client.post(f"/api/kb/{kb['kb_id']}/reindex", json={}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["submitted"] == 1 and data["skipped"] == []
        task_id = data["reindex_tasks"][0]["task_id"]
        assert database.get_upload_task(task_id)["task_type"] == "reindex"

        result = _run_worker(task_id)
        assert result["status"] == "completed", result

        # 重建后：父块/子块按新配置生成，指纹一致，不再标记过期
        assert len(database.get_chunks_for_doc(doc_id, role="parent")) == 3
        assert len(database.get_chunks_for_doc(doc_id, role="child")) == 6
        doc = database.get_document(doc_id)
        assert doc["chunk_config_hash"] == chunking.config_hash(chunking.validate_config(PC_CONFIG))
        status = client.get(f"/api/kb/{kb['kb_id']}/index-status", headers=auth_headers)
        assert status.json()["data"]["outdated_count"] == 0

    def test_reindex_without_outdated_documents_returns_400(self, client, auth_headers, fake_bge):
        fake_bge()
        kb = _create_kb(client, auth_headers, "无需重建库")
        upload = _upload(client, auth_headers, kb["kb_id"])
        _run_worker(upload.json()["data"]["task_id"])

        resp = client.post(f"/api/kb/{kb['kb_id']}/reindex", json={}, headers=auth_headers)
        assert resp.status_code == 400
        assert "没有需要重新索引的文档" in resp.json()["detail"]["message"]

    def test_reindex_skips_documents_without_file(self, client, auth_headers, fake_bge):
        fake_bge()
        kb = _create_kb(client, auth_headers, "缺文件重建库", chunk_size=100, chunk_overlap=0)
        upload = _upload(client, auth_headers, kb["kb_id"])
        doc_id = upload.json()["data"]["doc_id"]
        _run_worker(upload.json()["data"]["task_id"])
        client.put(f"/api/kb/{kb['kb_id']}/chunk-config",
                   json={"chunk_config": PC_CONFIG, "switch_and_reindex": True}, headers=auth_headers)

        import os
        os.remove(database.get_document(doc_id)["path"])

        resp = client.post(f"/api/kb/{kb['kb_id']}/reindex", json={}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["submitted"] == 0
        assert "原始文件不存在" in data["skipped"][0]["reason"]

    def test_reindex_progress_fields_visible(self, client, auth_headers, fake_bge):
        """重建过程中能看到阶段/父块/子块/已嵌入数量（需求四.2 的文档链路观测基础）。"""
        fake_bge()
        kb = _create_kb(client, auth_headers, "重建进度库", chunk_config=PC_CONFIG,
                        chunk_strategy="parent_child")
        upload = _upload(client, auth_headers, kb["kb_id"])
        task_id = upload.json()["data"]["task_id"]
        _run_worker(task_id)

        task = database.get_upload_task(task_id)
        assert task["task_type"] == "upload"
        assert task["current_stage"] == "completed"
        assert task["parent_count"] == 3 and task["child_count"] == 6 and task["embedded_count"] == 6
