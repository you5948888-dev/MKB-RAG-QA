"""
回归测试：任务领取、失败补偿、删除阻塞与僵尸清理。

测试范围：
  - 条件领取（同一任务只能被一个 Worker 领取）
  - dispatch_unknown 只允许从 uploaded 进入，不覆盖已被领取的任务
  - 删除文档时正在处理的任务被标记 failed，且不允许重试
  - 非终态任务阻止删除知识库（409）
  - 僵尸任务（parsing 超 30 分钟）自动标记 failed

说明：这些行为直接决定「同一文档是否会被重复向量化」「删库是否安全」，
属于必须回归的边界，因此独立成套。用例直接操作 database 层 + TestClient 混合验证。
"""
from datetime import datetime, timedelta

import database


def _make_task(kb_id, filename="repair.txt"):
    """在给定知识库下创建真实文档 + 上传任务（任务必须引用真实文档）。"""
    doc = database.create_document(kb_id, filename, f"/tmp/{filename}")
    assert doc is not None
    task = database.create_upload_task(
        kb_id=kb_id, doc_id=doc["doc_id"], filename=filename, filepath=f"/tmp/{filename}",
    )
    return doc, task


class TestClaimTask:
    """任务领取（防重复处理）"""

    def test_claim_upload_task_only_once(self, sample_kb):
        _, task = _make_task(sample_kb["kb_id"])

        claimed = database.claim_upload_task(task["task_id"])
        assert claimed is not None
        assert claimed["status"] == "uploaded"  # 返回领取前的快照

        current = database.get_upload_task(task["task_id"])
        assert current["status"] == "parsing"
        assert current["progress"] == 20

        # 第二个 Worker 再领取必须失败，否则同一文档会被重复向量化
        assert database.claim_upload_task(task["task_id"]) is None

    def test_claim_terminal_task_returns_none(self, sample_kb, set_upload_task_fields):
        _, task = _make_task(sample_kb["kb_id"])
        set_upload_task_fields(task["task_id"], status="completed")
        assert database.claim_upload_task(task["task_id"]) is None

    def test_mark_dispatch_unknown_only_when_uploaded(self, sample_kb):
        """投递结果未知时标记；一旦任务已被领取就不再覆盖。"""
        _, task = _make_task(sample_kb["kb_id"])
        assert database.mark_upload_dispatch_unknown(task["task_id"], "broker 不可达") is True
        assert database.get_upload_task(task["task_id"])["status"] == "dispatch_unknown"

        assert database.claim_upload_task(task["task_id"]) is not None  # dispatch_unknown 可被领取
        assert database.mark_upload_dispatch_unknown(task["task_id"], "再次投递失败") is False
        assert database.get_upload_task(task["task_id"])["status"] == "parsing"


class TestFailureCompensation:
    """失败补偿：文档删除与重试边界"""

    def test_delete_document_cancels_running_task(self, sample_kb, set_upload_task_fields):
        doc, task = _make_task(sample_kb["kb_id"])
        set_upload_task_fields(task["task_id"], status="parsing", progress=20)

        database.delete_document(doc["doc_id"])

        # 处理中的任务被保留并标记失败，Worker 下次检查点会发现并停止
        cancelled = database.get_upload_task(task["task_id"])
        assert cancelled is not None
        assert cancelled["status"] == "failed"
        assert "文档已被删除" in cancelled["step_message"]
        assert database.get_document(doc["doc_id"]) is None

    def test_retry_rejected_after_document_deleted(self, sample_kb, set_upload_task_fields):
        doc, task = _make_task(sample_kb["kb_id"])
        set_upload_task_fields(task["task_id"], status="parsing")
        database.delete_document(doc["doc_id"])

        # 文档已不存在，重试无意义 → 明确拒绝
        assert database.retry_upload_task(task["task_id"]) is None


class TestDeleteBlockedByRunningTask:
    """删除拦截：非终态任务阻止删库（Ch11）"""

    def test_delete_kb_blocked_by_running_upload_task(self, client, auth_headers, sample_kb):
        _make_task(sample_kb["kb_id"])  # uploaded 属于非终态

        resp = client.delete(f"/api/kb/{sample_kb['kb_id']}", headers=auth_headers)
        assert resp.status_code == 409
        assert "上传任务" in resp.json()["detail"]

        # 知识库必须仍然存在
        assert database.get_kb(sample_kb["kb_id"]) is not None

    def test_delete_kb_blocked_by_running_index_build_task(self, client, auth_headers, sample_kb):
        database.create_index_build_task(sample_kb["kb_id"])

        resp = client.delete(f"/api/kb/{sample_kb['kb_id']}", headers=auth_headers)
        assert resp.status_code == 409
        assert "索引构建" in resp.json()["detail"]
        assert database.get_kb(sample_kb["kb_id"]) is not None


class TestStaleTaskCleanup:
    """僵尸任务清理"""

    def test_stale_task_cleanup_marks_failed(self, sample_kb, set_upload_task_fields):
        """Worker 被 kill 后停在 parsing 的任务，30 分钟后应自动失败可重试。"""
        _, task = _make_task(sample_kb["kb_id"])
        stale_time = (datetime.now() - timedelta(minutes=40)).isoformat(timespec="seconds")
        set_upload_task_fields(task["task_id"], status="parsing", updated_at=stale_time)

        assert database.cleanup_stale_tasks() == 1

        cleaned = database.get_upload_task(task["task_id"])
        assert cleaned["status"] == "failed"
        assert "Worker" in cleaned["step_message"]
        # 清理后允许用户重试
        assert database.retry_upload_task(task["task_id"]) is not None
