"""
上传任务状态机单元测试。

测试范围：
  - 创建任务 → 初始状态为 uploaded
  - 获取任务详情（存在 / 不存在）
  - 任务列表（按知识库过滤 / 全部）
  - 重试失败任务 / 重试非 failed 任务应失败
  - 删除任务（只允许终态）

说明：任务状态由 Celery Worker 更新，TestClient 无法模拟，因此本文件直接操作
database 层；`/tmp/...` 只是「未执行解析」的记录值，不访问开发目录。
不要在测试收集阶段导入 app。
"""
import pytest

import database


@pytest.fixture(autouse=True)
def task_documents(sample_kb):
    """任务必须引用当前知识库内实际存在的文档，而不是任意字符串 ID。"""
    for number in (1, 2):
        doc = database.create_document(
            sample_kb["kb_id"], f"test-{number}.pdf", f"/tmp/test-{number}.pdf"
        )
        conn = database.get_connection()
        with conn:
            conn.execute(
                "UPDATE kb_documents SET doc_id=? WHERE doc_id=?",
                (f"doc-00{number}", doc["doc_id"]),
            )
        conn.close()


def _set_task_status(task_id: str, **fields):
    """直接改库模拟 Worker 走过的中间态。"""
    columns = ", ".join(f"{key}=?" for key in fields)
    conn = database.get_connection()
    with conn:
        conn.execute(
            f"UPDATE upload_tasks SET {columns} WHERE task_id=?",
            (*fields.values(), task_id),
        )
    conn.close()


class TestCreateTask:
    """创建任务"""

    def test_create_task_initial_status(self, sample_kb):
        task = database.create_upload_task(
            kb_id=sample_kb["kb_id"],
            doc_id="doc-001",
            filename="test.pdf",
            filepath="/tmp/test.pdf",
            chunk_size=800,
            chunk_overlap=100,
        )
        assert task["status"] == "uploaded"
        assert task["progress"] == 0
        assert task["step_message"] == "已上传，等待处理"
        assert task["task_id"] is not None


class TestGetTask:
    """获取任务详情"""

    def test_get_task_success(self, sample_kb):
        task = database.create_upload_task(
            kb_id=sample_kb["kb_id"], doc_id="doc-001",
            filename="test.pdf", filepath="/tmp/test.pdf",
        )
        result = database.get_upload_task(task["task_id"])
        assert result is not None
        assert result["status"] == "uploaded"

    def test_get_task_not_found(self, sample_kb):
        result = database.get_upload_task("nonexistent-task-id")
        assert result is None


class TestListTasks:
    """任务列表"""

    def test_list_tasks_by_kb(self, sample_kb):
        database.create_upload_task(
            kb_id=sample_kb["kb_id"], doc_id="doc-001",
            filename="a.pdf", filepath="/tmp/a.pdf",
        )
        database.create_upload_task(
            kb_id=sample_kb["kb_id"], doc_id="doc-002",
            filename="b.pdf", filepath="/tmp/b.pdf",
        )
        items = database.list_upload_tasks(sample_kb["kb_id"])
        assert len(items) == 2

    def test_list_tasks_all(self, sample_kb):
        database.create_upload_task(
            kb_id=sample_kb["kb_id"], doc_id="doc-001",
            filename="a.pdf", filepath="/tmp/a.pdf",
        )
        items = database.list_upload_tasks(None)
        assert len(items) >= 1


class TestRetryTask:
    """重试任务"""

    def test_retry_failed_task(self, sample_kb):
        task = database.create_upload_task(
            kb_id=sample_kb["kb_id"], doc_id="doc-001",
            filename="test.pdf", filepath="/tmp/test.pdf",
        )
        _set_task_status(task["task_id"], status="failed", error_message="测试错误")

        # 重试
        result = database.retry_upload_task(task["task_id"])
        assert result is not None
        assert result["status"] == "uploaded"

        # 验证 error_message 被清空、进度归零
        updated = database.get_upload_task(task["task_id"])
        assert updated["error_message"] == ""
        assert updated["progress"] == 0

    def test_retry_non_failed_task(self, sample_kb):
        task = database.create_upload_task(
            kb_id=sample_kb["kb_id"], doc_id="doc-001",
            filename="test.pdf", filepath="/tmp/test.pdf",
        )
        # uploaded 状态的任务不能重试
        result = database.retry_upload_task(task["task_id"])
        assert result is None


class TestDeleteTask:
    """删除任务"""

    def test_delete_task_success(self, sample_kb):
        task = database.create_upload_task(
            kb_id=sample_kb["kb_id"], doc_id="doc-001",
            filename="test.pdf", filepath="/tmp/test.pdf",
        )
        # 只有终态任务允许删除
        _set_task_status(task["task_id"], status="completed")

        ok = database.delete_upload_task(task["task_id"])
        assert ok is True

        # 验证已删除
        assert database.get_upload_task(task["task_id"]) is None
