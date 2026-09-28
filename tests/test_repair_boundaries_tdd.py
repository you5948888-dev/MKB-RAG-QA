"""
回归测试：上传边界、真实分数处理、资源归属与授权。

测试范围：
  - 同名上传 / 不支持扩展名 / 目标知识库不存在
  - 上传成功后文件落盘与任务创建（Celery 投递已被替换）
  - COSINE dense 分与 sparse IP 分的归一化契约
  - 普通用户只能看到自己的知识库，越权访问 KB / 任务返回 404

说明：分数归一化是纯函数（无外部依赖），直接调用即可，属于测试金字塔的单元层；
其余用例走 TestClient 集成层。
"""
from pathlib import Path

import pytest

from rag_pipeline import _normalize_dense_score, _normalize_sparse_score


@pytest.fixture
def second_user_headers(client):
    """注册一个普通用户，返回其 API Key 认证头（用于越权回归）。

    注册会写入 user2 的登录 Cookie，且 `get_current_user` 优先读 Cookie——
    若不清理，同一次用例里再带 admin 的 API Key 也会被识别成 user2，
    因此这里显式退出登录，让后续请求只依赖显式传入的认证头。
    """
    resp = client.post("/api/auth/register", json={
        "username": "user2",
        "password": "test1234",
    })
    assert resp.status_code == 200, resp.text
    api_key = resp.json()["data"]["api_key"]
    assert client.post("/api/auth/logout").status_code == 200
    return {"X-API-Key": api_key}


def _upload(client, headers, kb_id, filename="note.txt", content=b"hello rag"):
    return client.post(
        "/api/documents/upload",
        data={"kb_id": kb_id},
        files={"file": (filename, content, "text/plain")},
        headers=headers,
    )


class TestUploadBoundaries:
    """上传边界：重复文件名、非法类型、非法知识库"""

    def test_duplicate_filename_conflict(self, client, auth_headers, sample_kb):
        """同一知识库下同名文件 → 409，且不产生第二条文档记录。"""
        first = _upload(client, auth_headers, sample_kb["kb_id"])
        assert first.status_code == 200, first.text

        second = _upload(client, auth_headers, sample_kb["kb_id"])
        assert second.status_code == 409

        resp = client.get(f"/api/documents?kb_id={sample_kb['kb_id']}", headers=auth_headers)
        assert len(resp.json()["data"]["items"]) == 1

    def test_unsupported_extension_rejected(self, client, auth_headers, sample_kb):
        resp = _upload(client, auth_headers, sample_kb["kb_id"],
                       filename="恶意.exe", content=b"MZ")
        assert resp.status_code == 400

    def test_upload_unknown_kb(self, client, auth_headers):
        resp = _upload(client, auth_headers, "no-such-kb")
        assert resp.status_code == 404

    def test_upload_creates_task_and_saves_file(self, client, auth_headers, sample_kb, _use_test_db):
        """上传只负责落盘 + 建任务（异步），不等解析完成。"""
        resp = _upload(client, auth_headers, sample_kb["kb_id"])
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["status"] == "uploaded"
        assert data["task_id"]

        # 文件写入的是测试上传目录，不污染开发目录
        saved_files = list(Path(_use_test_db["upload_dir"]).iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == b"hello rag"

        # 任务可查询，且 Celery 投递替身后仍停留在 uploaded
        task = client.get(f"/api/upload-tasks/{data['task_id']}", headers=auth_headers)
        assert task.status_code == 200
        assert task.json()["data"]["status"] == "uploaded"


class TestScoreNormalization:
    """真实分数归一化契约"""

    def test_dense_score_clamped_to_unit_range(self):
        # COSINE 距离截断到 [0, 1]
        assert _normalize_dense_score(-0.5) == 0.0
        assert _normalize_dense_score(0.0) == 0.0
        assert _normalize_dense_score(0.85) == pytest.approx(0.85)
        assert _normalize_dense_score(1.0) == 1.0
        assert _normalize_dense_score(1.7) == 1.0

    def test_sparse_score_squashed_and_monotonic(self):
        # 非负 IP 用 x/(1+x) 压缩：负数归零、单调递增、不超过 1
        assert _normalize_sparse_score(-3.0) == 0.0
        assert _normalize_sparse_score(0.0) == 0.0
        assert _normalize_sparse_score(1.0) == pytest.approx(0.5)
        assert _normalize_sparse_score(3.0) == pytest.approx(0.75)
        assert _normalize_sparse_score(3.0) < 1.0
        assert _normalize_sparse_score(3.0) > _normalize_sparse_score(1.0)


class TestOwnershipAndAuthorization:
    """资源归属与越权访问"""

    def test_user_only_sees_own_kb(self, client, auth_headers, sample_kb, second_user_headers):
        """admin 建的库对普通用户不可见。"""
        admin_list = client.get("/api/kb", headers=auth_headers)
        assert "测试知识库" in [kb["name"] for kb in admin_list.json()["items"]]

        user_list = client.get("/api/kb", headers=second_user_headers)
        assert user_list.status_code == 200
        assert user_list.json()["items"] == []

    def test_cross_user_kb_access_denied(self, client, auth_headers, sample_kb, second_user_headers):
        kb_id = sample_kb["kb_id"]
        assert client.get(f"/api/kb/{kb_id}", headers=second_user_headers).status_code == 404
        assert client.delete(f"/api/kb/{kb_id}", headers=second_user_headers).status_code == 404
        # 知识库仍然存在（未被越权删除）
        assert client.get(f"/api/kb/{kb_id}", headers=auth_headers).status_code == 200

    def test_cross_user_task_access_denied(self, client, auth_headers, sample_kb, second_user_headers):
        """上传任务同样按知识库归属鉴权。"""
        resp = _upload(client, auth_headers, sample_kb["kb_id"])
        task_id = resp.json()["data"]["task_id"]

        assert client.get(f"/api/upload-tasks/{task_id}",
                          headers=second_user_headers).status_code == 404
        assert client.delete(f"/api/upload-tasks/{task_id}",
                             headers=second_user_headers).status_code == 404
