"""
知识库 CRUD 单元测试。

测试范围：
  - 创建知识库（成功 / 重复名称失败）
  - 获取知识库列表（空列表 / 创建后可见）
  - 删除知识库（成功 / 不存在）
  - 认证守卫（未登录 401 / 注册后登录）

说明：本项目的 `/api/kb` 列表直接返回 `{"items": [...], "total": n}`（不带 code/data 包装），
与创建/删除接口的 `{code, message, data}` 结构不同，用例按真实契约断言。
"""


class TestCreateKB:
    """创建知识库"""

    def test_create_kb_success(self, client, auth_headers):
        resp = client.post("/api/kb", json={
            "name": "新知识库",
            "description": "测试描述",
            "chunk_size": 800,
            "chunk_overlap": 100,
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "新知识库"
        assert data["chunk_size"] == 800
        assert data["chunk_overlap"] == 100
        assert data["doc_count"] == 0
        assert data["kb_id"] is not None

    def test_create_kb_duplicate_name(self, client, auth_headers, sample_kb):
        resp = client.post("/api/kb", json={
            "name": "测试知识库",
        }, headers=auth_headers)
        assert resp.status_code == 409


class TestListKB:
    """获取知识库列表"""

    def test_list_kbs_empty(self, client, auth_headers):
        """空列表用例不依赖 sample_kb，因此必须为空。"""
        resp = client.get("/api/kb", headers=auth_headers)
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert items == []

    def test_list_kbs_after_create(self, client, auth_headers, sample_kb):
        resp = client.get("/api/kb", headers=auth_headers)
        items = resp.json()["items"]
        names = [kb["name"] for kb in items]
        assert "测试知识库" in names


class TestDeleteKB:
    """删除知识库"""

    def test_delete_kb_success(self, client, auth_headers, sample_kb):
        kb_id = sample_kb["kb_id"]
        resp = client.delete(f"/api/kb/{kb_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True

        # 验证已删除
        resp = client.get("/api/kb", headers=auth_headers)
        names = [kb["name"] for kb in resp.json()["items"]]
        assert "测试知识库" not in names

    def test_delete_kb_not_found(self, client, auth_headers):
        resp = client.delete("/api/kb/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404


class TestAuthRequired:
    """认证守卫测试"""

    def test_unauthorized_access(self, client):
        """未登录访问 API 应返回 401。"""
        resp = client.get("/api/kb")
        assert resp.status_code == 401

    def test_register_and_login(self, client):
        """注册新用户 → 登录 → 访问 API。"""
        # 注册
        resp = client.post("/api/auth/register", json={
            "username": "testuser",
            "password": "test1234",
        })
        assert resp.status_code == 200

        # 登录
        resp = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "test1234",
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["role"] == "user"
