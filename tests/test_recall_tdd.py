"""
召回测试（mock Milvus 和 BGE 服务）。

测试范围：
  - POST /api/kb/{kb_id}/recall-test 端点
  - 检索模式校验（不支持的模式 → 400）
  - 空结果与知识库不存在
  - Milvus / 模型不可用时的错误处理（→ 500）
  - 参数透传（rerank / top_k）

测试环境没有 Milvus 与 BGE，用 `unittest.mock.patch` 替换
`routes.kb_routes.rag_pipeline.recall_test`——注意 patch 的位置必须是
「被测函数运行时查找符号的地方」，这里 KB 路由用的是 `rag_pipeline.recall_test`。
"""
from unittest.mock import patch


class TestRecallEndpoint:
    """召回测试 API 端点"""

    def test_recall_invalid_mode(self, client, auth_headers, sample_kb):
        """不支持的检索模式应返回 400。"""
        resp = client.post(
            f"/api/kb/{sample_kb['kb_id']}/recall-test",
            json={
                "query": "测试查询",
                "retrieval_mode": "invalid_mode",
                "top_k": 5,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    @patch("routes.kb_routes.rag_pipeline.recall_test")
    def test_recall_dense_only(self, mock_recall, client, auth_headers, sample_kb):
        """mock Milvus，测试 dense_only 模式。"""
        mock_recall.return_value = {
            "items": [
                {
                    "rank": 1,
                    "content": "测试内容",
                    "source": "test.pdf",
                    "chunk_index": 0,
                    "dense_sim": 0.85,
                    "sparse_sim": 0.60,
                    "fused_score": 0.73,
                    "rerank_score": None,
                }
            ],
            "total": 1,
            "avg_similarity": 0.73,
        }

        resp = client.post(
            f"/api/kb/{sample_kb['kb_id']}/recall-test",
            json={
                "query": "测试查询",
                "retrieval_mode": "dense_only",
                "top_k": 5,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["content"] == "测试内容"

        # 路由应把规范化后的模式传给管道
        assert mock_recall.call_args[1]["retrieval_mode"] == "dense_only"

    @patch("routes.kb_routes.rag_pipeline.recall_test")
    def test_recall_milvus_error(self, mock_recall, client, auth_headers, sample_kb):
        """Milvus 异常时应返回 500（而不是把异常泄漏成 50000 兜底错误）。"""
        mock_recall.side_effect = Exception("Milvus 连接失败")

        resp = client.post(
            f"/api/kb/{sample_kb['kb_id']}/recall-test",
            json={
                "query": "测试查询",
                "retrieval_mode": "hybrid",
                "top_k": 5,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 500
        assert "召回测试失败" in resp.json()["detail"]

    def test_recall_unknown_kb(self, client, auth_headers):
        """知识库不存在（或无权限）→ 404，且不会调用检索管道。"""
        resp = client.post(
            "/api/kb/no-such-kb/recall-test",
            json={"query": "测试", "retrieval_mode": "hybrid", "top_k": 5},
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestRecallParameters:
    """召回参数校验"""

    @patch("routes.kb_routes.rag_pipeline.recall_test")
    def test_recall_with_rerank(self, mock_recall, client, auth_headers, sample_kb):
        """开启 rerank 的请求。"""
        mock_recall.return_value = {"items": [], "total": 0, "avg_similarity": 0}

        resp = client.post(
            f"/api/kb/{sample_kb['kb_id']}/recall-test",
            json={
                "query": "测试",
                "retrieval_mode": "hybrid",
                "top_k": 3,
                "candidate_k": 10,
                "enable_rerank": True,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200

        # 验证 mock 被正确调用
        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["enable_rerank"] is True
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["candidate_k"] == 10
