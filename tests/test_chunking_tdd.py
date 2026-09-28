"""
通用分块配置测试（Advanced RAG 迭代 · 第 1 期）。

测试范围：
  - 纯函数层：分隔符优先级与回退、长度口径（char/token）、保留分隔符、
    空白过滤、过短分块策略、参数校验与明确报错
  - 接口层：建库时提交配置、查看当前配置、无文档时改配置生效、
    有文档时改配置被拦截（409）、分块预览
  - 一致性：**预览结果 == Worker 实际入库结果**（验收场景 1 的核心证据）

说明：`chunking` 是纯函数模块，大部分用例不依赖网络与数据库；
涉及接口的用例用 TestClient + 临时库（见 conftest）。
"""
import pytest

import chunking
import database


# ── 纯函数层 ────────────────────────────────────────────

class TestConfigValidation:
    """参数校验：非法参数必须给出明确原因"""

    @pytest.mark.parametrize("patch, keyword", [
        ({"chunk_overlap": 800}, "必须小于分块最大长度"),
        ({"chunk_size": 0}, "必须大于 0"),
        ({"chunk_size": 99999}, "不能超过"),
        ({"separators": []}, "至少需要配置一个分隔符"),
        ({"length_mode": "word"}, "长度计算方式仅支持"),
        ({"short_chunk_policy": "unknown"}, "过短分块处理方式仅支持"),
        ({"min_length": 900}, "不能大于分块最大长度"),
        ({"chunk_overlap": -1}, "不能为负数"),
        ({"strategy": "magic"}, "分块策略仅支持"),
        # 第 2 期：父子分块的跨层级约束
        ({"strategy": "parent_child", "parent": {"chunk_size": 100},
          "child": {"chunk_size": 200, "chunk_overlap": 10}}, "父块最大长度"),
        ({"strategy": "parent_child", "child": {"chunk_size": 100, "chunk_overlap": 100}},
         "子块重叠长度"),
    ])
    def test_invalid_config_rejected_with_reason(self, patch, keyword):
        with pytest.raises(chunking.ChunkConfigError) as exc:
            chunking.validate_config(patch)
        assert keyword in str(exc.value)

    def test_overlap_equal_size_rejected(self):
        """重叠等于最大长度同样非法（否则切分会死循环）。"""
        with pytest.raises(chunking.ChunkConfigError):
            chunking.validate_config({"chunk_size": 100, "chunk_overlap": 100})

    def test_char_fallback_separator_appended(self):
        """末尾自动补字符级兜底分隔符，保证"最大长度"是硬上限。"""
        config = chunking.validate_config({"separators": ["\\n\\n", "。"]})
        assert config["separators"][-1] == ""
        assert config["separators"][0] == "\n\n"  # \\n 转义被还原

    def test_loose_input_normalized(self):
        """表单/JSON 传来的字符串数字、字符串布尔、"|" 简写都要能正确归一。"""
        config = chunking.validate_config({
            "separators": "\\n\\n|。",
            "chunk_size": "500",
            "chunk_overlap": "50",
            "keep_separator": "true",
            "filter_whitespace": "0",
        })
        assert config["separators"][:2] == ["\n\n", "。"]
        assert config["chunk_size"] == 500
        assert config["chunk_overlap"] == 50
        assert config["keep_separator"] is True
        assert config["filter_whitespace"] is False

    def test_default_config_is_valid(self):
        config = chunking.validate_config(None)
        assert config["chunk_size"] == 800
        assert config["chunk_overlap"] == 100
        assert config["strategy"] == "general"


class TestChunkText:
    """切分行为"""

    def test_splits_by_configured_separator(self):
        text = "段落一。\n\n段落二。\n\n段落三。"
        items = chunking.chunk_text(text, {
            "separators": ["\n\n"], "chunk_size": 5, "chunk_overlap": 0,
        })
        assert [item["content"] for item in items] == ["段落一。", "段落二。", "段落三。"]

    def test_falls_back_when_high_priority_separator_insufficient(self):
        """高优先级分隔符切不动时继续往后回退，且不超过最大长度。"""
        text = "x" * 95  # 没有 \n\n，必须回退到字符级
        items = chunking.chunk_text(text, {
            "separators": ["\n\n"], "chunk_size": 30, "chunk_overlap": 5,
        })
        assert [item["length"] for item in items] == [30, 30, 30, 20]
        assert all(item["length"] <= 30 for item in items)

    def test_keep_separator(self):
        text = "aa|||bb|||cc"
        kept = chunking.chunk_text(text, {
            "separators": ["|||"], "chunk_size": 3, "chunk_overlap": 0, "keep_separator": True,
        })
        dropped = chunking.chunk_text(text, {
            "separators": ["|||"], "chunk_size": 3, "chunk_overlap": 0, "keep_separator": False,
        })
        assert kept[1]["content"].startswith("|||")
        assert all("|||" not in item["content"] for item in dropped)

    def test_filter_whitespace(self):
        text = "正文\n\n   \n\n结尾"
        items = chunking.chunk_text(text, {
            "separators": ["\n\n"], "chunk_size": 2, "chunk_overlap": 0,
            "filter_whitespace": True,
        })
        assert [item["content"] for item in items] == ["正文", "结尾"]

        # 关闭过滤：空白不被 strip，分块文本保留原文空白
        kept = chunking.chunk_text(text, {
            "separators": ["\n\n"], "chunk_size": 2, "chunk_overlap": 0,
            "filter_whitespace": False,
        })
        assert kept
        assert any(item["content"] != item["content"].strip() for item in kept)

    @pytest.mark.parametrize("policy, expected", [
        ("keep", [10, 10, 1]),
        ("drop", [10, 10]),
        ("merge", [10, 11]),
    ])
    def test_short_chunk_policy(self, policy, expected):
        text = "A" * 20 + "|||" + "B"
        items = chunking.chunk_text(text, {
            "separators": ["|||"], "chunk_size": 10, "chunk_overlap": 0,
            "min_length": 3, "short_chunk_policy": policy,
        })
        assert [item["length"] for item in items] == expected
        assert [item["chunk_index"] for item in items] == list(range(len(expected)))

    def test_token_length_mode(self):
        text = "你好世界，这是一个测试句子。"
        items = chunking.chunk_text(text, {
            "chunk_size": 6, "chunk_overlap": 0, "length_mode": "token",
        })
        assert items, "token 口径也应能切出分块"
        for item in items:
            assert item["length"] == chunking.token_length(item["content"])
            assert item["length"] <= 6
        # 字符数与 token 数口径不同，两者都要提供
        assert any(item["length"] != item["chars"] for item in items)

    def test_blank_text_returns_empty(self):
        assert chunking.chunk_text("   \n\n  ", None) == []

    def test_config_hash_changes_with_config(self):
        a = chunking.validate_config({"chunk_size": 500})
        b = chunking.validate_config({"chunk_size": 501})
        assert chunking.config_hash(a) == chunking.config_hash(
            chunking.validate_config({"chunk_size": 500}))
        assert chunking.config_hash(a) != chunking.config_hash(b)


class TestTextParserCompat:
    """text_parser 旧入口必须与分块引擎给出相同结果（防止两套逻辑漂移）"""

    def test_parse_and_split_matches_chunking(self, tmp_path):
        import text_parser

        path = tmp_path / "compat.txt"
        path.write_text("第一段。\n\n第二段。\n\n第三段。", encoding="utf-8")

        legacy = text_parser.parse_and_split(str(path), chunk_size=5, chunk_overlap=0)
        engine = chunking.chunk_document(str(path), {"chunk_size": 5, "chunk_overlap": 0})

        assert [item["content"] for item in legacy] == [item["content"] for item in engine]
        assert all("length" in item for item in legacy)


# ── 接口层 ──────────────────────────────────────────────

def _create_kb(client, headers, name, **extra):
    payload = {"name": name, "description": "分块测试库"}
    payload.update(extra)
    resp = client.post("/api/kb", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]


def _upload(client, headers, kb_id, filename="chunk.txt", content=b"hello"):
    return client.post(
        "/api/documents/upload",
        data={"kb_id": kb_id},
        files={"file": (filename, content, "text/plain")},
        headers=headers,
    )


class TestChunkConfigAPI:
    """知识库分块配置：创建 / 查看 / 修改"""

    def test_create_kb_with_custom_config(self, client, auth_headers):
        config = {
            "separators": ["|||"],
            "chunk_size": 123,
            "chunk_overlap": 12,
            "length_mode": "char",
            "keep_separator": False,
            "filter_whitespace": True,
            "short_chunk_policy": "drop",
            "min_length": 2,
        }
        kb = _create_kb(client, auth_headers, "自定义分块库", chunk_config=config)
        assert kb["chunk_config"]["chunk_size"] == 123
        assert kb["chunk_config"]["short_chunk_policy"] == "drop"
        assert kb["chunk_size"] == 123  # 扁平字段同步，旧前端展示不变

        resp = client.get(f"/api/kb/{kb['kb_id']}/chunk-config", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["strategy"] == "general"
        assert data["config"]["chunk_size"] == 123
        assert data["config"]["separators"] == ["|||", ""]
        assert data["doc_count"] == 0
        assert data["config_hash"]

    def test_create_kb_with_invalid_config_returns_400(self, client, auth_headers):
        resp = client.post("/api/kb", json={
            "name": "非法配置库",
            "chunk_config": {"chunk_size": 100, "chunk_overlap": 100},
        }, headers=auth_headers)
        assert resp.status_code == 400
        assert "必须小于分块最大长度" in resp.json()["detail"]["message"]

    def test_update_config_without_documents(self, client, auth_headers):
        kb = _create_kb(client, auth_headers, "待改配置库")
        resp = client.put(f"/api/kb/{kb['kb_id']}/chunk-config", json={
            "chunk_config": {"chunk_size": 256, "chunk_overlap": 16, "separators": ["\n\n", "。"]},
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["data"]["chunk_config"]["chunk_size"] == 256

        again = client.get(f"/api/kb/{kb['kb_id']}/chunk-config", headers=auth_headers)
        assert again.json()["data"]["config"]["chunk_size"] == 256

    def test_update_config_with_documents_is_blocked(self, client, auth_headers):
        """已有文档时不允许直接改配置（避免新旧索引混用），并给出新旧对比。"""
        kb = _create_kb(client, auth_headers, "有文档库", chunk_size=400, chunk_overlap=40)
        assert _upload(client, auth_headers, kb["kb_id"]).status_code == 200

        resp = client.put(f"/api/kb/{kb['kb_id']}/chunk-config", json={
            "chunk_config": {"chunk_size": 200, "chunk_overlap": 20},
        }, headers=auth_headers)
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["doc_count"] == 1
        assert detail["old_config"]["chunk_size"] == 400
        assert detail["new_config"]["chunk_size"] == 200

        # 配置未被改动
        current = client.get(f"/api/kb/{kb['kb_id']}/chunk-config", headers=auth_headers)
        assert current.json()["data"]["config"]["chunk_size"] == 400


class TestChunkPreviewAPI:
    """分块预览接口"""

    def test_preview_text_with_override(self, client, auth_headers):
        kb = _create_kb(client, auth_headers, "预览库")
        resp = client.post(f"/api/kb/{kb['kb_id']}/chunk-preview", json={
            "text": "第一段。\n\n第二段。\n\n第三段。",
            "chunk_config": {"separators": ["\n\n"], "chunk_size": 5, "chunk_overlap": 0},
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_chunks"] == 3
        assert data["overridden"] is True
        assert data["items"][0]["content"] == "第一段。"
        assert data["items"][0]["length"] == 4
        assert data["items"][0]["chars"] == 4

    def test_preview_without_override_uses_kb_config(self, client, auth_headers):
        kb = _create_kb(client, auth_headers, "预览用库配置", chunk_size=6, chunk_overlap=0)
        resp = client.post(f"/api/kb/{kb['kb_id']}/chunk-preview", json={
            "text": "甲" * 20,
        }, headers=auth_headers)
        data = resp.json()["data"]
        assert data["overridden"] is False
        assert data["config"]["chunk_size"] == 6
        assert data["total_chunks"] == 4  # 20 / 6 → 4 块

    def test_preview_same_config_not_marked_overridden(self, client, auth_headers):
        """前端回传与知识库完全相同的配置时，不应被标记成"临时覆盖"。"""
        kb = _create_kb(client, auth_headers, "回传同配置", chunk_size=300, chunk_overlap=30)
        resp = client.post(f"/api/kb/{kb['kb_id']}/chunk-preview", json={
            "text": "内容" * 60,
            "chunk_config": kb["chunk_config"],
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["data"]["overridden"] is False

    def test_preview_invalid_override_returns_400(self, client, auth_headers):
        kb = _create_kb(client, auth_headers, "预览非法参数")
        resp = client.post(f"/api/kb/{kb['kb_id']}/chunk-preview", json={
            "text": "内容", "chunk_config": {"chunk_size": 10, "chunk_overlap": 10},
        }, headers=auth_headers)
        assert resp.status_code == 400
        assert "必须小于分块最大长度" in resp.json()["detail"]["message"]

    def test_preview_requires_text_or_doc(self, client, auth_headers):
        kb = _create_kb(client, auth_headers, "预览缺参数")
        resp = client.post(f"/api/kb/{kb['kb_id']}/chunk-preview", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_preview_unknown_document_returns_404(self, client, auth_headers):
        kb = _create_kb(client, auth_headers, "预览缺文档")
        resp = client.post(f"/api/kb/{kb['kb_id']}/chunk-preview", json={
            "doc_id": "not-exist",
        }, headers=auth_headers)
        assert resp.status_code == 404


class TestPreviewMatchesIngest:
    """验收场景 1：预览结果与实际入库结果必须一致"""

    def test_document_preview_defaults_to_kb_config(self, client, auth_headers):
        """文档预览不再使用固定 800/100，而是知识库当前配置。"""
        kb = _create_kb(client, auth_headers, "预览一致性A", chunk_size=120, chunk_overlap=0)
        upload = _upload(client, auth_headers, kb["kb_id"], filename="a.txt", content=("甲" * 500).encode())
        doc_id = upload.json()["data"]["doc_id"]

        resp = client.post(f"/api/documents/{doc_id}/preview-chunks", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["chunk_size"] == 120
        assert data["chunk_overlap"] == 0
        assert data["overridden"] is False
        assert data["total_chunks"] == 5  # 500 / 120 → 5 块

    def test_preview_matches_ingest_end_to_end(self, client, auth_headers, sample_kb, fake_bge):
        """上传 → 预览 → 真实执行 Worker → 逐块比对入库结果。"""
        fake_bge()  # 提供离线向量，Worker 才能跑完整链路
        content = ("第一段内容。" * 20 + "\n\n" + "第二段内容。" * 20 + "\n\n短").encode("utf-8")
        upload = _upload(client, auth_headers, sample_kb["kb_id"],
                         filename="consistency.txt", content=content)
        assert upload.status_code == 200, upload.text
        doc_id = upload.json()["data"]["doc_id"]
        task_id = upload.json()["data"]["task_id"]

        preview = client.post(f"/api/kb/{sample_kb['kb_id']}/chunk-preview",
                              json={"doc_id": doc_id}, headers=auth_headers)
        assert preview.status_code == 200
        preview_items = preview.json()["data"]["items"]
        assert preview_items

        # 直接执行 Celery 任务的函数体（投递已被 conftest 替换，不会真的发到 Redis）
        import tasks
        result = tasks.process_upload_task(task_id)
        assert result["status"] == "completed", result

        rows = database.get_chunks_for_doc(doc_id)
        assert [row["chunk_index"] for row in rows] == [item["chunk_index"] for item in preview_items]
        assert [row["content"] for row in rows] == [item["content"] for item in preview_items]

        # 任务里冻结的配置应与预览所用配置完全一致
        task_row = database.get_upload_task(task_id)
        assert task_row["chunk_config"] == chunking.canonical_json(
            chunking.config_from_kb(database.get_kb(sample_kb["kb_id"])))
