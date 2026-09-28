"""LangSmith 开关式接入的离线用例（不联网、不上报任何数据）。

覆盖点：
1. 默认关闭 / 开启但缺 Key → 一律不开启，且不写任何环境变量；
2. 正常开启 → 写入 langsmith 与 legacy 两套变量、项目名、采样率；
3. 面板状态里 Key 只有掩码，绝不含明文（安全红线）；
4. 关闭时清理"自己写过的"变量，避免上次开启的残留继续生效；
5. `traced_llm` 对业务零风险：未开启、替身不支持 with_config、挂载异常 → 一律原样返回。
"""
import langsmith_tracing


def _enabled_env(**overrides) -> dict:
    env = {
        "LANGSMITH_ENABLED": "true",
        "LANGSMITH_API_KEY": "lsv2_pt_0123456789abcdef_test_key",
        "LANGSMITH_PROJECT": "RAG实战",
        "LANGSMITH_TRACING_SAMPLING_RATE": "1",
    }
    env.update(overrides)
    return env


# ── 开关与配置 ──────────────────────────────────────────


class TestConfigure:
    def test_disabled_by_default(self):
        """没有任何变量时不开启，也不写环境变量。"""
        env: dict = {}
        status = langsmith_tracing.configure(env)
        assert status["enabled"] is False
        assert "未开启" in status["reason"]
        assert env == {}

    def test_enabled_without_api_key_is_rejected(self):
        """开了开关但没 Key：不开启，并给出明确原因。"""
        env = {"LANGSMITH_ENABLED": "true"}
        status = langsmith_tracing.configure(env)
        assert status["enabled"] is False
        assert "LANGSMITH_API_KEY" in status["reason"]
        assert "LANGSMITH_TRACING" not in env

    def test_enabled_writes_langsmith_env(self):
        """正常开启：写入 langsmith 变量与 legacy 兼容变量。"""
        env = _enabled_env()
        status = langsmith_tracing.configure(env)
        assert status["enabled"] is True
        assert status["reason"] == ""
        assert status["project"] == "RAG实战"
        assert env["LANGSMITH_TRACING"] == "true"
        assert env["LANGSMITH_API_KEY"] == "lsv2_pt_0123456789abcdef_test_key"
        assert env["LANGSMITH_PROJECT"] == "RAG实战"
        assert env["LANGSMITH_TRACING_SAMPLING_RATE"] == "1"
        # 兼容旧变量名（部分 langchain 版本只认 LANGCHAIN_*）
        assert env["LANGCHAIN_TRACING_V2"] == "true"
        assert env["LANGCHAIN_API_KEY"] == env["LANGSMITH_API_KEY"]
        assert env["LANGCHAIN_PROJECT"] == "RAG实战"

    def test_default_project_is_rag_course_name(self):
        """未指定项目名时用默认项目名，便于在 LangSmith 里一眼认出。"""
        env = _enabled_env(LANGSMITH_PROJECT="")
        status = langsmith_tracing.configure(env)
        assert status["project"] == langsmith_tracing.DEFAULT_PROJECT == "RAG实战"
        assert status["project_url"].endswith("/projects/p/RAG实战")

    def test_sampling_rate_is_clamped_and_invalid_ignored(self):
        """采样率：越界夹紧到 [0,1]，非法值当未配置（全量）。"""
        assert langsmith_tracing.configure(_enabled_env(LANGSMITH_TRACING_SAMPLING_RATE="5"))["sampling_rate"] == 1.0
        assert langsmith_tracing.configure(_enabled_env(LANGSMITH_TRACING_SAMPLING_RATE="-1"))["sampling_rate"] == 0.0
        assert langsmith_tracing.configure(_enabled_env(LANGSMITH_TRACING_SAMPLING_RATE="0.1"))["sampling_rate"] == 0.1
        assert langsmith_tracing.configure(_enabled_env(LANGSMITH_TRACING_SAMPLING_RATE="abc"))["sampling_rate"] == 1.0

    def test_status_masks_api_key(self):
        """安全红线：状态里绝不能出现明文 Key。"""
        secret = "lsv2_pt_0123456789abcdef_test_key"
        status = langsmith_tracing.configure(_enabled_env(LANGSMITH_API_KEY=secret))
        assert status["api_key"] == "lsv2_pt_****"
        assert secret not in str(status)

    def test_disabling_clears_owned_vars(self):
        """先开启再关闭：本模块写过的变量必须清掉，否则会继续上报。"""
        env = _enabled_env()
        langsmith_tracing.configure(env)
        assert env.get("LANGSMITH_TRACING") == "true"

        status = langsmith_tracing.configure(env)  # 此时 LANGSMITH_ENABLED 仍是 true
        assert status["enabled"] is True

        env["LANGSMITH_ENABLED"] = "false"
        status = langsmith_tracing.configure(env)
        assert status["enabled"] is False
        for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"):
            assert key not in env


# ── traced_llm：对业务零风险 ────────────────────────────


class _FakeClient:
    """替身：可记录 with_config 的入参，模拟 langchain 的 RunnableBinding 行为。"""

    def __init__(self, support_config=True, raise_on_config=False):
        self.support_config = support_config
        self.raise_on_config = raise_on_config
        self.config = None

    def with_config(self, config):
        if self.raise_on_config:
            raise RuntimeError("模拟挂载失败")
        self.config = config
        return self

    def invoke(self, messages):
        return "ok"


class TestTracedLlm:
    def test_returns_same_client_when_disabled(self, monkeypatch):
        """未开启追踪：原样返回同一个对象，零开销、零行为变化。"""
        monkeypatch.setitem(langsmith_tracing._STATUS, "enabled", False)
        client = _FakeClient()
        assert langsmith_tracing.traced_llm(client, mode="basic") is client
        assert client.config is None

    def test_client_without_with_config_is_returned_as_is(self, monkeypatch):
        """替身/非 langchain 客户端没有 with_config：不能因为观测而报错。"""
        monkeypatch.setitem(langsmith_tracing._STATUS, "enabled", True)
        client = _FakeClient(support_config=False)
        assert langsmith_tracing.traced_llm(client, mode="stream") is client

    def test_attaches_metadata_tags_and_trace_id(self, monkeypatch):
        """开启时挂上 metadata/tags，并带上本地 trace_id 便于两边对照。"""
        monkeypatch.setitem(langsmith_tracing._STATUS, "enabled", True)
        monkeypatch.setattr(langsmith_tracing, "_current_trace_id", lambda: "trace-123")
        client = _FakeClient()

        bound = langsmith_tracing.traced_llm(client, mode="advanced", ability="hyde")
        assert bound is client
        assert client.config["metadata"]["trace_id"] == "trace-123"
        assert client.config["metadata"]["mode"] == "advanced"
        assert client.config["metadata"]["ability"] == "hyde"
        assert "rag-tutorial" in client.config["tags"]
        assert "hyde" in client.config["tags"]

    def test_metadata_skips_empty_values(self, monkeypatch):
        """空值不进 metadata，避免在 LangSmith 里出现一堆空字段。"""
        monkeypatch.setitem(langsmith_tracing._STATUS, "enabled", True)
        monkeypatch.setattr(langsmith_tracing, "_current_trace_id", lambda: "")
        client = _FakeClient()
        langsmith_tracing.traced_llm(client, ability="", mode="basic", task_id=None)
        assert client.config["metadata"] == {"mode": "basic"}
        assert "trace_id" not in client.config["metadata"]

    def test_attach_failure_never_breaks_qa(self, monkeypatch):
        """挂载异常必须被吞掉并原样返回客户端（观测失败不影响主流程）。"""
        monkeypatch.setitem(langsmith_tracing._STATUS, "enabled", True)
        monkeypatch.setattr(langsmith_tracing, "_current_trace_id", lambda: "t-1")
        client = _FakeClient(raise_on_config=True)
        assert langsmith_tracing.traced_llm(client, mode="basic") is client


# ── 面板可见性 ──────────────────────────────────────────


class TestPanelIntegration:
    def test_overview_exposes_langsmith_status(self, client, auth_headers):
        """观测面板能拿到 LangSmith 状态（测试环境强制关闭 → reason 有值）。"""
        resp = client.get("/api/observability/overview?with_dependencies=false",
                          headers=auth_headers)
        assert resp.status_code == 200
        info = resp.json()["data"]["langsmith"]
        assert set(info) >= {"enabled", "reason", "project", "api_key", "project_url", "sampling_rate"}
        assert info["enabled"] is False
        assert info["reason"]
        assert info["project"] == "RAG实战"
