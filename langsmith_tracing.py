"""LangSmith LLM 调用追踪（可选的第三方观测，**默认关闭**）。

与自研可观测性的分工（两者互补，不是替代）：

| | 自研 `rag_traces` / `rag_spans` | LangSmith |
|---|---|---|
| 覆盖 | 问答 10 阶段 + 文档 7 阶段**全链路**（含向量化、Milvus 召回、融合、Rerank、父块回查） | **只有 LLM 调用内部**（prompt 原文、回复、token 用量、流式逐块时序、重试） |
| 数据位置 | 本地 SQLite | LangSmith 云端（除非把 Endpoint 指向自托管地址） |

因此做成"开关式"接入，三条硬约束：
1. **默认关闭**：`LANGSMITH_ENABLED` 非真值时，不写任何环境变量、不建客户端、不产生网络请求；
2. **零业务改造**：`embedding.llm_client` 是 `langchain_openai.ChatOpenAI`，只要本进程环境里有
   langsmith 变量，LangChain 会自动上报（含流式），业务代码不需要改；
3. **绝不影响主流程**：上报在 langsmith 的后台线程批量发送，失败只丢观测数据
   （对应需求六"观测记录失败不能影响主流程"）。

【安全】API Key 只从 `.env` 读：不写进代码、不进日志、不进观测详情；
`status()` 只返回掩码（`lsv2_pt_****`），可以安全地暴露给前端面板。
【注意】开启后，用户问题、检索到的上下文（父块正文＝文档原文）与回答会**上传到 LangSmith**，
私有知识库场景请先确认这一数据出境行为可接受；若不能，把 Endpoint 换自托管服务或保持关闭。
"""
import logging
import os

logger = logging.getLogger("rag.langsmith")

TRUTHY = {"1", "true", "yes", "on", "是"}

DEFAULT_PROJECT = "RAG实战"
DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
DEFAULT_UI = "https://smith.langchain.com"

# 本模块负责写入的环境变量（关闭时会清理"自己写过的"项，不动外部已有设置）
_MANAGED_KEYS = (
    "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT", "LANGSMITH_TRACING_SAMPLING_RATE",
    # 兼容旧变量名：部分 langchain 版本只认 LANGCHAIN_*
    "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT",
)
_OWNED_KEYS: set[str] = set()


def _as_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUTHY


def _as_rate(value) -> float | None:
    """采样率：合法为 0~1；非法或未配置返回 None（表示全量）。"""
    if value is None or str(value).strip() == "":
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate < 0:
        return 0.0
    if rate > 1:
        return 1.0
    return rate


def mask_key(key: str) -> str:
    """把 API Key 掩码成 `lsv2_pt_****`（面板展示用，永不回显完整值）。"""
    if not key:
        return ""
    return f"{key[:8]}****" if len(key) > 12 else "****"


def configure(environ=None) -> dict:
    """按环境变量决定是否开启追踪，并写入 langsmith 需要的变量。

    `environ` 仅用于测试注入，默认操作 `os.environ`。
    返回状态字典（不含明文 Key），可安全返回给前端。
    """
    env = os.environ if environ is None else environ

    enabled = _as_bool(env.get("LANGSMITH_ENABLED"))
    api_key = (env.get("LANGSMITH_API_KEY") or "").strip()
    project = (env.get("LANGSMITH_PROJECT") or "").strip() or DEFAULT_PROJECT
    endpoint = (env.get("LANGSMITH_ENDPOINT") or "").strip()
    ui_url = (env.get("LANGSMITH_UI_URL") or "").strip() or DEFAULT_UI
    rate = _as_rate(env.get("LANGSMITH_TRACING_SAMPLING_RATE"))

    reason = "未开启（LANGSMITH_ENABLED 非真值）"
    if enabled and not api_key:
        enabled = False
        reason = "已设置 LANGSMITH_ENABLED 但缺少 LANGSMITH_API_KEY，未开启"

    if enabled:
        env["LANGSMITH_TRACING"] = "true"
        env["LANGSMITH_API_KEY"] = api_key
        env["LANGSMITH_PROJECT"] = project
        if endpoint:
            env["LANGSMITH_ENDPOINT"] = endpoint
        if rate is not None:
            # 归一化写法：1.0 → "1"、0.1 → "0.1"（:g 会去掉多余的小数位）
            env["LANGSMITH_TRACING_SAMPLING_RATE"] = f"{rate:g}"
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGCHAIN_API_KEY"] = api_key
        env["LANGCHAIN_PROJECT"] = project
        if endpoint:
            env["LANGCHAIN_ENDPOINT"] = endpoint
        _OWNED_KEYS.update(_MANAGED_KEYS)
        logger.info("LangSmith 追踪已开启：项目=%s，采样率=%s，Endpoint=%s",
                    project, rate if rate is not None else 1.0, endpoint or DEFAULT_ENDPOINT)
    else:
        # 关闭时清理"本模块自己写进去的"变量，避免上次开启的残留继续生效
        for key in list(_OWNED_KEYS):
            env.pop(key, None)
            _OWNED_KEYS.discard(key)

    return {
        "enabled": enabled,
        "reason": "" if enabled else reason,
        "project": project,
        "endpoint": endpoint or DEFAULT_ENDPOINT,
        "sampling_rate": rate if rate is not None else (1.0 if enabled else None),
        "api_key": mask_key(api_key),
        "project_url": f"{ui_url.rstrip('/')}/projects/p/{project}",
    }


_STATUS: dict = configure()


def is_enabled() -> bool:
    """当前进程是否开启了 LangSmith 追踪。"""
    return bool(_STATUS.get("enabled"))


def status() -> dict:
    """当前状态（不含明文 Key），给观测面板用。"""
    return dict(_STATUS)


def _current_trace_id() -> str:
    """取当前请求的本地 trace_id（同一 context 内），用于两边记录互相对照。"""
    try:
        import observability

        recorder = observability.current()
        return getattr(recorder, "trace_id", "") or ""
    except Exception:  # pragma: no cover - 观测模块异常不能影响问答
        return ""


def traced_llm(client, **metadata):
    """给 LLM 客户端挂上 LangSmith 的 metadata / tags，返回可调用对象。

    关键点：
    - 传入的是**调用点自己持有的客户端对象**，这样测试里替换掉的替身也能正常传入
      （替身没有 `with_config`，会原样返回，不影响离线用例）；
    - 未开启追踪、客户端不支持、或挂载异常时，一律**原样返回**，绝不影响问答主流程。
    """
    if client is None or not is_enabled():
        return client
    with_config = getattr(client, "with_config", None)
    if not callable(with_config):
        return client

    meta = {key: value for key, value in metadata.items() if value not in (None, "")}
    trace_id = meta.get("trace_id") or _current_trace_id()
    if trace_id:
        meta["trace_id"] = trace_id
    tags = ["rag-tutorial"]
    for key in ("mode", "ability"):
        if meta.get(key):
            tags.append(str(meta[key]))
    try:
        return with_config({"metadata": meta, "tags": tags, "run_name": meta.get("ability") or None})
    except Exception as exc:  # pragma: no cover - 观测失败不能影响主流程
        logger.warning("LangSmith metadata 挂载失败（忽略，不影响问答）: %s", exc)
        return client
