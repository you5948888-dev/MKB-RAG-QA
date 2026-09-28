"""Ch13：pytest 共享 fixture（离线集成测试基础设施）。

设计目标：不启动真实 uvicorn、不连真实 Milvus / BGE / LLM，也不向 Redis 投递消息，
仅用 FastAPI TestClient + 临时数据库验证「接口契约」与「业务状态机」。

两个容易踩的坑：

1. 只改 `config.CHAT_DB_PATH` 不够。
   `from config import CHAT_DB_PATH` 会在导入方（database / auth / tasks / 路由）
   各保存一份字符串引用，所以必须逐个模块替换模块级变量，测试库才会真正生效。

2. 外部依赖必须显式替身。
   Milvus 的启动初始化与删除操作、BGE / LLM 调用、Celery 的 delay 投递
   在本文件中被统一替换；socket 层再兜底阻断非回环连接，避免误连真实服务。

注意：测试通过只证明离线接口与业务契约，不代表真实 Milvus / Redis / 模型服务已联调。
"""
import importlib
import os
import socket

import pytest
from fastapi.testclient import TestClient

# ── 测试环境强制关闭第三方观测上报 ───────────────────────
# 开发机 .env 里可能开着 LangSmith（LANGSMITH_ENABLED=true），测试必须与真实上报解耦：
# 在导入 config（它进而导入 langsmith_tracing）之前先把它关掉，用例绝不产生外发请求。
os.environ["LANGSMITH_ENABLED"] = "false"

# 默认管理员（auth.ensure_default_admin 在应用启动时创建）
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456"

# 这些模块都用 `from config import X` 复制了路径引用，必须逐个替换
_PATH_PATCH_MODULES = (
    "database",
    "auth",
    "tasks",
    "routes.document_routes",
    "routes.qa_routes",
)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


# ── 网络保护 ────────────────────────────────────────────

def _is_internal_address(address) -> bool:
    """是否允许连接：回环地址与本地 socket 放行，其余（真实 Milvus/BGE/LLM）阻断。

    放行回环是必要的：Windows 上 asyncio 事件循环的唤醒连接、socketpair()
    都走 127.0.0.1。
    """
    if isinstance(address, (str, bytes)):
        return True  # AF_UNIX 等本地地址
    if isinstance(address, tuple) and address:
        host = str(address[0])
        return host in _LOOPBACK_HOSTS or host.startswith("127.")
    return True


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch):
    """阻断非回环连接，保证用例离线可重复（误连外部服务会立刻失败而非超时）。"""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(this, address, *args, **kwargs):
        if not _is_internal_address(address):
            raise RuntimeError(f"离线测试禁止外部网络连接: {address}")
        return real_connect(this, address, *args, **kwargs)

    def guarded_connect_ex(this, address, *args, **kwargs):
        if not _is_internal_address(address):
            raise RuntimeError(f"离线测试禁止外部网络连接: {address}")
        return real_connect_ex(this, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


# ── 临时数据目录 / 数据库 ───────────────────────────────

@pytest.fixture
def _use_test_db(tmp_path, monkeypatch):
    """为每个用例建立独立的数据库、上传目录与数据目录。

    返回路径字典，供需要检查落盘文件的用例使用。
    """
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "chat_memory.db"

    for module_name in _PATH_PATCH_MODULES:
        module = importlib.import_module(module_name)
        for attr, value in (
            ("CHAT_DB_PATH", str(db_path)),
            ("UPLOAD_DIR", str(upload_dir)),
            ("DATA_DIR", str(data_dir)),
        ):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, value)

    return {"data_dir": data_dir, "upload_dir": upload_dir, "db_path": db_path}


# ── 外部依赖替身 ────────────────────────────────────────

def _noop(*args, **kwargs):
    """空实现：用于替换所有会产生外部副作用的调用。"""
    return None


@pytest.fixture
def _isolate_external_services(monkeypatch):
    """Milvus 读写、Celery 投递全部替换为空实现。

    Milvus 替换后，应用启动的 `ensure_collection()` 与删除知识库时的向量清理
    都不会真的连接向量库；Celery 的 `delay` 被替换后也不会向 Redis 发消息。
    """
    import milvus_ops
    import tasks

    for name in (
        "ensure_collection",
        "ensure_index_question_collection",
        "upsert_vectors",
        "delete_vectors_by_pk",
        "delete_vectors_by_kb",
        "delete_vectors_for_doc",
        "upsert_index_question_vector",
        "delete_index_question_vector",
        "delete_index_question_vectors_for_kb",
        "delete_index_question_vectors_by_ids",
        "search_vectors",
        "search_index_questions",
    ):
        monkeypatch.setattr(milvus_ops, name, _noop)

    monkeypatch.setattr(tasks.process_upload_task, "delay", _noop)
    monkeypatch.setattr(tasks.process_index_build_task, "delay", _noop)


class _Chunk:
    """模仿 langchain 流式 chunk（生成器只用到 .content）。"""

    def __init__(self, content: str):
        self.content = content


class _Message:
    """模仿 langchain 的 AIMessage（同步问答只用到 .content）。"""

    def __init__(self, content: str):
        self.content = content


class RefusingLLM:
    """默认 LLM 替身：被调用即报错，提醒用例显式声明期望结果。"""

    def stream(self, messages):
        raise AssertionError("离线测试未提供 LLM 替身，请使用 fake_llm fixture")

    def invoke(self, messages):
        raise AssertionError("离线测试未提供 LLM 替身，请使用 fake_llm fixture")


class FakeLLM:
    """离线 LLM 替身：可返回固定回答，或按 token 列表模拟流式输出。

    answers: [(提示词关键字, 回复), ...] —— 按提示词内容匹配，用于让
    问题改写 / 多查询 / HyDE / 最终回答在一次问答里各自返回不同结果
    （这三个能力是并发调用的，不能靠调用顺序区分）。
    """

    def __init__(self, tokens=None, answer="", stream_error=None, invoke_error=None, answers=None):
        self.tokens = list(tokens or [])
        self.answer = answer
        self.stream_error = stream_error
        self.invoke_error = invoke_error
        self.answers = list(answers or [])
        self.stream_calls = []
        self.invoke_calls = []

    def stream(self, messages):
        self.stream_calls.append(messages)
        # 先吐出已排队的 token，再抛错：用于模拟「已输出部分回答后模型中断」
        for token in self.tokens:
            yield _Chunk(token)
        if self.stream_error is not None:
            raise self.stream_error

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        if self.invoke_error is not None:
            raise self.invoke_error
        prompt = "\n".join(str(getattr(msg, "content", "")) for msg in messages)
        for hint, reply in self.answers:
            if hint in prompt:
                return _Message(reply)
        return _Message(self.answer)


@pytest.fixture(autouse=True)
def _forbid_external_models(monkeypatch):
    """BGE 与 LLM 默认拒绝执行：需要时用 fake_llm / fake_bge 显式提供结果。"""
    import embedding
    import rag_stream

    def _refuse(*args, **kwargs):
        raise AssertionError("离线测试不应调用真实模型服务，请显式提供替身")

    monkeypatch.setattr(embedding.bge_model, "embed", _refuse)
    monkeypatch.setattr(embedding.bge_model, "rerank", _refuse)

    refusing = RefusingLLM()
    monkeypatch.setattr(embedding, "llm_client", refusing)
    monkeypatch.setattr(rag_stream, "llm_client", refusing)


@pytest.fixture
def fake_llm(monkeypatch, _forbid_external_models):
    """安装离线 LLM 替身，返回替身对象以便断言调用情况。

    用法：fake_llm(tokens=["你", "好"]) / fake_llm(answer="回答")
    """
    def _install(tokens=None, answer="", stream_error=None, invoke_error=None, answers=None):
        import embedding
        import rag_stream

        fake = FakeLLM(tokens=tokens, answer=answer, stream_error=stream_error,
                       invoke_error=invoke_error, answers=answers)
        # rag_pipeline 在函数内 `from embedding import llm_client`，替换模块属性即可；
        # rag_stream 在导入时复制了引用，需要同时替换它自己的模块属性。
        monkeypatch.setattr(embedding, "llm_client", fake)
        monkeypatch.setattr(rag_stream, "llm_client", fake)
        return fake

    return _install


@pytest.fixture
def fake_bge(monkeypatch, _forbid_external_models):
    """显式提供 BGE 返回值（dense 向量 / sparse 向量 / rerank 结果）。"""
    def _install(embed_result=None, rerank_result=None):
        import embedding

        if embed_result is None:
            embed_result = {"dense": [[0.1] * 8], "sparse": {1: 0.5}}
        monkeypatch.setattr(embedding.bge_model, "embed", lambda text: embed_result)
        monkeypatch.setattr(
            embedding.bge_model, "rerank",
            lambda query, contents, top_k=20: rerank_result or [],
        )
        return embed_result

    return _install


# ── 应用与业务 fixture ──────────────────────────────────

@pytest.fixture
def client(_use_test_db, _isolate_external_services):
    """进程内启动应用。

    进入 `with TestClient(app)` 才执行应用启动事件：在临时库建表、初始化认证表、
    创建默认 admin；不在 fixture 中重复初始化。
    """
    from app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    """以默认管理员登录，返回 API Key 认证头。

    用 API Key 而非 Cookie：注册新用户的用例会覆盖 Cookie，显式 Header 更独立。
    """
    resp = client.post("/api/auth/login", json={
        "username": ADMIN_USERNAME,
        "password": ADMIN_PASSWORD,
    })
    assert resp.status_code == 200, resp.text

    me = client.get("/api/auth/me")
    assert me.status_code == 200, me.text
    api_key = me.json()["data"]["api_key"]
    return {"X-API-Key": api_key}


@pytest.fixture
def set_upload_task_fields():
    """直接改库模拟 Worker 走过的中间态（parsing / 超时的 updated_at 等）。"""
    def _set(task_id: str, **fields):
        import database

        columns = ", ".join(f"{key}=?" for key in fields)
        conn = database.get_connection()
        with conn:
            conn.execute(
                f"UPDATE upload_tasks SET {columns} WHERE task_id=?",
                (*fields.values(), task_id),
            )
        conn.close()

    return _set


@pytest.fixture
def sample_kb(client, auth_headers):
    """按需创建知识库：只有声明依赖它的用例才会建库。"""
    resp = client.post("/api/kb", json={
        "name": "测试知识库",
        "description": "pytest 专用",
        "chunk_size": 800,
        "chunk_overlap": 100,
    }, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]
