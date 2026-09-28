"""
数据库层：SQLite 持久化存储。
本章新增 knowledge_bases 表，后续章节会逐步添加更多表。
"""
import json
import sqlite3
import uuid
from datetime import datetime, timedelta  # Ch10 新增 timedelta：僵尸任务清理需要计算 30 分钟前

from config import CHAT_DB_PATH


def get_connection() -> sqlite3.Connection:
    """获取数据库连接，查询结果使用 sqlite3.Row。"""
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── 增量迁移（幂等）──────────────────────────────────
# 只新增列、不动旧数据；列已存在时跳过，因此可重复执行。
# 新增列必须带默认值，保证旧库读出来的行也有合法取值。
_COLUMN_MIGRATIONS = (
    # Advanced RAG 第 1 期：知识库分块策略与完整配置
    ("knowledge_bases", "chunk_strategy", "TEXT DEFAULT 'general'"),
    ("knowledge_bases", "chunk_config", "TEXT DEFAULT ''"),
    # 上传任务冻结创建时的分块配置快照（避免排队期间改配置导致预览与入库不一致）
    ("upload_tasks", "chunk_config", "TEXT DEFAULT ''"),
    # Advanced RAG 第 2 期：文档实际索引配置指纹 + 父子块计数
    ("kb_documents", "chunk_config_hash", "TEXT DEFAULT ''"),
    ("kb_documents", "parent_count", "INTEGER DEFAULT 0"),
    ("kb_documents", "child_count", "INTEGER DEFAULT 0"),
    # Advanced RAG 第 2 期：kb_chunks 支持父子层级
    # chunk_role: parent（只落 SQLite，用于回查上下文）/ child（进向量库，可检索）
    ("kb_chunks", "parent_chunk_id", "TEXT DEFAULT ''"),
    ("kb_chunks", "chunk_role", "TEXT DEFAULT 'child'"),
    ("kb_chunks", "parent_index", "INTEGER DEFAULT -1"),
    ("kb_chunks", "sense_index", "INTEGER DEFAULT 0"),
    ("kb_chunks", "length", "INTEGER DEFAULT 0"),
    # Advanced RAG 第 2 期：任务类型与文档链路统计
    ("upload_tasks", "task_type", "TEXT DEFAULT 'upload'"),
    ("upload_tasks", "parent_count", "INTEGER DEFAULT 0"),
    ("upload_tasks", "child_count", "INTEGER DEFAULT 0"),
    ("upload_tasks", "embedded_count", "INTEGER DEFAULT 0"),
    ("upload_tasks", "retry_count", "INTEGER DEFAULT 0"),
    ("upload_tasks", "current_stage", "TEXT DEFAULT ''"),
)


def _migrate_columns(conn) -> None:
    """按需补齐后续迭代新增的列（旧数据库直接升级，无需重建、不丢数据）。"""
    for table, column, ddl in _COLUMN_MIGRATIONS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


# ── 建表 ──────────────────────────────────────────────

def init_db():
    """初始化数据库：创建本章需要的表。"""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            kb_id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT '',
            chunk_size INTEGER DEFAULT 800,
            chunk_overlap INTEGER DEFAULT 100,
            doc_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_documents (
            doc_id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            file_size INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(kb_id, filename)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Ch6 新增：对话会话 + 消息记录
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            title TEXT DEFAULT '新对话',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )
    """)
    # Ch8 新增：索引问题系统（2 张表）
    # source 字段区分来源：'manual'（手动创建）/ 'ai'（大模型自动生成）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_index_questions (
            index_id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            question TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            created_at TEXT NOT NULL
        )
    """)
    # 绑定关系表，通过 chunk_id 关联 kb_chunks，不再冗余存储 chunk_text / source
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_index_question_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            UNIQUE(index_id, chunk_id)
        )
    """)
    # Ch8 新增：7 个性能索引
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_kb_id ON kb_chunks(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON kb_chunks(doc_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_kb_id ON chat_sessions(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_index_questions_kb_id ON kb_index_questions(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_index_links_kb_id ON kb_index_question_links(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_index_links_chunk_id ON kb_index_question_links(chunk_id)")
    # Ch10 新增：上传任务表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_tasks (
            task_id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            doc_id TEXT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            status TEXT DEFAULT 'uploaded',
            progress INTEGER DEFAULT 0,
            step_message TEXT DEFAULT '',
            error_message TEXT DEFAULT '',
            chunk_size INTEGER DEFAULT 800,
            chunk_overlap INTEGER DEFAULT 100,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT
        )
    """)
    # Ch11 新增：索引构建任务表（自动生成索引问题的异步任务）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_build_tasks (
            task_id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            prompt TEXT DEFAULT '',
            status TEXT DEFAULT 'queued',
            progress INTEGER DEFAULT 0,
            step_message TEXT DEFAULT '',
            error_message TEXT DEFAULT '',
            total_chunks INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT
        )
    """)
    # Advanced RAG 第 3 期：知识库检索策略（Advanced RAG 各能力的开关与参数）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_retrieval_strategies (
            kb_id TEXT PRIMARY KEY,
            config TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    # Advanced RAG 第 4 期：全链路追踪（trace + span）
    # 【注意】观测数据不自动清理，由用户手动调用 /api/observability/traces/cleanup
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_traces (
            trace_id TEXT PRIMARY KEY,
            biz_type TEXT NOT NULL,
            kb_id TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            question TEXT DEFAULT '',
            rewritten_question TEXT DEFAULT '',
            expansions TEXT DEFAULT '[]',
            hyde_answer TEXT DEFAULT '',
            config_snapshot TEXT DEFAULT '{}',
            status TEXT DEFAULT 'running',
            error TEXT DEFAULT '',
            recall_empty INTEGER DEFAULT 0,
            degrade_count INTEGER DEFAULT 0,
            first_token_ms INTEGER,
            total_ms INTEGER,
            detail TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            finished_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            seq INTEGER NOT NULL DEFAULT 0,
            stage TEXT NOT NULL,
            started_at TEXT,
            duration_ms INTEGER DEFAULT 0,
            status TEXT DEFAULT 'ok',
            input_count INTEGER,
            output_count INTEGER,
            error TEXT DEFAULT '',
            detail TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_created ON rag_traces(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_biz ON rag_traces(biz_type, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace ON rag_spans(trace_id, id)")
    _migrate_columns(conn)  # 旧库补齐新增列
    conn.commit()
    conn.close()


# ── 知识库 CRUD ──────────────────────────────────────

def create_kb(name: str, description: str = "", chunk_size: int = 800, chunk_overlap: int = 100,
              chunk_strategy: str = "general", chunk_config: str = "") -> dict:
    """创建知识库，返回知识库字典。名称重复时返回 None。

    chunk_strategy / chunk_config：分块策略与完整配置（JSON 字符串，由路由层用
    chunking 校验并序列化后传入）；chunk_size / chunk_overlap 保留为扁平字段，
    旧前端与旧查询按原样读取，无需改动。
    """
    kb_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO knowledge_bases (kb_id, name, description, chunk_size, chunk_overlap, doc_count, chunk_count, chunk_strategy, chunk_config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (kb_id, name, description, chunk_size, chunk_overlap, 0, 0, chunk_strategy, chunk_config, now, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return None  # 当前实现统一转换完整性异常

    conn.close()
    return get_kb(kb_id)


def get_kb(kb_id: str) -> dict | None:
    """根据 ID 获取单个知识库。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM knowledge_bases WHERE kb_id = ?", (kb_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_kbs(keyword: str | None = None, page: int = 1, page_size: int = 10) -> dict:
    """获取知识库列表，按更新时间倒序，支持分页。

    【本练习】分页：SQL 用 LIMIT+OFFSET 只取当前页需要的一页数据，
    同时返回 total 供前端判断是否还有更多。
    参数：keyword 可选（按名称忽略大小写过滤）；page 第几页（从 1 起）；
    page_size 每页条数；传入 page_size=None 表示不分页（返回全部）。
    返回：{"items": [...], "total": int}
    """
    conn = get_connection()
    base_where = " WHERE lower(name) LIKE lower(?)" if keyword else ""
    params: tuple = (f"%{keyword}%",) if keyword else ()

    # 1) 查总数（分页需要知道总共有多少条，才能算是否还有下一页）
    total = conn.execute(
        f"SELECT COUNT(*) FROM knowledge_bases{base_where}", params
    ).fetchone()[0]

    # 2) 查当前页数据
    if page_size is not None:
        # LIMIT 每页条数，OFFSET 跳过前面 (page-1)*page_size 条
        offset = (page - 1) * page_size
        sql = (
            f"SELECT * FROM knowledge_bases{base_where}"
            " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        )
        rows = conn.execute(sql, (*params, page_size, offset)).fetchall()
    else:
        sql = f"SELECT * FROM knowledge_bases{base_where} ORDER BY updated_at DESC"
        rows = conn.execute(sql, params).fetchall()
    conn.close()

    return {"items": [dict(row) for row in rows], "total": total}


def update_kb_name(kb_id: str, new_name: str) -> dict | None:
    """修改知识库名称（UPDATE 语句）。

    【作业】SQLite 的 UPDATE：UPDATE 表 SET 列=新值 WHERE 条件。
    - 只更新 name 和 updated_at 两列，其它列不动；
    - WHERE kb_id = ? 精确定位要改的那一行；
    - 返回成功与否：改到 0 行说明该 ID 不存在 → 返回 None。
    """
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE knowledge_bases SET name = ?, updated_at = ? WHERE kb_id = ?",
        (new_name, now, kb_id),
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        return None
    return get_kb(kb_id)


# ── 知识库检索策略（Advanced RAG 第 3 期）───────────────

def get_retrieval_strategy(kb_id: str) -> str | None:
    """读取知识库检索策略（JSON 字符串）；从未配置过返回 None。

    容错：表缺失（老库未跑 init_db 就调用检索）时返回 None 并告警——
    读策略失败绝不能让检索主流程报错，此时行为等价于"未配置策略"。
    """
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT config FROM kb_retrieval_strategies WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        conn.close()
    except sqlite3.OperationalError as exc:
        print(f"[WARNING] 读取检索策略失败（将使用默认策略）: {exc}")
        return None
    return row["config"] if row else None


def upsert_retrieval_strategy(kb_id: str, config: str) -> None:
    """保存（或覆盖）知识库检索策略。"""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    conn.execute(
        "INSERT INTO kb_retrieval_strategies (kb_id, config, updated_at) VALUES (?, ?, ?)"
        " ON CONFLICT(kb_id) DO UPDATE SET config = excluded.config, updated_at = excluded.updated_at",
        (kb_id, config, now),
    )
    conn.commit()
    conn.close()


def delete_retrieval_strategy(kb_id: str) -> bool:
    """删除知识库策略（恢复默认用）。"""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM kb_retrieval_strategies WHERE kb_id = ?", (kb_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def update_kb_chunk_config(kb_id: str, chunk_strategy: str, chunk_config: str,
                           chunk_size: int, chunk_overlap: int) -> dict | None:
    """更新知识库的分块配置（仅在无文档时由路由层调用）。

    同时同步扁平字段 chunk_size / chunk_overlap，保证旧前端读取到的展示值一致。
    """
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE knowledge_bases SET chunk_strategy = ?, chunk_config = ?, chunk_size = ?, "
        "chunk_overlap = ?, updated_at = ? WHERE kb_id = ?",
        (chunk_strategy, chunk_config, chunk_size, chunk_overlap, now, kb_id),
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        return None
    return get_kb(kb_id)


def delete_kb(kb_id: str) -> bool:
    """删除知识库，返回是否成功。Ch9 新增：同步清理 user_kbs 绑定。级联删除索引问题及其绑定。Ch10 新增：清理 upload_tasks。Ch11：非终态任务阻止删除（抛 ValueError）。"""
    conn = get_connection()
    # Ch11：存在处理中的上传任务或索引构建任务时禁止删除知识库，
    # 避免 Worker 正在写入时删库导致数据不一致。
    if conn.execute(
        "SELECT 1 FROM upload_tasks WHERE kb_id=? AND status NOT IN ('completed','failed') LIMIT 1",
        (kb_id,),
    ).fetchone():
        conn.close()
        raise ValueError("知识库存在处理中的上传任务，请等待处理完成后再删除")
    if conn.execute(
        "SELECT 1 FROM index_build_tasks WHERE kb_id=? AND status NOT IN ('completed','failed') LIMIT 1",
        (kb_id,),
    ).fetchone():
        conn.close()
        raise ValueError("知识库存在处理中的索引构建任务，请等待处理完成后再删除")
    # 级联删除：先清会话消息、会话、用户绑定、索引问题绑定、索引问题、chunks、上传任务、索引构建任务、文档，再删知识库本身
    conn.execute(
        "DELETE FROM chat_messages WHERE session_id IN (SELECT id FROM chat_sessions WHERE kb_id = ?)",
        (kb_id,),
    )
    conn.execute("DELETE FROM chat_sessions WHERE kb_id = ?", (kb_id,))
    conn.execute("DELETE FROM user_kbs WHERE kb_id = ?", (kb_id,))
    conn.execute("DELETE FROM kb_retrieval_strategies WHERE kb_id = ?", (kb_id,))  # 第 3 期
    conn.execute("DELETE FROM kb_index_question_links WHERE kb_id = ?", (kb_id,))
    conn.execute("DELETE FROM kb_index_questions WHERE kb_id = ?", (kb_id,))
    conn.execute("DELETE FROM kb_chunks WHERE kb_id = ?", (kb_id,))
    conn.execute("DELETE FROM upload_tasks WHERE kb_id = ?", (kb_id,))  # Ch10 新增
    conn.execute("DELETE FROM index_build_tasks WHERE kb_id = ?", (kb_id,))  # Ch11 新增
    conn.execute("DELETE FROM kb_documents WHERE kb_id = ?", (kb_id,))
    cursor = conn.execute("DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


# ── 文档 CRUD ──────────────────────────────────────

def create_document(kb_id: str, filename: str, path: str, file_size: int = 0) -> dict | None:
    """创建文档记录。同一知识库下文件名不可重复，重复时返回 None。"""
    doc_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO kb_documents (doc_id, kb_id, filename, path, file_size, chunk_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, kb_id, filename, path, file_size, 0, now, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return None  # UNIQUE(kb_id, filename) 冲突 = 文件名重复

    conn.close()
    return get_document(doc_id)


def get_document(doc_id: str) -> dict | None:
    """根据 ID 获取单个文档。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM kb_documents WHERE doc_id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_documents(kb_id: str) -> list[dict]:
    """获取某个知识库下的所有文档。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM kb_documents WHERE kb_id = ? ORDER BY created_at DESC",
        (kb_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_document(doc_id: str) -> dict | None:
    """删除文档，级联清理 chunks、索引绑定、上传任务（Ch10 新增）。Ch11：索引构建期间删除返回 409。"""
    doc = get_document(doc_id)
    if doc is None:
        return None

    conn = get_connection()
    # Ch11：索引构建任务处理期间禁止删除文档，避免生成的索引问题
    # 绑定到已被删除的 chunk（构建结束后 chunks 可能已不存在）。
    if conn.execute(
        "SELECT 1 FROM index_build_tasks WHERE kb_id=? AND status NOT IN ('completed','failed') LIMIT 1",
        (doc["kb_id"],),
    ).fetchone():
        conn.close()
        raise ValueError("知识库正在构建索引问题，请等待构建完成后再删除文档")
    # 1. 删除索引绑定
    conn.execute("""
        DELETE FROM kb_index_question_links
        WHERE chunk_id IN (SELECT chunk_id FROM kb_chunks WHERE doc_id = ?)
    """, (doc_id,))
    # 2. 删除 chunks
    conn.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
    # 3. 先删除终态和初始态任务（必须在步骤 4 之前，否则刚标记的 "failed" 会被立即删掉）
    conn.execute(
        "DELETE FROM upload_tasks WHERE doc_id = ? AND status IN ('completed', 'failed', 'uploaded', 'dispatch_unknown')",
        (doc_id,),
    )
    # 4. 将正在处理的任务标记为 failed（保留记录，Worker 会检查发现文档已删）
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute("""
        UPDATE upload_tasks SET status='failed',
            step_message='文档已被删除，任务自动取消',
            error_message='文档删除时任务正在处理中',
            finished_at=?, updated_at=?
        WHERE doc_id = ? AND status IN ('parsing', 'indexing')
    """, (now, now, doc_id))
    # 5. 删除文档记录
    conn.execute("DELETE FROM kb_documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return doc


def delete_chunks_for_doc(doc_id: str):
    """删除某文档的所有分块及其索引问题绑定。"""
    conn = get_connection()
    conn.execute("""
        DELETE FROM kb_index_question_links
        WHERE chunk_id IN (SELECT chunk_id FROM kb_chunks WHERE doc_id = ?)
    """, (doc_id,))
    conn.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()


def save_chunks(doc_id: str, kb_id: str, chunks: list[dict], config_hash: str = ""):
    """批量保存分块到 kb_chunks，返回保存数量。

    支持父子分块：chunk 元素可带 chunk_id / parent_chunk_id / chunk_role /
    parent_index / sense_index / length；缺省表示通用分块
    （chunk_id = {doc_id}_{chunk_index}，chunk_role = child）。
    `config_hash` 是本次入库所用分块配置的指纹，写入文档行用于一致性校验。
    """
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") or f"{doc_id}_{chunk['chunk_index']}"
        conn.execute(
            "INSERT OR REPLACE INTO kb_chunks (chunk_id, doc_id, kb_id, chunk_index, content, created_at,"
            " parent_chunk_id, chunk_role, parent_index, sense_index, length)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk_id, doc_id, kb_id, chunk["chunk_index"], chunk["content"], now,
                chunk.get("parent_chunk_id", ""),
                chunk.get("chunk_role", "child"),
                chunk.get("parent_index", -1),
                chunk.get("sense_index", chunk["chunk_index"]),
                chunk.get("length", len(chunk["content"])),
            ),
        )
    conn.commit()
    conn.close()
    if config_hash:
        update_doc_config_hash(doc_id, config_hash)
    return len(chunks)  # Ch10 修复：返回保存数量，供 Worker 校验完整性


def get_chunks_for_doc(doc_id: str, role: str | None = None) -> list[dict]:
    """获取某文档的分块；role='child' / 'parent' 可只取叶子块或父块。"""
    conn = get_connection()
    if role:
        rows = conn.execute(
            "SELECT * FROM kb_chunks WHERE doc_id = ? AND chunk_role = ? ORDER BY chunk_index",
            (doc_id, role),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_chunks_by_ids(chunk_ids: list[str]) -> dict[str, dict]:
    """按 chunk_id 批量取块，返回 {chunk_id: row}（父子分块回查父块用）。"""
    unique = list(dict.fromkeys([cid for cid in chunk_ids if cid]))
    if not unique:
        return {}
    conn = get_connection()
    result: dict[str, dict] = {}
    for start in range(0, len(unique), 500):
        batch = unique[start:start + 500]
        placeholders = ",".join(["?"] * len(batch))
        rows = conn.execute(
            f"SELECT * FROM kb_chunks WHERE chunk_id IN ({placeholders})", batch
        ).fetchall()
        result.update({row["chunk_id"]: dict(row) for row in rows})
    conn.close()
    return result


def update_doc_chunk_count(doc_id: str):
    """重新计算文档的块数并更新。

    chunk_count 只统计**可检索叶子块**（child），父块数量单独记在 parent_count，
    这样前端的"分块数"与向量库里的条数始终一致。
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT SUM(CASE WHEN chunk_role = 'parent' THEN 1 ELSE 0 END) AS parents,"
        "       SUM(CASE WHEN chunk_role = 'child' THEN 1 ELSE 0 END) AS children"
        " FROM kb_chunks WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    parents = row["parents"] or 0
    children = row["children"] or 0
    conn.execute(
        "UPDATE kb_documents SET chunk_count = ?, parent_count = ?, child_count = ?, updated_at = ? WHERE doc_id = ?",
        (children, parents, children, datetime.now().isoformat(timespec="seconds"), doc_id),
    )
    conn.commit()
    conn.close()


def update_kb_chunk_count(kb_id: str):
    """重新计算知识库的总块数（只算可检索叶子块）并更新。"""
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM kb_chunks WHERE kb_id = ? AND chunk_role = 'child'", (kb_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE knowledge_bases SET chunk_count = ?, updated_at = ? WHERE kb_id = ?",
        (count, datetime.now().isoformat(timespec="seconds"), kb_id),
    )
    conn.commit()
    conn.close()


# ── 分块配置指纹与重索引（Advanced RAG 第 2 期）──────────

def update_doc_config_hash(doc_id: str, config_hash: str) -> None:
    """记录某文档**实际入库**所用的分块配置指纹。"""
    conn = get_connection()
    conn.execute("UPDATE kb_documents SET chunk_config_hash = ? WHERE doc_id = ?",
                 (config_hash, doc_id))
    conn.commit()
    conn.close()


def backfill_doc_config_hash(kb_id: str, config_hash: str) -> int:
    """把缺少配置指纹的旧文档补齐为给定值（启动迁移用），返回更新条数。

    旧库升级后：知识库配置尚未被改过，因此"当前配置"就是这些文档当初的入库配置。
    """
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE kb_documents SET chunk_config_hash = ?"
        " WHERE kb_id = ? AND (chunk_config_hash IS NULL OR chunk_config_hash = '')",
        (config_hash, kb_id),
    )
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


def retarget_pending_upload_tasks(kb_id: str, chunk_config: str,
                                  chunk_size: int, chunk_overlap: int) -> int:
    """配置切换后，把**尚未被 Worker 领取**的任务快照换成新配置，返回更新条数。

    只更新 uploaded / dispatch_unknown 状态：正在解析/向量化的任务必须继续用完它自己
    冻结的配置，中途换配置会让同一批数据前后不一致。
    """
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE upload_tasks SET chunk_config = ?, chunk_size = ?, chunk_overlap = ?, updated_at = ?"
        " WHERE kb_id = ? AND status IN ('uploaded', 'dispatch_unknown')",
        (chunk_config, chunk_size, chunk_overlap, now, kb_id),
    )
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


def list_docs_needing_reindex(kb_id: str, current_hash: str) -> list[dict]:
    """列出**已按其它配置入库过**、需要重新索引的文档。

    说明：指纹为空（从未成功入库，如刚上传还在排队）的文档不算"配置过期"，
    它们会在正常的入库流程里用当前配置处理。
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM kb_documents"
        " WHERE kb_id = ? AND COALESCE(chunk_config_hash, '') <> ''"
        "   AND chunk_config_hash <> ?"
        " ORDER BY created_at",
        (kb_id, current_hash),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_kb_doc_count(kb_id: str):
    """重新计算知识库的文档数量并更新。"""
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM kb_documents WHERE kb_id = ?", (kb_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE knowledge_bases SET doc_count = ?, updated_at = ? WHERE kb_id = ?",
        (count, datetime.now().isoformat(timespec="seconds"), kb_id),
    )
    conn.commit()
    conn.close()


# ── 对话会话管理（Ch6 新增）─────────────────────────────

def create_session(kb_id: str, title: str = "新对话") -> dict:
    """创建新的对话会话。"""
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_sessions (id, kb_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, kb_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return {"id": session_id, "kb_id": kb_id, "title": title, "created_at": now}


def get_session(session_id: str) -> dict | None:
    """获取会话信息，不存在返回 None。"""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, kb_id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def ensure_session(session_id: str | None, kb_id: str, title: str = "新对话") -> dict:
    """获取或创建会话。"""
    if session_id:
        session = get_session(session_id)
        if session:
            if session["kb_id"] != kb_id:
                raise ValueError("会话不属于当前知识库")
            return session
    return create_session(kb_id, title)


def save_message(session_id: str, role: str, content: str, citations: list | None = None) -> None:
    """保存一条消息到数据库。"""
    message_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_messages (id, session_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (message_id, session_id, role, content, json.dumps(citations or [], ensure_ascii=False), now),
    )
    conn.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
        (now, session_id),
    )
    conn.commit()
    conn.close()


def get_session_messages(session_id: str) -> list[dict]:
    """获取会话的所有消息，按时间升序。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, role, content, citations, created_at FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        msg = dict(row)
        msg["citations"] = json.loads(msg["citations"]) if msg["citations"] else []
        result.append(msg)
    return result


# ── 会话 CRUD（Ch7 新增）─────────────────────────────

def list_sessions(kb_id: str) -> list[dict]:
    """获取指定知识库的会话列表，按更新时间降序。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, kb_id, title, created_at, updated_at FROM chat_sessions WHERE kb_id = ? ORDER BY updated_at DESC",
        (kb_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def rename_session(session_id: str, new_title: str) -> bool:
    """重命名会话，返回是否成功。"""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
        (new_title, now, session_id),
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def delete_session(session_id: str) -> bool:
    """删除会话及其所有消息（级联删除），返回是否成功。"""
    conn = get_connection()
    conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def clear_session_messages(session_id: str) -> int:
    """清空会话的所有消息，返回删除的消息数量。"""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


def update_session_title(session_id: str, title: str):
    """更新会话标题（首次提问时自动设置）。仅在标题为默认值时更新。"""
    conn = get_connection()
    conn.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ? AND title = '新对话'",
        (title, datetime.now().isoformat(timespec="seconds"), session_id),
    )
    conn.commit()
    conn.close()


# ── 索引问题 CRUD（Ch8 新增）────────────────────────────

def create_index_question(kb_id: str, question: str, source: str = "manual") -> dict:
    """
    创建索引问题。
    source: 'manual'（手动创建）或 'ai'（大模型自动生成）。
    """
    index_id = f"idx_{uuid.uuid4().hex[:10]}"
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    conn.execute(
        "INSERT INTO kb_index_questions (index_id, kb_id, question, source, created_at) VALUES (?, ?, ?, ?, ?)",
        (index_id, kb_id, question, source, now),
    )
    conn.commit()
    conn.close()
    return {"index_id": index_id, "kb_id": kb_id, "question": question, "source": source, "created_at": now}


def get_index_question(index_id: str) -> dict | None:
    """获取单个索引问题。"""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM kb_index_questions WHERE index_id = ?", (index_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_index_questions(kb_id: str) -> list[dict]:
    """获取指定知识库的索引问题列表（含 link_count）。"""
    conn = get_connection()
    rows = conn.execute("""
        SELECT q.index_id, q.kb_id, q.question, q.source, q.created_at,
               COALESCE(c.cnt, 0) AS link_count
        FROM kb_index_questions q
        LEFT JOIN (
            SELECT l.index_id, COUNT(1) AS cnt
            FROM kb_index_question_links l
            JOIN kb_index_questions iq ON iq.index_id = l.index_id AND iq.kb_id = l.kb_id
            JOIN kb_chunks kc ON kc.chunk_id = l.chunk_id AND kc.kb_id = iq.kb_id
            JOIN kb_documents kd ON kd.doc_id = kc.doc_id AND kd.kb_id = iq.kb_id
            GROUP BY l.index_id
        ) c ON c.index_id = q.index_id
        WHERE q.kb_id = ?
        ORDER BY q.created_at DESC
    """, (kb_id,)).fetchall()
    conn.close()
    return [
        {
            "index_id": row["index_id"], "kb_id": row["kb_id"],
            "question": row["question"], "source": row["source"],
            "created_at": row["created_at"],
            "link_count": int(row["link_count"] or 0),
        }
        for row in rows
    ]


def update_index_question(index_id: str, question: str) -> bool:
    """更新索引问题的文本。"""
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE kb_index_questions SET question = ? WHERE index_id = ?",
        (question, index_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def delete_index_question(index_id: str) -> bool:
    """删除索引问题及其所有绑定关系。"""
    conn = get_connection()
    conn.execute("DELETE FROM kb_index_question_links WHERE index_id = ?", (index_id,))
    cursor = conn.execute("DELETE FROM kb_index_questions WHERE index_id = ?", (index_id,))
    conn.commit()
    return cursor.rowcount > 0


def add_index_links(index_id: str, kb_id: str, items: list[dict]) -> int:
    """
    为索引问题绑定 chunk。items 格式: [{"chunk_id": str}]。
    先校验整批归属，再写入；返回实际新增数量，重复绑定不计数。
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        question = conn.execute(
            "SELECT 1 FROM kb_index_questions WHERE index_id=? AND kb_id=?", (index_id, kb_id),
        ).fetchone()
        if question is None:
            raise ValueError("索引问题不属于当前知识库")
        chunk_ids = list(dict.fromkeys(item["chunk_id"] for item in items))
        for chunk_id in chunk_ids:
            chunk = conn.execute(
                "SELECT 1 FROM kb_chunks WHERE chunk_id=? AND kb_id=?", (chunk_id, kb_id),
            ).fetchone()
            if chunk is None:
                raise ValueError("绑定的分块不存在或不属于当前知识库")
        written = 0
        for chunk_id in chunk_ids:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO kb_index_question_links (index_id, chunk_id, kb_id) VALUES (?, ?, ?)",
                (index_id, chunk_id, kb_id),
            )
            written += cursor.rowcount
        conn.commit()
        return written
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def remove_index_link(index_id: str, chunk_id: str) -> bool:
    """解除索引问题与某个 chunk 的绑定。"""
    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM kb_index_question_links WHERE index_id = ? AND chunk_id = ?",
        (index_id, chunk_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_index_links(index_id: str) -> list[dict]:
    """获取索引问题的所有绑定（JOIN 获取 chunk 内容和来源）。"""
    conn = get_connection()
    rows = conn.execute("""
        SELECT l.id, l.index_id, l.chunk_id, l.kb_id,
               kc.doc_id, kc.chunk_index, kc.content as chunk_text,
               kd.filename as source
        FROM kb_index_question_links l
        JOIN kb_index_questions q ON q.index_id = l.index_id AND q.kb_id = l.kb_id
        JOIN kb_chunks kc ON l.chunk_id = kc.chunk_id AND kc.kb_id = q.kb_id
        JOIN kb_documents kd ON kc.doc_id = kd.doc_id AND kd.kb_id = q.kb_id
        WHERE l.index_id = ?
        ORDER BY kc.chunk_index
    """, (index_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_index_links_batch(index_ids: list[str]) -> list[dict]:
    """批量获取多个索引问题的绑定（JOIN 获取 chunk 内容和来源）。"""
    if not index_ids:
        return []

    conn = get_connection()
    placeholders = ",".join(["?"] * len(index_ids))
    rows = conn.execute(f"""
        SELECT l.id, l.index_id, l.chunk_id, l.kb_id,
               kc.doc_id, kc.chunk_index, kc.content as chunk_text,
               kd.filename as source
        FROM kb_index_question_links l
        JOIN kb_index_questions q ON q.index_id = l.index_id AND q.kb_id = l.kb_id
        JOIN kb_chunks kc ON l.chunk_id = kc.chunk_id AND kc.kb_id = q.kb_id
        JOIN kb_documents kd ON kc.doc_id = kd.doc_id AND kd.kb_id = q.kb_id
        WHERE l.index_id IN ({placeholders})
        ORDER BY kc.chunk_index
    """, index_ids).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_chunks_for_index(
    kb_id: str,
    index_id: str | None = None,
    link_filter: str = "all",
    file_name: str | None = None,
    keyword: str | None = None,
) -> list[dict]:
    """获取知识库中所有 chunk（用于候选浏览和绑定），支持筛选和关联状态查询。"""
    conn = get_connection()
    if index_id:
        rows = conn.execute("""
            SELECT kc.chunk_id, kc.doc_id, kc.kb_id, kc.chunk_index, kc.content, kc.created_at,
                   kd.filename AS source,
                   CASE WHEN l.id IS NOT NULL THEN 1 ELSE 0 END AS linked
            FROM kb_chunks kc
            JOIN kb_documents kd ON kc.doc_id = kd.doc_id AND kd.kb_id = kc.kb_id
            LEFT JOIN kb_index_questions q ON q.index_id = ? AND q.kb_id = kc.kb_id
            LEFT JOIN kb_index_question_links l
                ON l.chunk_id = kc.chunk_id AND l.index_id = q.index_id AND l.kb_id = q.kb_id
            WHERE kc.kb_id = ? AND kc.chunk_role = 'child'
            ORDER BY kc.doc_id, kc.chunk_index
        """, (index_id, kb_id)).fetchall()
    else:
        rows = conn.execute("""
            SELECT kc.chunk_id, kc.doc_id, kc.kb_id, kc.chunk_index, kc.content, kc.created_at,
                   kd.filename AS source, 0 AS linked
            FROM kb_chunks kc
            JOIN kb_documents kd ON kc.doc_id = kd.doc_id AND kd.kb_id = kc.kb_id
            WHERE kc.kb_id = ? AND kc.chunk_role = 'child'
            ORDER BY kc.doc_id, kc.chunk_index
        """, (kb_id,)).fetchall()
    conn.close()

    items = []
    for row in rows:
        item = {
            "chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
            "kb_id": row["kb_id"], "chunk_index": row["chunk_index"],
            "content": row["content"], "created_at": row["created_at"],
            "source": row["source"] or "", "linked": bool(row["linked"]),
        }
        if file_name and item["source"] != file_name:
            continue
        if keyword and keyword.lower() not in (item["content"] or "").lower():
            continue
        if link_filter == "linked" and not item["linked"]:
            continue
        if link_filter == "unlinked" and item["linked"]:
            continue
        items.append(item)
    return items


# ── 上传任务管理（Ch10 新增）────────────────────────────

def create_upload_task(kb_id: str, doc_id: str, filename: str, filepath: str,
                        chunk_size: int = 800, chunk_overlap: int = 100,
                        chunk_config: str = "", task_type: str = "upload") -> dict:
    """创建上传任务记录。

    chunk_config：创建任务时的分块配置快照（JSON 字符串）。Worker 只认这份快照，
    这样"排队等待期间知识库配置被改动"也不会让入库结果与当时预览的结果不一致。
    task_type：'upload'（首次入库）或 'reindex'（配置变更后重建索引）。
    """
    task_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    conn.execute("BEGIN IMMEDIATE")
    doc = conn.execute(
        "SELECT 1 FROM kb_documents d JOIN knowledge_bases k ON k.kb_id=d.kb_id WHERE d.doc_id=? AND d.kb_id=?",
        (doc_id, kb_id),
    ).fetchone()
    busy = conn.execute(
        "SELECT 1 FROM upload_tasks WHERE doc_id=? AND status NOT IN ('completed','failed') LIMIT 1",
        (doc_id,),
    ).fetchone()
    if doc is None or busy:
        conn.rollback()
        conn.close()
        raise ValueError("文档不存在或已有运行中任务")
    conn.execute(
        "INSERT INTO upload_tasks (task_id, kb_id, doc_id, filename, filepath, status, progress, step_message, chunk_size, chunk_overlap, chunk_config, task_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'uploaded', 0, '已上传，等待处理', ?, ?, ?, ?, ?, ?)",
        (task_id, kb_id, doc_id, filename, filepath, chunk_size, chunk_overlap,
         chunk_config, task_type, now, now),
    )
    conn.commit()
    conn.close()
    return {"task_id": task_id, "kb_id": kb_id, "doc_id": doc_id, "filename": filename,
            "status": "uploaded", "progress": 0, "step_message": "已上传，等待处理",
            "task_type": task_type}


def update_upload_task_counts(task_id: str, *, parent_count=None, child_count=None,
                              embedded_count=None, retry_count=None, current_stage=None) -> None:
    """更新任务的分块/向量化统计（Worker 处理过程中调用，供前端展示进度明细）。"""
    fields: dict = {}
    if parent_count is not None:
        fields["parent_count"] = parent_count
    if child_count is not None:
        fields["child_count"] = child_count
    if embedded_count is not None:
        fields["embedded_count"] = embedded_count
    if retry_count is not None:
        fields["retry_count"] = retry_count
    if current_stage is not None:
        fields["current_stage"] = current_stage
    if not fields:
        return
    fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
    sql = ", ".join(f"{key}=?" for key in fields)
    conn = get_connection()
    conn.execute(f"UPDATE upload_tasks SET {sql} WHERE task_id=?", (*fields.values(), task_id))
    conn.commit()
    conn.close()


def get_upload_task(task_id: str) -> dict | None:
    """获取任务详情。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM upload_tasks WHERE task_id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_tasks_by_doc_id(doc_id: str) -> list[dict]:
    """获取指定文档的所有上传任务，按创建时间降序。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM upload_tasks WHERE doc_id = ? ORDER BY created_at DESC",
        (doc_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def list_upload_tasks(kb_id: str | None = None) -> list[dict]:
    """获取任务列表，按创建时间降序。"""
    conn = get_connection()
    if kb_id:
        rows = conn.execute(
            "SELECT * FROM upload_tasks WHERE kb_id = ? ORDER BY created_at DESC", (kb_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM upload_tasks ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_upload_task(task_id: str) -> bool:
    """删除任务记录。只允许删除终态任务。"""
    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM upload_tasks WHERE task_id = ? AND status IN ('completed', 'failed')",
        (task_id,),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def delete_upload_task_raw(task_id: str) -> bool:
    """无条件删除任务记录（用于 delay 失败时的回滚）。"""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM upload_tasks WHERE task_id = ?", (task_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def retry_upload_task(task_id: str) -> dict | None:
    """
    重试失败的任务：将状态改为 uploaded，让 Worker 重新处理。
    排除因"文档被删除"而失败的任务——文档已不存在，重试无意义。
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM upload_tasks WHERE task_id = ? AND status = 'failed'", (task_id,),
    ).fetchone()
    if not row:
        conn.close()
        return None

    # 文档已被删除的任务不允许重试
    step_msg = dict(row).get("step_message", "")
    if "文档已被删除" in step_msg:
        conn.close()
        return None

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE upload_tasks SET status='uploaded', progress=0, "
        "step_message='等待重新处理', error_message='', updated_at=?, finished_at=NULL "
        "WHERE task_id=?", (now, task_id),
    )
    conn.commit()
    conn.close()
    return {"task_id": task_id, "status": "uploaded"}


def cleanup_stale_tasks() -> int:
    """
    清理僵尸任务：将 parsing/indexing 状态超过 30 分钟的任务标记为 failed。

    场景：Celery Worker 崩溃或被 kill，任务永远停在中间状态。
    启动时以及查询任务列表前调用，确保用户看到的任务状态是准确的。
    返回清理的任务数。
    """
    conn = get_connection()
    now = datetime.now().isoformat(timespec="seconds")
    cutoff = (datetime.now() - timedelta(minutes=30)).isoformat(timespec="seconds")
    # 查找 parsing/indexing 状态且 updated_at 超过 30 分钟的任务
    cursor = conn.execute("""
        UPDATE upload_tasks
        SET status = 'failed', step_message = 'Worker 异常退出，任务已自动标记为失败',
            error_message = '任务长时间未更新，疑似 Worker 崩溃', finished_at = ?, updated_at = ?
        WHERE status IN ('parsing', 'indexing')
          AND updated_at < ?
    """, (now, now, cutoff))
    count = cursor.rowcount
    if count > 0:
        conn.commit()
    conn.close()
    if count > 0:
        print(f"[startup] 清理 {count} 个僵尸任务（Worker 崩溃导致）")
    return count


def claim_upload_task(task_id: str) -> dict | None:
    """
    Worker 领取任务（条件更新，防止同一任务被多个 Worker 重复处理）。

    仅当任务处于 uploaded/dispatch_unknown 且未被领取时，才原子地
    更新为 parsing 并返回任务信息；任务不存在、已被领取或已进入终态时返回 None。
    """
    conn = get_connection()
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT * FROM upload_tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    if row is None:
        conn.rollback()
        conn.close()
        return None
    task = dict(row)
    if task["status"] not in ("uploaded", "dispatch_unknown"):
        # 已进入 parsing/indexing（被别的 Worker 领取）或终态，跳过
        conn.rollback()
        conn.close()
        return None

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE upload_tasks SET status='parsing', progress=20, "
        "step_message='正在解析文档...', updated_at=? WHERE task_id=?",
        (now, task_id),
    )
    conn.commit()
    conn.close()
    return task


def mark_upload_dispatch_unknown(task_id: str, error: str = "") -> bool:
    """
    发布结果未知时条件更新状态：仅当任务仍为 uploaded 时标记为 dispatch_unknown。

    场景：delay() 抛异常但消息可能已送达 Redis。保留任务、文档和文件，
    返回 True 表示已标记；任务已被 Worker 领取（不再是 uploaded）时不覆盖。
    """
    conn = get_connection()
    now = datetime.now().isoformat(timespec="seconds")
    cursor = conn.execute(
        "UPDATE upload_tasks SET status='dispatch_unknown', "
        "step_message='发送结果待确认，请查询任务状态，不要重复上传', "
        "error_message=?, updated_at=? WHERE task_id=? AND status='uploaded'",
        (str(error)[:200], now, task_id),
    )
    conn.commit()
    ok = cursor.rowcount > 0
    conn.close()
    return ok


# ── 索引构建任务管理（Ch11 新增）────────────────────────

def create_index_build_task(kb_id: str, prompt: str = "") -> dict:
    """创建索引构建任务。知识库不存在或已有运行中任务时抛 ValueError。"""
    task_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    conn.execute("BEGIN IMMEDIATE")
    kb = conn.execute(
        "SELECT 1 FROM knowledge_bases WHERE kb_id = ?", (kb_id,)
    ).fetchone()
    busy = conn.execute(
        "SELECT 1 FROM index_build_tasks WHERE kb_id=? AND status NOT IN ('completed','failed') LIMIT 1",
        (kb_id,),
    ).fetchone()
    if kb is None or busy:
        conn.rollback()
        conn.close()
        raise ValueError("知识库不存在或已有运行中的索引构建任务")
    conn.execute(
        "INSERT INTO index_build_tasks (task_id, kb_id, prompt, status, progress, step_message, created_at, updated_at) VALUES (?, ?, ?, 'queued', 0, '已提交，等待处理', ?, ?)",
        (task_id, kb_id, prompt, now, now),
    )
    conn.commit()
    conn.close()
    return {"task_id": task_id, "kb_id": kb_id, "prompt": prompt,
            "status": "queued", "progress": 0, "step_message": "已提交，等待处理"}


def get_index_build_task(task_id: str) -> dict | None:
    """获取索引构建任务详情。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM index_build_tasks WHERE task_id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_index_build_tasks(kb_id: str | None = None) -> list[dict]:
    """获取索引构建任务列表，按创建时间降序。"""
    conn = get_connection()
    if kb_id:
        rows = conn.execute(
            "SELECT * FROM index_build_tasks WHERE kb_id = ? ORDER BY created_at DESC", (kb_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM index_build_tasks ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_index_build_task(task_id: str) -> bool:
    """删除索引构建任务记录。只允许删除终态任务。"""
    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM index_build_tasks WHERE task_id = ? AND status IN ('completed', 'failed')",
        (task_id,),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def claim_index_build_task(task_id: str) -> dict | None:
    """
    Worker 领取索引构建任务（条件更新，防止同一任务被多个 Worker 重复执行）。

    仅当任务处于 queued/dispatch_unknown 时，原子地更新为 building 并返回任务信息；
    任务不存在、已被领取或已进入终态时返回 None。
    """
    conn = get_connection()
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT * FROM index_build_tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    if row is None:
        conn.rollback()
        conn.close()
        return None
    task = dict(row)
    if task["status"] not in ("queued", "dispatch_unknown"):
        conn.rollback()
        conn.close()
        return None

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE index_build_tasks SET status='building', progress=5, "
        "step_message='Worker 已领取，开始处理', updated_at=? WHERE task_id=?",
        (now, task_id),
    )
    conn.commit()
    conn.close()
    return task


def mark_index_build_dispatch_unknown(task_id: str, error: str = "") -> bool:
    """发布结果未知时条件更新状态：仅当任务仍为 queued 时标记为 dispatch_unknown。"""
    conn = get_connection()
    now = datetime.now().isoformat(timespec="seconds")
    cursor = conn.execute(
        "UPDATE index_build_tasks SET status='dispatch_unknown', "
        "step_message='发送结果待确认，请查询任务状态', "
        "error_message=?, updated_at=? WHERE task_id=? AND status='queued'",
        (str(error)[:200], now, task_id),
    )
    conn.commit()
    ok = cursor.rowcount > 0
    conn.close()
    return ok


def cleanup_stale_index_build_tasks() -> int:
    """
    清理僵尸索引构建任务：building 状态超过 30 分钟标记为 failed。

    场景：Worker 崩溃导致索引构建永远停在中间状态，进而阻塞知识库删除。
    启动时以及查询任务列表前调用。返回清理的任务数。
    """
    conn = get_connection()
    now = datetime.now().isoformat(timespec="seconds")
    cutoff = (datetime.now() - timedelta(minutes=30)).isoformat(timespec="seconds")
    cursor = conn.execute("""
        UPDATE index_build_tasks
        SET status = 'failed', step_message = 'Worker 异常退出，任务已自动标记为失败',
            error_message = '任务长时间未更新，疑似 Worker 崩溃', finished_at = ?, updated_at = ?
        WHERE status = 'building' AND updated_at < ?
    """, (now, now, cutoff))
    count = cursor.rowcount
    if count > 0:
        conn.commit()
    conn.close()
    return count


def cleanup_index_questions(kb_id: str, mode: str = "orphan") -> dict:
    """
    清理索引问题。mode='orphan' 清理无绑定的；mode='all' 全部清空。

    返回 {"deleted_count": int, "index_ids": [str, ...]}，
    路由据此同步清理 Milvus 中的索引问题向量。
    """
    conn = get_connection()
    if mode == "all":
        rows = conn.execute(
            "SELECT index_id FROM kb_index_questions WHERE kb_id = ?", (kb_id,)
        ).fetchall()
        index_ids = [r["index_id"] for r in rows]
        conn.execute("DELETE FROM kb_index_question_links WHERE kb_id = ?", (kb_id,))
        conn.execute("DELETE FROM kb_index_questions WHERE kb_id = ?", (kb_id,))
    else:
        # orphan：没有任何绑定关系的索引问题
        rows = conn.execute("""
            SELECT q.index_id FROM kb_index_questions q
            LEFT JOIN kb_index_question_links l ON l.index_id = q.index_id AND l.kb_id = q.kb_id
            WHERE q.kb_id = ? AND l.id IS NULL
        """, (kb_id,)).fetchall()
        index_ids = [r["index_id"] for r in rows]
        if index_ids:
            placeholders = ",".join(["?"] * len(index_ids))
            conn.execute(
                f"DELETE FROM kb_index_question_links WHERE kb_id=? AND index_id IN ({placeholders})",
                (kb_id, *index_ids),
            )
            conn.execute(
                f"DELETE FROM kb_index_questions WHERE kb_id=? AND index_id IN ({placeholders})",
                (kb_id, *index_ids),
            )
    conn.commit()
    conn.close()
    return {"deleted_count": len(index_ids), "index_ids": index_ids}