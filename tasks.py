"""
Celery 异步任务：文档上传 → 解析 → 向量化 → 入库。

Ch10 新增：将同步的上传流程改为异步任务队列。

核心流程：
  1. FastAPI 收到文件，保存到磁盘，创建任务记录
  2. 通过 Celery 将任务发送到 Redis 队列
  3. Celery Worker 从队列取出任务，执行解析+向量化
  4. 每个阶段更新任务状态和进度

状态机：
  uploaded → parsing → indexing → completed
                              ↘ failed（回滚已插入数据）

设计原则：
  - 失败补偿：任何 chunk 向量生成失败（重试后仍失败）→ 回滚所有已插入数据 → 标记 failed
  - 每个 chunk 最多重试 3 次
  - Worker 崩溃保护：Celery time_limit + SoftTimeLimitExceeded
  - 任务取消：_update_task() 写入前检测外部取消，防止覆盖 delete_document 的状态
"""
import os
import sys
import time
import re
import traceback
import logging
from uuid import uuid4
from datetime import datetime

# 确保 Worker 不管从哪个目录启动都能找到同目录模块
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# Ch11 改造：用 logging 替代 print()，便于按模块控制日志级别
logger = logging.getLogger("rag.tasks")

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from config import (
    CELERY_BROKER_URL, CELERY_RESULT_BACKEND, UPLOAD_TASK_QUEUE,
    PARSE_TIMEOUT, INDEX_TIMEOUT,
    CHAT_DB_PATH, UPLOAD_DIR, DATA_DIR,
)
import langsmith_tracing

# ── Celery 实例 ───────────────────────────────────────

celery_app = Celery(
    "rag_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# ── Celery 配置 ──────────────────────────────────────
# gevent 供本地协作并发调试，不提供软时限保证
# 教学版采用 early ack；Worker 崩溃后由僵尸清理标记 failed，再手动重试
# 启动命令：celery -A tasks.celery_app worker -Q upload_queue -l info -P gevent -c 3
celery_app.conf.update(
    task_acks_late=False,
    task_reject_on_worker_lost=False,
)

# ── 常量 ──────────────────────────────────────────────

CHUNK_EMBED_RETRY = 3          # 单个 chunk 向量生成最大重试次数
CHUNK_EMBED_RETRY_DELAY = 2    # 重试间隔（秒）
TASK_SOFT_TIME_LIMIT = 1200    # Celery 软超时（20 分钟）
TASK_HARD_TIME_LIMIT = 1500    # Celery 硬超时（25 分钟）

# ── 任务取消机制 ──────────────────────────────────────

class TaskCancelled(Exception):
    """任务被外部取消（如文档被删除时，delete_document 将任务标记为 failed）。"""
    pass


def _check_task_cancelled(task_id: str):
    """
    检查任务记录是否已删除，或者已经进入 completed/failed 终态。
    """
    import sqlite3
    conn = sqlite3.connect(CHAT_DB_PATH)
    row = conn.execute(
        "SELECT status, step_message FROM upload_tasks WHERE task_id = ?",
        (task_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise TaskCancelled("任务记录已被删除")
    status, step_message = row
    if status in ("completed", "failed", "parse_failed"):
        raise TaskCancelled(step_message or f"任务已进入终态: {status}")


# ── 任务处理函数 ───────────────────────────────────────

def _update_task(task_id: str, status: str, progress: int, step_message: str = "", error_message: str = ""):
    """
    更新任务状态到数据库。

    写入前先检查任务是否已被外部取消（如文档被删除）。
    如果已被取消，抛出 TaskCancelled 而非覆盖取消状态。
    这让每次进度更新都变成隐式的取消检查点。
    """
    import sqlite3
    conn = sqlite3.connect(CHAT_DB_PATH)

    # 先检查：任务是否已被 delete_document 标记为取消
    conn.execute("BEGIN IMMEDIATE")
    current = conn.execute(
        "SELECT status, step_message FROM upload_tasks WHERE task_id = ?",
        (task_id,)
    ).fetchone()
    if current is None:
        conn.close()
        raise TaskCancelled("任务记录已被删除")
    cur_status, cur_step = current
    if cur_status == "failed" and cur_step and "文档已被删除" in cur_step:
        conn.close()
        raise TaskCancelled(cur_step)

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE upload_tasks SET status=?, progress=?, step_message=?, error_message=?, updated_at=?, finished_at=? WHERE task_id=?",
        (status, progress, step_message, error_message, now, now if status in ("completed", "failed") else None, task_id),
    )
    conn.commit()
    conn.close()


def _cleanup_inserted_vectors(kb_id, doc_id, filename):
    """回滚已插入的 Milvus 向量和 SQLite chunk 数据。"""
    try:
        import milvus_ops
        milvus_ops.delete_vectors_for_doc(kb_id, filename)
    except Exception as exc:
        logger.warning("清理 Milvus 向量失败（可忽略）: %s", exc)
    try:
        import database
        database.delete_chunks_for_doc(doc_id)
    except Exception as exc:
        logger.warning("清理 chunks 失败（可忽略）: %s", exc)


@celery_app.task(
    name="process_upload",
    queue=UPLOAD_TASK_QUEUE,
    soft_time_limit=TASK_SOFT_TIME_LIMIT,
    time_limit=TASK_HARD_TIME_LIMIT,
)
def process_upload_task(task_id: str):
    """
    Celery 任务：处理文件上传的完整流程。

    1. 解析文档并分块
    2. 生成向量（每个 chunk 最多重试 3 次）
    3. 写入 Milvus

    失败补偿：任何 chunk 失败 → 回滚所有已插入数据 → 标记 failed
    """
    # 用于回滚的上下文
    kb_id = None
    doc_id = None
    filename = None
    current_progress = 0

    # Advanced RAG 第 4 期：文档链路追踪（trace_id 直接用 task_id，与任务表天然对齐）
    import observability
    trace = None

    try:
        # 1. 加载任务信息
        import database
        task = database.claim_upload_task(task_id)
        if task is None:
            return {"status": "skipped", "message": "任务不存在、已领取或已结束"}
        kb_id = task["kb_id"]
        doc_id = task["doc_id"]
        filename = task["filename"]
        filepath = task["filepath"]
        chunk_size = task["chunk_size"]
        chunk_overlap = task["chunk_overlap"]

        # 文档链路追踪：trace_id 直接用 task_id
        trace = observability.TraceRecorder.for_task(
            task_id, "upload", kb_id=kb_id, question=filename)

        if not os.path.exists(filepath):
            _update_task(task_id, "failed", 0, "文件不存在")
            trace.finish(status=observability.STATUS_FAILED, error="原始文件不存在")
            return {"status": "failed", "message": "文件不存在"}

        # ✅ 检查点 1：加载任务后，确认任务未被外部取消
        _check_task_cancelled(task_id)

        # 2. 解析文档
        _update_task(task_id, "parsing", 20, "正在解析文档...")
        current_progress = 20

        start_time = time.time()
        # Advanced RAG 第 1/2 期：优先使用任务内**冻结的分块配置快照**（没有快照的旧任务
        # 回落到知识库当前配置），保证入库结果与上传时预览的结果一致。
        import chunking
        task_config = chunking.config_from_upload_task(task, database.get_kb(kb_id))
        _parse_started = time.time()
        built = chunking.build_document_chunks(filepath, task_config)
        _parse_ms = int((time.time() - _parse_started) * 1000)
        config_hash = chunking.config_hash(task_config)
        trace.mark(observability.STAGE_PARSE, duration_ms=_parse_ms,
                   output_count=len(built["children"]),
                   chars=sum(item["chars"] for item in built["children"]))
        trace.mark(observability.STAGE_SPLIT, duration_ms=_parse_ms,
                   strategy=built["strategy"], parents=len(built["parents"]),
                   children=len(built["children"]),
                   config_hash=config_hash, length_mode=task_config["length_mode"])

        # 父子块分配稳定 chunk_id：父块 {doc_id}_p{n}，子块 {doc_id}_c{n}
        # 通用分块沿用历史方案 {doc_id}_{n}，存量向量不受影响。
        parent_child_mode = built["strategy"] == "parent_child"
        parent_rows = [
            {**parent,
             "chunk_id": f"{doc_id}_p{parent['chunk_index']}",
             "chunk_role": "parent"}
            for parent in built["parents"]
        ]
        child_rows = [
            {**child,
             "chunk_id": (f"{doc_id}_c{child['chunk_index']}" if parent_child_mode
                          else f"{doc_id}_{child['chunk_index']}"),
             "chunk_role": "child",
             "parent_chunk_id": (f"{doc_id}_p{child['parent_index']}"
                                 if child.get("parent_index", -1) >= 0 else "")}
            for child in built["children"]
        ]
        chunks = child_rows  # 向量化对象：只有可检索叶子块进 Milvus

        if time.time() - start_time > PARSE_TIMEOUT:
            _update_task(task_id, "failed", 20, "解析超时", f"解析耗时超过 {PARSE_TIMEOUT} 秒")
            trace.finish(status=observability.STATUS_FAILED,
                         error=f"解析耗时超过 {PARSE_TIMEOUT} 秒")
            return {"status": "failed", "message": "解析超时"}

        if not chunks:
            _update_task(task_id, "failed", 20, "未解析到文本内容")
            trace.finish(status=observability.STATUS_FAILED, error="未解析到文本内容",
                         stage=observability.STAGE_SPLIT)
            return {"status": "failed", "message": "未解析到内容"}

        database.update_upload_task_counts(
            task_id, parent_count=len(parent_rows), child_count=len(child_rows),
            current_stage="split",
        )

        # ✅ 检查点 2：解析完成后，确认文档未被删除
        _check_task_cancelled(task_id)

        # 3. 生成向量（带重试）
        if parent_child_mode:
            stage_msg = (f"正在向量化（父块 {len(parent_rows)} / 子块 {len(child_rows)}）...")
        else:
            stage_msg = f"正在向量化（共 {len(chunks)} 个块）..."
        _update_task(task_id, "indexing", 50, stage_msg)
        database.update_upload_task_counts(task_id, current_stage="embedding")
        current_progress = 50

        import embedding
        import milvus_ops


        rows = [ ]


        failed_chunks = [ ]

        retry_total = 0
        _embed_started = time.time()

        for i, chunk in enumerate(chunks):
            if time.time() - start_time > INDEX_TIMEOUT:
                raise TimeoutError("索引处理超过应用级时间预算")
            chunk_data = None
            last_error = None

            # 每个 chunk 最多重试 CHUNK_EMBED_RETRY 次
            for attempt in range(1, CHUNK_EMBED_RETRY + 1):
                try:
                    emb = embedding.bge_model.embed(chunk["content"])
                    chunk_data = {
                        "pk": chunk["chunk_id"],
                        "kb_id": kb_id,
                        "source": filename,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["content"],
                        "dense": embedding.get_dense_vector(emb),
                        "sparse": embedding.get_sparse_vector(emb),
                    }
                    break  # 成功，跳出重试循环
                except Exception as exc:
                    last_error = exc
                    if attempt < CHUNK_EMBED_RETRY:
                        retry_total += 1
                        database.update_upload_task_counts(task_id, retry_count=retry_total)
                        logger.warning("chunk #%s 向量生成失败（第 %s/%s 次）: %s，%ss 后重试", i, attempt, CHUNK_EMBED_RETRY, exc, CHUNK_EMBED_RETRY_DELAY)
                        time.sleep(CHUNK_EMBED_RETRY_DELAY)

            if chunk_data is not None:
                rows.append(chunk_data)
            else:
                logger.error("chunk #%s 向量生成最终失败（已重试 %s 次）: %s", i, CHUNK_EMBED_RETRY, last_error)
                failed_chunks.append((i, str(last_error)[:100]))

            # 更新进度
            progress = 50 + int((i + 1) / len(chunks) * 45)
            if progress > current_progress:
                current_progress = progress
                _update_task(task_id, "indexing", progress, f"向量化进度: {i+1}/{len(chunks)}")
                database.update_upload_task_counts(task_id, embedded_count=len(rows))

            # ✅ 检查点 3：每 5 个 chunk 检查一次取消状态
            if (i + 1) % 5 == 0:
                _check_task_cancelled(task_id)

        _embed_ms = int((time.time() - _embed_started) * 1000)
        trace.mark(observability.STAGE_EMBED_DOC, duration_ms=_embed_ms,
                   input_count=len(chunks), output_count=len(rows),
                   retry_count=retry_total, failed_chunks=len(failed_chunks))

        # 检查超时
        if time.time() - start_time > INDEX_TIMEOUT:
            _update_task(task_id, "failed", current_progress, "向量化超时", f"总耗时超过 {INDEX_TIMEOUT} 秒")
            trace.finish(status=observability.STATUS_FAILED, error=f"向量化超过 {INDEX_TIMEOUT} 秒")
            return {"status": "failed", "message": "向量化超时"}

        # 失败补偿：任何 chunk 失败 → 回滚 → 标记 failed
        if failed_chunks:
            error_detail = "; ".join([f"chunk#{idx}: {err}" for idx, err in failed_chunks[:5]])
            _update_task(
                task_id, "failed", current_progress,
                f"向量生成失败（{len(failed_chunks)}/{len(chunks)} 个块失败）",
                f"已重试 {CHUNK_EMBED_RETRY} 次。{error_detail}"
            )
            # 回滚已插入的数据
            logger.warning("回滚已插入数据: doc_id=%s, filename=%s", doc_id, filename)
            _cleanup_inserted_vectors(kb_id, doc_id, filename)
            trace.finish(status=observability.STATUS_FAILED,
                         error=f"{len(failed_chunks)}/{len(chunks)} 个块向量生成失败（已重试 {CHUNK_EMBED_RETRY} 次）")
            return {"status": "failed", "message": f"{len(failed_chunks)} 个块向量化失败，已回滚"}

        # ✅ 检查点 4：写入 Milvus 前最后确认
        _check_task_cancelled(task_id)

        # 4. 写入 Milvus
        _write_started = time.time()
        try:
            # 重建沿用稳定 chunk ID；先清理旧版本，避免较短新文档残留旧块。
            milvus_ops.delete_vectors_for_doc(kb_id, filename)
            database.delete_chunks_for_doc(doc_id)
            if rows:
                milvus_ops.upsert_vectors(rows)
        except Exception as exc:
            _cleanup_inserted_vectors(kb_id, doc_id, filename)
            _update_task(task_id, "failed", current_progress, "Milvus 写入失败", str(exc)[:200])
            trace.mark(observability.STAGE_VECTOR_WRITE, status=observability.STATUS_FAILED,
                       duration_ms=int((time.time() - _write_started) * 1000), error=str(exc))
            trace.finish(status=observability.STATUS_FAILED, error=f"Milvus 写入失败: {exc}")
            return {"status": "failed", "message": f"Milvus 写入失败: {exc}"}
        trace.mark(observability.STAGE_VECTOR_WRITE,
                   duration_ms=int((time.time() - _write_started) * 1000),
                   output_count=len(rows), source=filename)

        # 5. 保存父块 + 子块到 SQLite（父块只落库，检索回查后用于组装完整上下文）
        import database
        _check_task_cancelled(task_id)
        _save_started = time.time()
        all_rows = parent_rows + child_rows
        saved = database.save_chunks(doc_id, kb_id, all_rows, config_hash=config_hash)
        if saved != len(all_rows):
            raise RuntimeError("SQLite 分块保存数量不完整")
        database.update_doc_chunk_count(doc_id)
        database.update_kb_chunk_count(kb_id)
        database.update_kb_doc_count(kb_id)
        database.update_upload_task_counts(
            task_id, parent_count=len(parent_rows), child_count=len(child_rows),
            embedded_count=len(rows), current_stage="completed",
        )
        trace.mark(observability.STAGE_CHUNK_SAVE,
                   duration_ms=int((time.time() - _save_started) * 1000),
                   output_count=saved, parents=len(parent_rows), children=len(child_rows),
                   config_hash=config_hash)

        _update_task(task_id, "completed", 100, "完成", "")
        trace.mark(observability.STAGE_INDEX_DONE, duration_ms=trace.elapsed_ms(),
                   output_count=len(rows), task_type=task.get("task_type", "upload"))
        trace.finish(status=observability.STATUS_OK)
        return {"status": "completed", "total_chunks": len(child_rows),
                "indexed_chunks": len(rows), "parent_chunks": len(parent_rows)}

    except SoftTimeLimitExceeded:
        # Celery 软超时：优雅退出
        error_msg = f"任务超时（超过 {TASK_SOFT_TIME_LIMIT} 秒），被 Celery 终止"
        logger.error("任务 %s 软超时: %s", task_id, error_msg)
        _update_task(task_id, "failed", current_progress, "任务超时", error_msg)
        if kb_id and doc_id and filename:
            _cleanup_inserted_vectors(kb_id, doc_id, filename)
        if trace is not None:
            trace.finish(status=observability.STATUS_FAILED, error=error_msg)
        return {"status": "failed", "message": error_msg}

    except TaskCancelled as exc:
        # 文档被删除导致任务取消
        logger.warning("任务 %s 被取消: %s", task_id, exc)
        # 不调用 _update_task()——delete_document 已经设置了最终的 failed 状态
        # 清理本 Worker 可能已插入的中间数据（Milvus 向量、SQLite chunks）
        if kb_id and doc_id and filename:
            _cleanup_inserted_vectors(kb_id, doc_id, filename)
        if trace is not None:
            trace.finish(status=observability.STATUS_FAILED, error=f"任务被取消: {exc}")
        return {"status": "cancelled", "message": str(exc)}

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.error("任务 %s 异常: %s", task_id, error_msg)
        logger.error(traceback.format_exc())

        # 先检查任务是否已被外部取消（文档被删除），避免覆盖友好的取消消息
        try:
            _check_task_cancelled(task_id)
        except TaskCancelled as cancel_exc:
            logger.warning("任务 %s 已在外部被取消，跳过异常覆盖: %s", task_id, cancel_exc)
            if kb_id and doc_id and filename:
                _cleanup_inserted_vectors(kb_id, doc_id, filename)
            if trace is not None:
                trace.finish(status=observability.STATUS_FAILED, error=f"任务被取消: {cancel_exc}")
            return {"status": "cancelled", "message": str(cancel_exc)}

        _update_task(task_id, "failed", current_progress, "任务异常", error_msg)
        if kb_id and doc_id and filename:
            _cleanup_inserted_vectors(kb_id, doc_id, filename)
        if trace is not None:
            trace.finish(status=observability.STATUS_FAILED, error=error_msg)
        return {"status": "failed", "message": error_msg}


# ══════════════════════════════════════════════════════════
# 索引构建任务（Ch11 新增）：自动为知识库的每个 chunk 生成索引问题
#
# 状态机：queued → building → completed
#                              ↘ failed（任一 chunk 生成失败即整次失败）
#
# 设计要点：
#   - 条件领取（claim_index_build_task）：防止同一任务被多个 Worker 重复执行
#   - 任一 chunk 生成/入库失败（重试后仍失败）→ 整次任务标记 failed
#   - 与 process_upload_task 同队列、同超时保护
# ══════════════════════════════════════════════════════════

# 默认 prompt 模板（沿用 Ch8 的约定：问题答案必须完整落在段落内）
DEFAULT_INDEX_PROMPT = (
    "内容：{data}\n\n"
    "请仅根据上述内容生成 1-3 个问题。\n"
    "硬性要求（务必遵守）：\n"
    "- 每个问题的答案【必须完整包含在上面的内容里】，且能由这段内容准确回答；\n"
    "- 禁止生成需要段落外知识才能回答的宽泛问题；\n"
    "- 问题要具体、明确，直指本段的核心要点；\n"
    "- 只输出问题，不要输出答案或解释；\n"
    "- 将每个问题放在<question></question>标签中。"
)


def _update_index_build_task(task_id: str, status: str, progress: int,
                             step_message: str = "", error_message: str = ""):
    """更新索引构建任务状态。任务记录被删除时抛 TaskCancelled（不再写入）。"""
    import sqlite3
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.execute("BEGIN IMMEDIATE")
    current = conn.execute(
        "SELECT status FROM index_build_tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    if current is None:
        conn.close()
        raise TaskCancelled("索引构建任务记录已被删除")

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE index_build_tasks SET status=?, progress=?, step_message=?, error_message=?, updated_at=?, finished_at=? WHERE task_id=?",
        (status, progress, step_message, error_message, now,
         now if status in ("completed", "failed") else None, task_id),
    )
    conn.commit()
    conn.close()


@celery_app.task(
    name="process_index_build",
    queue=UPLOAD_TASK_QUEUE,
    soft_time_limit=TASK_SOFT_TIME_LIMIT,
    time_limit=TASK_HARD_TIME_LIMIT,
)
def process_index_build_task(task_id: str):
    """
    Celery 任务：为知识库所有 chunk 自动生成索引问题。

    流程：领取任务 → 读取 chunk → 逐块 LLM 生成问题 → 入库（SQLite + Milvus）
         → 绑定到源 chunk → 更新进度 → completed。

    失败策略：任一 chunk 生成失败（重试后仍失败）→ 整次任务标记 failed。
    """
    kb_id = None
    current_progress = 0

    try:
        # 1. 领取任务（条件更新，防止重复执行）
        import database
        task = database.claim_index_build_task(task_id)
        if task is None:
            return {"status": "skipped", "message": "任务不存在、已领取或已结束"}
        kb_id = task["kb_id"]
        prompt_template = task.get("prompt") or DEFAULT_INDEX_PROMPT

        # 2. 读取知识库的所有 chunk
        _update_index_build_task(task_id, "building", 8, "正在读取分块...")
        chunks = database.get_chunks_for_index(kb_id)
        if not chunks:
            _update_index_build_task(task_id, "failed", 8, "知识库中没有已索引的 chunk")
            return {"status": "failed", "message": "知识库中没有已索引的 chunk"}

        total = len(chunks)
        _update_index_build_task(task_id, "building", 10, f"共 {total} 个块，开始生成索引问题...")
        current_progress = 10

        import embedding
        import milvus_ops

        generated = 0
        start_time = time.time()

        # 3. 逐块生成索引问题
        for i, chunk in enumerate(chunks):
            if time.time() - start_time > INDEX_TIMEOUT:
                _update_index_build_task(task_id, "failed", current_progress,
                                         "索引构建超时", f"总耗时超过 {INDEX_TIMEOUT} 秒")
                return {"status": "failed", "message": "索引构建超时"}

            chunk_id = f"{chunk['doc_id']}_{chunk['chunk_index']}"
            ok = False
            last_error = None

            # 每个 chunk 的生成 + 入库最多重试 CHUNK_EMBED_RETRY 次
            for attempt in range(1, CHUNK_EMBED_RETRY + 1):
                try:
                    user_msg = prompt_template.replace("{data}", chunk["content"] or "")
                    # LangSmith 未开启时原样返回客户端；trace_id 即本任务的 task_id
                    llm = langsmith_tracing.traced_llm(embedding.llm_client,
                                                       ability="index_questions", task_id=task_id)
                    answer = llm.invoke([
                        {"role": "user", "content": user_msg}
                    ]).content

                    # 解析 <question>...</question> 标签
                    questions = re.findall(r"<question>(.*?)</question>", answer, re.DOTALL)
                    questions = [q.strip() for q in questions if q.strip()]
                    # fallback：按行提取问句
                    if not questions:
                        for line in answer.splitlines():
                            line = line.strip()
                            if line.endswith("?") or line.endswith("？") or (len(line) > 10):
                                questions.append(line)
                            if len(questions) >= 3:
                                break

                    for q in questions[:3]:
                        item = database.create_index_question(kb_id, q, source="ai")
                        try:
                            emb = embedding.bge_model.embed(q)
                            dense_vec = embedding.get_dense_vector(emb)
                            milvus_ops.upsert_index_question_vector(item["index_id"], kb_id, q, dense_vec)
                        except Exception as exc:
                            logger.warning("索引问题向量写入 Milvus 失败: %s", exc)
                        database.add_index_links(item["index_id"], kb_id, [{"chunk_id": chunk_id}])
                        generated += 1
                    ok = True
                    break  # 成功，跳出重试循环
                except Exception as exc:
                    last_error = exc
                    if attempt < CHUNK_EMBED_RETRY:
                        logger.warning("chunk %s 索引问题生成失败（第 %s/%s 次）: %s，%ss 后重试",
                                       chunk_id, attempt, CHUNK_EMBED_RETRY, exc, CHUNK_EMBED_RETRY_DELAY)
                        time.sleep(CHUNK_EMBED_RETRY_DELAY)

            # 任一 chunk 失败 → 整次任务标记 failed
            if not ok:
                error_msg = f"chunk {chunk_id} 生成索引问题失败（已重试 {CHUNK_EMBED_RETRY} 次）: {last_error}"
                logger.error("索引构建任务 %s: %s", task_id, error_msg)
                _update_index_build_task(task_id, "failed", current_progress,
                                         f"生成失败（第 {i+1}/{total} 个块）", str(last_error)[:200])
                return {"status": "failed", "message": error_msg}

            # 更新进度
            progress = 10 + int((i + 1) / total * 85)
            if progress > current_progress:
                current_progress = progress
                _update_index_build_task(task_id, "building", progress, f"生成进度: {i+1}/{total}")

        # 4. 完成
        _update_index_build_task(task_id, "completed", 100, f"完成，共生成 {generated} 个索引问题")
        return {"status": "completed", "generated": generated, "total_chunks": total}

    except SoftTimeLimitExceeded:
        error_msg = f"任务超时（超过 {TASK_SOFT_TIME_LIMIT} 秒），被 Celery 终止"
        logger.error("索引构建任务 %s 软超时: %s", task_id, error_msg)
        try:
            _update_index_build_task(task_id, "failed", current_progress, "任务超时", error_msg)
        except TaskCancelled:
            pass
        return {"status": "failed", "message": error_msg}

    except TaskCancelled as exc:
        logger.warning("索引构建任务 %s 被取消: %s", task_id, exc)
        return {"status": "cancelled", "message": str(exc)}

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.error("索引构建任务 %s 异常: %s", task_id, error_msg)
        logger.error(traceback.format_exc())
        try:
            _update_index_build_task(task_id, "failed", current_progress, "任务异常", error_msg)
        except TaskCancelled:
            pass
        return {"status": "failed", "message": error_msg}
