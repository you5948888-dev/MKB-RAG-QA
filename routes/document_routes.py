"""文档管理路由（Ch3-Ch10）。Ch11 改造：从 app.py 拆分为独立模块。"""
import os
import json
import time
from uuid import uuid4
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from config import UPLOAD_DIR
import database
import chunking
import milvus_ops
import observability
import auth
from auth import ensure_kb_access as _ensure_kb_access

logger = logging.getLogger("rag.routes.document")

router = APIRouter(prefix="/api", tags=["document"])


def _resolve_preview_config(kb: dict | None, chunk_config_json: str | None = None,
                            chunk_size: int | None = None, chunk_overlap: int | None = None):
    """解析预览使用的分块配置，保证预览与入库同源（Advanced RAG 第 1 期）。

    优先级：显式覆盖（chunk_config JSON / chunk_size / chunk_overlap）
            > 知识库当前配置 > 系统默认值。
    返回 (config, 是否使用了临时覆盖)。
    """
    base = chunking.config_from_kb(kb) if kb else chunking.validate_config({})
    override: dict = {}
    if chunk_config_json:
        try:
            parsed = json.loads(chunk_config_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400,
                                detail={"code": 40003, "message": "chunk_config 不是合法的 JSON"})
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400,
                                detail={"code": 40003, "message": "chunk_config 必须是 JSON 对象"})
        override.update(parsed)
    if chunk_size is not None:
        override["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        override["chunk_overlap"] = chunk_overlap

    if not override:
        return base, False
    try:
        merged = chunking.validate_config({**base, **override})
    except chunking.ChunkConfigError as exc:
        raise HTTPException(status_code=400, detail={"code": 40003, "message": str(exc)})
    # overridden 表示"本次预览的配置确实与知识库当前配置不同"，
    # 而不是"客户端传了覆盖参数"——避免前端回传同一份配置时误导用户。
    return merged, merged != base


@router.post("/documents/upload")
async def upload_document(
    kb_id: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(auth.get_current_user),
):
    """上传文档到指定知识库，创建异步处理任务（Ch10 改为异步）。

    异步（Ch10）：保存文件 + 创建任务 + 投递 Celery 后立即返回 task_id，
    由后台 Worker 慢慢处理，前端轮询 /api/documents 查看进度。
    """
    kb = _ensure_kb_access(kb_id, user)
    try:
        chunk_config = chunking.config_from_kb(kb)
    except chunking.ChunkConfigError as exc:
        raise HTTPException(status_code=400,
                            detail={"code": 40003, "message": f"知识库分块配置非法: {exc}"})

    # 校验文件名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    safe_filename = os.path.basename(file.filename)
    ext = os.path.splitext(safe_filename)[1].lower()
    # 注：教程示例只允许 .pdf/.txt/.md；本项目从 Ch3 起一直支持 .docx，保留避免功能回退
    if ext not in {".pdf", ".txt", ".md", ".docx"}:
        raise HTTPException(status_code=400, detail="仅支持 PDF / TXT / MD / DOCX 文件")

    # 保存文件到 UPLOAD_DIR
    dest_path = os.path.join(UPLOAD_DIR, f"{uuid4().hex}{ext}")
    _save_started = time.time()
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk_data = await file.read(1024 * 1024)
                if not chunk_data:
                    break
                f.write(chunk_data)
    except Exception as exc:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"保存文件失败: {exc}")
    finally:
        await file.close()

    # 记录到数据库
    file_size = os.path.getsize(dest_path)
    try:
        doc = database.create_document(
            kb_id=kb_id,
            filename=safe_filename,
            path=dest_path,
            file_size=file_size,
        )
    except Exception:
        os.remove(dest_path)
        raise
    if doc is None:
        # 同一知识库下文件名重复
        os.remove(dest_path)
        raise HTTPException(
            status_code=409,
            detail={"code": 40902, "message": f"文件 '{safe_filename}' 已存在于该知识库中"},
        )

    # 更新知识库的文档计数
    database.update_kb_doc_count(kb_id)

    # 创建上传任务（Ch10 新增）：把分块配置**冻结**进任务，Worker 只认这份快照，
    # 避免排队期间知识库改了配置，导致入库结果与当时预览的结果不一致。
    try:
        task = database.create_upload_task(
            kb_id=kb_id, doc_id=doc["doc_id"], filename=safe_filename,
            filepath=dest_path, chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            chunk_config=chunking.canonical_json(chunk_config),
        )
    except Exception as exc:
        # 任务创建失败（如已有运行中任务）：回滚文档记录和物理文件
        database.delete_document(doc["doc_id"])
        if os.path.exists(dest_path):
            os.remove(dest_path)
        database.update_kb_doc_count(kb_id)
        raise HTTPException(status_code=503, detail=f"创建任务失败: {exc}")

    # 发送到 Celery 队列（Ch10：异步投递）
    try:
        from tasks import process_upload_task
        process_upload_task.delay(task["task_id"])
    except Exception as exc:
        # delay 抛异常 ≠ 消息一定没送达：仅当任务仍为 uploaded 时标记 dispatch_unknown，
        # 保留任务、文档和文件，让用户查询任务状态而不是盲目重传。
        database.mark_upload_dispatch_unknown(task["task_id"], str(exc))
        raise HTTPException(
            status_code=503,
            detail={"message": "发送结果待确认，请查询任务状态，不要重复上传", "task_id": task["task_id"]},
        )

    # 观测（第 4 期）：文件保存发生在路由侧，作为文档链路的第一阶段追记到该任务的追踪里
    observability.record_span(
        task["task_id"], observability.STAGE_FILE_SAVE,
        duration_ms=int((time.time() - _save_started) * 1000),
        output_count=file_size, filename=safe_filename, suffix=ext,
    )

    return {
        "code": 0,
        "message": "ok",
        "data": {**doc, "task_id": task["task_id"], "status": "uploaded", "step_message": task["step_message"]},
    }


@router.post("/documents/{doc_id}/index")
def index_document(doc_id: str, user: dict = Depends(auth.get_current_user)):
    """手动触发索引构建（Ch10：改为创建上传任务，复用同一 Worker，返回 task_id）。"""
    doc = database.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "文档不存在"})
    kb = _ensure_kb_access(doc["kb_id"], user)
    try:
        chunk_config = chunking.config_from_kb(kb)
    except chunking.ChunkConfigError as exc:
        raise HTTPException(status_code=400,
                            detail={"code": 40003, "message": f"知识库分块配置非法: {exc}"})
    try:
        task = database.create_upload_task(
            kb_id=doc["kb_id"], doc_id=doc["doc_id"], filename=doc["filename"],
            filepath=doc["path"], chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            chunk_config=chunking.canonical_json(chunk_config), task_type="reindex",
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"创建任务失败: {exc}")

    try:
        from tasks import process_upload_task
        process_upload_task.delay(task["task_id"])
    except Exception as exc:
        database.mark_upload_dispatch_unknown(task["task_id"], str(exc))
        raise HTTPException(status_code=503,
                            detail={"message": "发送结果待确认", "task_id": task["task_id"]})

    return {"code": 0, "message": "ok",
            "data": {"doc_id": doc_id, "task_id": task["task_id"], "status": "uploaded"}}


@router.get("/documents")
def list_documents(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """获取指定知识库的文档列表（含任务处理状态，Ch10 新增）。

    Advanced RAG 第 2 期：附带 `config_outdated`——该文档实际入库配置与知识库
    当前分块配置不一致时为 true，前端据此提示"需重新索引"。
    """
    kb = _ensure_kb_access(kb_id, user)
    items = database.list_documents(kb_id)
    current_hash = chunking.config_hash(chunking.config_from_kb(kb))
    # 关联最新的上传任务状态
    for doc in items:
        doc_hash = doc.get("chunk_config_hash") or ""
        doc["config_outdated"] = bool(doc_hash) and doc_hash != current_hash
        tasks = database.get_tasks_by_doc_id(doc["doc_id"])
        if tasks:
            latest = tasks[0]
            doc["task_status"] = latest["status"]
            doc["task_step_message"] = latest["step_message"] or ""
        else:
            # 无任务记录（旧数据或直接索引的文档），视为已完成
            doc["task_status"] = "completed"
            doc["task_step_message"] = ""
    return {"code": 0, "message": "ok", "data": {"items": items}}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user: dict = Depends(auth.get_current_user)):
    """删除文档（同时删除 Milvus 向量、SQLite chunk 和索引问题绑定）。
    Ch11：索引构建期间删除 → database.delete_document 抛 ValueError → 409。"""
    doc = database.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "文档不存在"})
    _ensure_kb_access(doc["kb_id"], user)
    # SQLite 删除是业务事实源；database.delete_document 会级联清理
    # chunks / 索引绑定 / 上传任务（处理中的任务标记 failed）
    try:
        database.delete_document(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    # 物理文件删除失败时记录告警，但继续删向量与刷新统计，
    # 避免数据库已删而向量仍可被检索。
    try:
        if doc.get("path") and os.path.exists(doc["path"]):
            os.remove(doc["path"])
    except OSError as exc:
        logger.warning("物理文件清理失败，需要人工清理: %s", exc)
    # 清理 Milvus 中的向量（SQLite 中的 chunk 已在 delete_document 里级联删除）
    try:
        milvus_ops.delete_vectors_for_doc(doc["kb_id"], doc["filename"])
    except Exception as exc:
        logger.warning("删除 Milvus 向量失败: %s", exc)
    database.update_kb_doc_count(doc["kb_id"])
    database.update_kb_chunk_count(doc["kb_id"])
    return {"code": 0, "message": "ok", "data": {"doc_id": doc_id, "deleted": True}}


@router.post("/documents/{doc_id}/preview-chunks")
def preview_chunks(doc_id: str, chunk_size: int | None = Form(None), chunk_overlap: int | None = Form(None),
                   chunk_config: str | None = Form(None), user: dict = Depends(auth.get_current_user)):
    """预览文档的分块效果。

    Advanced RAG 第 1 期：默认使用**知识库当前分块配置**，因此预览结果与入库结果一致
    （旧版固定 800/100 默认值会让两者不一致）。传 `chunk_config`(JSON) 或
    `chunk_size`/`chunk_overlap` 可临时覆盖，仅影响本次预览、不落库。
    """
    doc = database.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "文档不存在"})
    kb = _ensure_kb_access(doc["kb_id"], user)
    if not os.path.exists(doc["path"]):
        raise HTTPException(status_code=404, detail="文档文件不存在")

    config, overridden = _resolve_preview_config(kb, chunk_config, chunk_size, chunk_overlap)
    built = chunking.build_document_chunks(doc["path"], config)
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "doc": doc,
            "chunks": built["children"],            # 可检索叶子块（与实际入库一致）
            "parents": chunking.group_parents(built),  # 父子层级（通用分块时为空）
            "strategy": built["strategy"],
            "total_chunks": len(built["children"]),
            "total_parents": len(built["parents"]),
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "config": config,
            "overridden": overridden,
        },
    }


@router.post("/documents/preview")
async def preview_upload_file(
    file: UploadFile = File(...),
    chunk_size: int | None = Form(None),
    chunk_overlap: int | None = Form(None),
    kb_id: str | None = Form(None),
    chunk_config: str | None = Form(None),
    user: dict = Depends(auth.get_current_user),
):
    """上传前预览分块：接收未入库的文件，临时解析并分块返回，不入库不建索引。

    传 `kb_id` 时按该知识库的当前分块配置预览（与入库一致）；也可用
    `chunk_config` / `chunk_size` / `chunk_overlap` 临时覆盖做参数对比。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".txt", ".md", ".docx"}:
        raise HTTPException(status_code=400, detail="仅支持 PDF / TXT / MD / DOCX 文件")

    kb = _ensure_kb_access(kb_id, user) if kb_id else None
    config, overridden = _resolve_preview_config(kb, chunk_config, chunk_size, chunk_overlap)

    # 临时保存到 UPLOAD_DIR 进行解析，预览后删除
    dest_path = os.path.join(UPLOAD_DIR, f"preview_{uuid4().hex}{ext}")
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk_data = file.file.read(1024 * 1024)
                if not chunk_data:
                    break
                f.write(chunk_data)
        chunks = chunking.chunk_document(dest_path, config)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"解析文件失败: {exc}")
    finally:
        await file.close()
        if os.path.exists(dest_path):
            os.remove(dest_path)

    # 只返回块内容，不带 path 等无关字段
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "filename": file.filename,
            "kb_id": kb_id,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "config": config,
            "overridden": overridden,
        },
    }
