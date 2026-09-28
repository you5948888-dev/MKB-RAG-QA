"""异步任务管理路由（Ch10-Ch11）。Ch11 改造：从 app.py 拆分为独立模块。"""
import logging

from fastapi import APIRouter, HTTPException, Depends
import database
import auth

logger = logging.getLogger("rag.routes.task")

router = APIRouter(prefix="/api", tags=["task"])


@router.get("/upload-tasks/{task_id}")
def get_upload_task(task_id: str, user: dict = Depends(auth.get_current_user)):
    """查询单个上传任务的状态和进度。"""
    task = database.get_upload_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    auth.ensure_kb_access(task["kb_id"], user)
    return {"code": 0, "message": "ok", "data": task}


@router.get("/upload-tasks")
def list_upload_tasks(kb_id: str | None = None, user: dict = Depends(auth.get_current_user)):
    """获取上传任务列表，可按知识库过滤。"""
    if kb_id:
        auth.ensure_kb_access(kb_id, user)
    database.cleanup_stale_tasks()  # Ch10：清理上次 Worker 崩溃留下的僵尸任务
    items = database.list_upload_tasks(kb_id)
    if user["role"] != "admin":
        owned = set(auth.get_user_owned_kb_ids(user["user_id"]))
        items = [item for item in items if item["kb_id"] in owned]
    return {"code": 0, "message": "ok", "data": {"items": items}}


@router.delete("/upload-tasks/{task_id}")
def delete_upload_task(task_id: str, user: dict = Depends(auth.get_current_user)):
    """删除上传任务记录（只允许删除终态任务）。"""
    task = database.get_upload_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    auth.ensure_kb_access(task["kb_id"], user)
    ok = database.delete_upload_task(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="任务不存在或状态不允许删除")
    return {"code": 0, "message": "ok", "data": {"task_id": task_id, "deleted": True}}


@router.post("/upload-tasks/{task_id}/retry")
def retry_upload_task(task_id: str, user: dict = Depends(auth.get_current_user)):
    """重试失败的任务：将状态重置，重新发送到 Celery 队列。"""
    existing = database.get_upload_task(task_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    auth.ensure_kb_access(existing["kb_id"], user)
    task = database.retry_upload_task(task_id)
    if task is None:
        raise HTTPException(status_code=400,
                            detail="任务不存在、状态不是 failed 或文档已被删除，无法重试")

    # 重新发送到 Celery 队列
    try:
        from tasks import process_upload_task
        process_upload_task.delay(task_id)
    except Exception as exc:
        database.mark_upload_dispatch_unknown(task_id, str(exc))
        raise HTTPException(status_code=503,
                            detail={"message": "发送结果待确认", "task_id": task_id})

    return {"code": 0, "message": "ok",
            "data": {"task_id": task_id, "status": "uploaded"}}


# ── 索引构建任务（Ch11 新增）──────────────────────────

@router.get("/index-build-tasks/{task_id}")
def get_index_build_task(task_id: str, user: dict = Depends(auth.get_current_user)):
    """查询索引构建任务的状态和进度。"""
    task = database.get_index_build_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    auth.ensure_kb_access(task["kb_id"], user)
    return {"code": 0, "message": "ok", "data": task}


@router.get("/index-build-tasks")
def list_index_build_tasks(kb_id: str | None = None, user: dict = Depends(auth.get_current_user)):
    """获取索引构建任务列表，可按知识库过滤。"""
    if kb_id:
        auth.ensure_kb_access(kb_id, user)
    # Ch11：清理僵尸索引构建任务（building 超时），避免其永久阻塞删除知识库
    database.cleanup_stale_index_build_tasks()
    items = database.list_index_build_tasks(kb_id)
    if user["role"] != "admin":
        owned = set(auth.get_user_owned_kb_ids(user["user_id"]))
        items = [item for item in items if item["kb_id"] in owned]
    return {"code": 0, "message": "ok", "data": {"items": items}}


@router.delete("/index-build-tasks/{task_id}")
def delete_index_build_task(task_id: str, user: dict = Depends(auth.get_current_user)):
    """删除索引构建任务记录（只允许删除终态任务）。"""
    task = database.get_index_build_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    auth.ensure_kb_access(task["kb_id"], user)
    ok = database.delete_index_build_task(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="任务不存在或状态不允许删除")
    return {"code": 0, "message": "ok", "data": {"task_id": task_id, "deleted": True}}