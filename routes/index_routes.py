"""索引问题管理路由（Ch8-Ch10）。Ch11 改造：从 app.py 拆分为独立模块。"""
import logging

from fastapi import APIRouter, HTTPException, Depends
import database
import embedding
import milvus_ops
import rag_pipeline
import auth
from auth import ensure_kb_access as _ensure_kb_access
from models import (
    IndexQuestionCreateRequest, IndexQuestionUpdateRequest, IndexQuestionLinkRequest,
    IndexRetrieveRequest, UnlinkRequest, IndexBuildRequest, CleanupRequest,
)

logger = logging.getLogger("rag.routes.index")

router = APIRouter(prefix="/api", tags=["index"])


@router.post("/kb/{kb_id}/index-questions")
def create_index_question(kb_id: str, payload: IndexQuestionCreateRequest, user: dict = Depends(auth.get_current_user)):
    """创建索引问题，同时将问题向量化存入 Milvus。"""
    _ensure_kb_access(kb_id, user)

    question_text = payload.question.strip()
    item = database.create_index_question(kb_id, question_text, source="manual")

    # 将索引问题向量化存入 Milvus
    try:
        emb = embedding.bge_model.embed(question_text)
        dense_vec = embedding.get_dense_vector(emb)
        milvus_ops.upsert_index_question_vector(item["index_id"], kb_id, question_text, dense_vec)
    except Exception as exc:
        logger.warning("索引问题向量写入 Milvus 失败: %s", exc)

    return {"code": 0, "message": "ok", "data": item}


@router.get("/kb/{kb_id}/index-questions")
def list_index_questions(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """获取索引问题列表。"""
    _ensure_kb_access(kb_id, user)
    items = database.list_index_questions(kb_id)
    return {"code": 0, "message": "ok", "data": {"items": items}}


@router.put("/kb/{kb_id}/index-questions/{index_id}")
def update_index_question(kb_id: str, index_id: str, payload: IndexQuestionUpdateRequest, user: dict = Depends(auth.get_current_user)):
    """修改索引问题，同时更新 Milvus 中的向量。"""
    _ensure_kb_access(kb_id, user)
    iq = database.get_index_question(index_id)
    if iq is None or iq["kb_id"] != kb_id:
        raise HTTPException(status_code=404, detail="索引问题不存在")

    question_text = payload.question.strip()
    ok = database.update_index_question(index_id, question_text)
    if not ok:
        raise HTTPException(status_code=500, detail="更新失败")

    # 更新 Milvus 中的向量
    try:
        emb = embedding.bge_model.embed(question_text)
        dense_vec = embedding.get_dense_vector(emb)
        milvus_ops.upsert_index_question_vector(index_id, kb_id, question_text, dense_vec)
    except Exception as exc:
        logger.warning("索引问题向量更新 Milvus 失败: %s", exc)

    return {"code": 0, "message": "ok", "data": {"index_id": index_id, "question": question_text}}


@router.delete("/kb/{kb_id}/index-questions/{index_id}")
def delete_index_question(kb_id: str, index_id: str, user: dict = Depends(auth.get_current_user)):
    """删除索引问题（级联删除绑定关系 + Milvus 向量）。"""
    _ensure_kb_access(kb_id, user)
    iq = database.get_index_question(index_id)
    if iq is None or iq["kb_id"] != kb_id:
        raise HTTPException(status_code=404, detail="索引问题不存在")

    ok = database.delete_index_question(index_id)
    if not ok:
        raise HTTPException(status_code=500, detail="删除失败")

    # 删除 Milvus 中的向量
    try:
        milvus_ops.delete_index_question_vector(index_id)
    except Exception as exc:
        logger.warning("删除索引问题向量失败: %s", exc)

    return {"code": 0, "message": "ok", "data": {"index_id": index_id, "deleted": True}}


@router.post("/kb/{kb_id}/index-questions/cleanup")
def cleanup_index_questions(kb_id: str, payload: CleanupRequest, user: dict = Depends(auth.get_current_user)):
    """清理索引问题。mode='orphan' 清理无关联的，mode='all' 全部清空。"""
    _ensure_kb_access(kb_id, user)
    mode = payload.mode
    if mode not in {"orphan", "all"}:
        raise HTTPException(status_code=400, detail="mode 仅支持 orphan / all")
    result = database.cleanup_index_questions(kb_id, mode=mode)
    if result["index_ids"]:
        try:
            if mode == "all":
                milvus_ops.delete_index_question_vectors_for_kb(kb_id)
            else:
                milvus_ops.delete_index_question_vectors_by_ids(result["index_ids"])
        except Exception as exc:
            logger.warning("清理索引问题向量失败: %s", exc)
    return {"code": 0, "message": "ok", "data": {"deleted_count": result["deleted_count"], "mode": mode}}


@router.post("/kb/{kb_id}/index-questions/{index_id}/links")
def bind_index_links(kb_id: str, index_id: str, payload: IndexQuestionLinkRequest, user: dict = Depends(auth.get_current_user)):
    """为索引问题绑定 chunk。"""
    _ensure_kb_access(kb_id, user)
    iq = database.get_index_question(index_id)
    if iq is None or iq["kb_id"] != kb_id:
        raise HTTPException(status_code=404, detail="索引问题不存在")

    items = [item.model_dump() for item in payload.items]
    try:
        written = database.add_index_links(index_id, kb_id, items)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"code": 0, "message": "ok", "data": {"index_id": index_id, "written": written}}


@router.delete("/kb/{kb_id}/index-questions/{index_id}/links")
def remove_index_link(kb_id: str, index_id: str, payload: UnlinkRequest, user: dict = Depends(auth.get_current_user)):
    """解除索引问题与某个 chunk 的绑定。"""
    _ensure_kb_access(kb_id, user)
    iq = database.get_index_question(index_id)
    if iq is None or iq["kb_id"] != kb_id:
        raise HTTPException(status_code=404, detail="索引问题不存在")

    ok = database.remove_index_link(index_id, payload.chunk_id)
    if not ok:
        raise HTTPException(status_code=404, detail="绑定关系不存在")

    return {"code": 0, "message": "ok", "data": {"index_id": index_id, "unlinked": True}}


@router.get("/kb/{kb_id}/index-questions/{index_id}/links")
def get_index_links(kb_id: str, index_id: str, user: dict = Depends(auth.get_current_user)):
    """查看索引问题的绑定。"""
    _ensure_kb_access(kb_id, user)
    iq = database.get_index_question(index_id)
    if iq is None or iq["kb_id"] != kb_id:
        raise HTTPException(status_code=404, detail="索引问题不存在")

    links = database.get_index_links(index_id)
    return {"code": 0, "message": "ok", "data": {"index_id": index_id, "question": iq["question"], "items": links}}


@router.get("/kb/{kb_id}/index-chunk-candidates")
def get_chunk_candidates(
    kb_id: str,
    index_id: str | None = None,
    link_filter: str = "all",
    file_name: str | None = None,
    keyword: str | None = None,
    user: dict = Depends(auth.get_current_user),
):
    """获取知识库中所有 chunk（用于候选浏览和绑定），支持筛选和关联状态。"""
    _ensure_kb_access(kb_id, user)

    chunks = database.get_chunks_for_index(
        kb_id, index_id=index_id, link_filter=link_filter,
        file_name=file_name, keyword=keyword,
    )
    return {"code": 0, "message": "ok", "data": {"items": chunks}}


@router.post("/kb/{kb_id}/index-retrieve")
def index_retrieve(kb_id: str, payload: IndexRetrieveRequest, user: dict = Depends(auth.get_current_user)):
    """索引检索：通过索引问题匹配返回相关 chunk。"""
    _ensure_kb_access(kb_id, user)

    try:
        result = rag_pipeline.recall_with_index_questions(
            kb_id=kb_id,
            query=payload.query,
            top_k=payload.top_k,
            min_score=payload.min_score,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"索引检索失败: {exc}")

    return {"code": 0, "message": "ok", "data": {"kb_id": kb_id, "query": payload.query, **result}}


@router.post("/kb/{kb_id}/index-questions/build")
def build_index_questions(kb_id: str, payload: IndexBuildRequest | None = None, user: dict = Depends(auth.get_current_user)):
    """自动为知识库的所有 chunk 生成索引问题（Ch11 改为异步任务，由 Worker 逐块调用 LLM 生成）。"""
    _ensure_kb_access(kb_id, user)

    chunks = database.get_chunks_for_index(kb_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="知识库中没有已索引的 chunk")

    prompt_text = (payload and payload.prompt) or ""
    try:
        task = database.create_index_build_task(kb_id, prompt=prompt_text)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # 发送到 Celery 队列（Ch11：异步投递）
    try:
        from tasks import process_index_build_task
        process_index_build_task.delay(task["task_id"])
    except Exception as exc:
        # 仅当任务仍为 queued 时标记 dispatch_unknown，保留任务供查询
        database.mark_index_build_dispatch_unknown(task["task_id"], str(exc))
        raise HTTPException(
            status_code=503,
            detail={"message": "发送结果待确认", "task_id": task["task_id"]},
        )

    return {"code": 0, "message": "ok",
            "data": {"task_id": task["task_id"], "kb_id": kb_id, "status": "queued",
                     "step_message": task["step_message"], "total_chunks": len(chunks)}}
