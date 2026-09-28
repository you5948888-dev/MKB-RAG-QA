"""会话管理路由（Ch7）。Ch11 改造：从 app.py 拆分为独立模块。"""
import logging

from fastapi import APIRouter, HTTPException, Depends
import database
import auth
from auth import ensure_kb_access as _ensure_kb_access
from models import RenameSessionRequest

logger = logging.getLogger("rag.routes.chat")

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/chat/sessions")
def list_sessions(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """获取指定知识库的会话列表。"""
    _ensure_kb_access(kb_id, user)
    items = database.list_sessions(kb_id)
    return {"code": 0, "message": "ok", "data": {"items": items}}


@router.post("/chat/sessions")
def create_session(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """创建新的空会话。"""
    _ensure_kb_access(kb_id, user)
    session = database.create_session(kb_id)
    return {"code": 0, "message": "ok", "data": session}


@router.patch("/chat/sessions/{session_id}")
def rename_session(session_id: str, payload: RenameSessionRequest, user: dict = Depends(auth.get_current_user)):
    """重命名会话。"""
    session = database.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    _ensure_kb_access(session["kb_id"], user)

    ok = database.rename_session(session_id, payload.title)
    if not ok:
        raise HTTPException(status_code=500, detail="重命名失败")

    return {"code": 0, "message": "ok", "data": {"session_id": session_id, "title": payload.title}}


@router.delete("/chat/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(auth.get_current_user)):
    """删除会话及其所有消息。"""
    session = database.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    _ensure_kb_access(session["kb_id"], user)

    ok = database.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=500, detail="删除失败")

    return {"code": 0, "message": "ok", "data": {"session_id": session_id, "deleted": True}}


@router.get("/chat/sessions/{session_id}/messages")
def get_session_messages(session_id: str, user: dict = Depends(auth.get_current_user)):
    """获取会话的所有消息。"""
    session = database.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    _ensure_kb_access(session["kb_id"], user)

    messages = database.get_session_messages(session_id)
    return {"code": 0, "message": "ok", "data": {"items": messages, "session": session}}


@router.delete("/chat/sessions/{session_id}/messages")
def clear_session_messages(session_id: str, user: dict = Depends(auth.get_current_user)):
    """清空会话的所有消息。"""
    session = database.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    _ensure_kb_access(session["kb_id"], user)

    count = database.clear_session_messages(session_id)
    return {"code": 0, "message": "ok", "data": {"session_id": session_id, "cleared": count}}
