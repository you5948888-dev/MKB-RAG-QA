"""RAG 问答路由（Ch6-Ch7）。Ch11 改造：从 app.py 拆分为独立模块。"""
import json
import os
import logging
from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from config import DATA_DIR
import database
import observability
import rag_pipeline
import auth
from auth import ensure_kb_access as _ensure_kb_access
from models import QueryRequest, FeedbackRequest

logger = logging.getLogger("rag.routes.qa")

router = APIRouter(prefix="/api", tags=["qa"])


@router.post("/qa/query")
def qa_query(payload: QueryRequest, user: dict = Depends(auth.get_current_user)):
    """
    RAG 问答：检索相关文档 → 调用 LLM 生成回答。

    流程：
    1. 获取或创建会话
    2. 保存用户问题
    3. 检索相关文档
    4. 拼接上下文 → LLM 生成
    5. 保存回答 + 引用
    """
    # 校验知识库访问权限
    _ensure_kb_access(payload.kb_id, user)

    # Advanced RAG 第 3 期：允许本次请求临时覆盖检索策略（不写库）
    strategy_override = dict(payload.strategy or {})
    if payload.mode:
        strategy_override["mode"] = payload.mode

    # Advanced RAG 第 4 期：每次问答生成唯一追踪编号，全过程写进 rag_traces / rag_spans
    trace = observability.TraceRecorder.start(
        "qa", kb_id=payload.kb_id, session_id=payload.session_id, question=payload.question,
    )
    with observability.active_trace(trace):
        with trace.span(observability.STAGE_RECEIVE, input_count=1) as span:
            span.set_output(1, mode=strategy_override.get("mode") or "kb_default")
        try:
            result = rag_pipeline.rag_query(
                kb_id=payload.kb_id,
                question=payload.question,
                session_id=payload.session_id,
                context_turns=3,  # Ch7 新增：固定 3 轮上下文
                strategy=strategy_override or None,
            )
        except ValueError as exc:
            trace.finish(status=observability.STATUS_FAILED, error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            trace.finish(status=observability.STATUS_FAILED, error=str(exc))
            raise HTTPException(status_code=500, detail=f"问答失败: {exc}")

    plan = result.get("query_plan") or {}
    trace.set_result(
        rewritten_question=plan.get("rewritten") or None,
        expansions=plan.get("multi_queries"),
        hyde_answer=plan.get("hyde_answer"),
        config_snapshot=result.get("strategy_snapshot"),
        recall_empty=not result.get("citations"),
    )
    # 【注意】降级由检索管道（recall_pipeline）在发生处**直接写进本 trace**（span + 降级记录）。
    # 这里曾再遍历 result["degradations"] 补记一次，导致同一次降级出现两条阶段行、两条记录；
    # 现统一由管道单点负责，路由不再重复登记（result["degradations"] 仍原样返回给前端展示）。
    trace.finish(status=observability.STATUS_OK)

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "answer_id": str(uuid4()),
            "trace_id": trace.trace_id,
            **result,
        },
    }


@router.post("/qa/feedback")
def qa_feedback(payload: FeedbackRequest, user: dict = Depends(auth.get_current_user)):
    """保存用户对回答的反馈评分。"""
    feedback_file = os.path.join(DATA_DIR, "feedback.json")
    feedback = {
        "answer_id": payload.answer_id,
        "score": payload.score,
        "text": payload.text,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    existing = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.append(feedback)
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return {"code": 0, "message": "ok", "data": feedback}
