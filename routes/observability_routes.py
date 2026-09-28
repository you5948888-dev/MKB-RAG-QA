"""全链路可观测性路由（Advanced RAG 迭代 · 第 4 期）。

- 运行观测面板：`GET /api/observability/overview`
- 请求列表与单次详情：`GET /api/observability/traces`、`GET /api/observability/traces/{trace_id}`
- 文档链路详情（trace_id 即 task_id）：`GET /api/observability/tasks/{task_id}`
- 手动清理（不自动清理）：`POST /api/observability/traces/cleanup`，仅管理员

权限：普通用户只能看到自己名下知识库的记录；清理只允许管理员。
"""
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

import auth
import database
import observability
import langsmith_tracing
from models import CleanupTracesRequest

logger = logging.getLogger("rag.routes.observability")

router = APIRouter(prefix="/api/observability", tags=["observability"])


def _owned_kb_ids(user: dict) -> list[str] | None:
    """admin 返回 None（不过滤）；普通用户返回其可见的知识库 ID 列表。"""
    if user["role"] == "admin":
        return None
    return auth.get_user_owned_kb_ids(user["user_id"])


@router.get("/overview")
def observability_overview(hours: int = 24, biz_type: str | None = None,
                           with_dependencies: bool = True,
                           user: dict = Depends(auth.get_current_user)):
    """运行观测面板数据：问答量/成功率/耗时/首 Token/无召回比/能力使用次数/文档任务/依赖状态/最近错误。"""
    dependencies = observability.check_dependencies() if with_dependencies else {}
    owned = _owned_kb_ids(user)
    data = observability.overview(hours=hours, biz_type=biz_type, dependencies=dependencies,
                                 kb_ids=owned)
    if owned is not None:
        data["scope"] = {"kb_ids": owned, "note": "仅统计你名下的知识库"}
    # 可选第三方观测（LangSmith）的状态：只回传掩码 Key 与项目链接，不含明文
    data["langsmith"] = langsmith_tracing.status()
    return {"code": 0, "message": "ok", "data": data}


@router.get("/traces")
def observability_traces(limit: int = 20, offset: int = 0, biz_type: str | None = None,
                         status: str | None = None, hours: int | None = None,
                         user: dict = Depends(auth.get_current_user)):
    """请求追踪列表（分页；支持类型/状态/时间快捷范围筛选）。

    hours：只返回最近 N 小时内的记录（1=最近 1 小时，24=最近 24 小时…），
    与清理接口的 `mode=range` 使用同一套时间口径，保证"看到的就是能删的"。
    """
    limit = max(1, min(limit, 100))
    owned = _owned_kb_ids(user)
    data = observability.list_traces(limit=limit, offset=max(offset, 0), biz_type=biz_type,
                                     status=status, hours=hours, kb_ids=owned)
    return {"code": 0, "message": "ok", "data": data}


@router.get("/traces/export")
def observability_export(limit: int = 2000, user: dict = Depends(auth.require_role("admin"))):
    """导出观测数据（JSON 下载，管理员）。

    【路由顺序】必须声明在 `/traces/{trace_id}` **之前**，否则 "export" 会被当成 trace_id。
    用途：清理前先备份（观测数据不自动清理，删掉不可恢复）。
    """
    data = observability.export_data(limit=limit)
    filename = f"rag-observability-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    return JSONResponse(
        content={"code": 0, "message": "ok", "data": data},
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/traces/{trace_id}")
def observability_trace_detail(trace_id: str, user: dict = Depends(auth.get_current_user)):
    """单次请求详情：阶段耗时、问题改写、扩展问题、假设答案、配置快照、候选与降级。"""
    trace = observability.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail={"code": 40404, "message": "追踪记录不存在"})
    owned = _owned_kb_ids(user)
    if owned is not None and trace.get("kb_id") and trace["kb_id"] not in owned:
        raise HTTPException(status_code=404, detail={"code": 40401, "message": "知识库不存在或无权访问"})
    return {"code": 0, "message": "ok", "data": trace}


@router.get("/tasks/{task_id}")
def observability_task_detail(task_id: str, user: dict = Depends(auth.get_current_user)):
    """文档链路详情：任务进度（父块/子块/已嵌入/重试）+ 各阶段耗时。"""
    task = database.get_upload_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"code": 40405, "message": "任务不存在"})
    auth.ensure_kb_access(task["kb_id"], user)
    trace = observability.get_trace(task_id) or {}
    return {"code": 0, "message": "ok", "data": {
        "task": task,
        "trace": {key: value for key, value in trace.items() if key != "spans"},
        "spans": trace.get("spans", []),
    }}


@router.post("/traces/cleanup")
def observability_cleanup(payload: CleanupTracesRequest,
                          user: dict = Depends(auth.require_role("admin"))):
    """手动清理观测数据（默认 dry_run，只返回将删除的条数）。"""
    try:
        result = observability.cleanup_traces(
            mode=payload.mode, days=payload.days, hours=payload.hours,
            trace_ids=payload.trace_ids, dry_run=payload.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": 40007, "message": str(exc)})
    logger.info("观测数据清理: mode=%s dry_run=%s → %s", payload.mode, payload.dry_run, result)
    return {"code": 0, "message": "ok", "data": result}
