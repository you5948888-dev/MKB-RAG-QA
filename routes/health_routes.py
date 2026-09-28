"""健康检查路由（Advanced RAG 迭代 · 第 4 期）。

区分两种语义（需求四.5）：
  - **存活（live）**：应用进程是否正常响应，不探测任何外部依赖；
  - **就绪（ready）**：数据库 / Redis / Milvus / Embedding / 大模型是否真的可用，
    并逐项返回状态、耗时与失败原因，而不是笼统的"系统异常"。

就绪接口在任一依赖不可用时返回 503，便于编排系统（K8s / Compose）正确处置。
"""
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import observability

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live():
    """存活探针：只要进程能响应就返回 200。"""
    return {
        "status": "alive",
        "service": "rag-tutorial",
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }


@router.get("/health/ready")
def ready():
    """就绪探针：逐项探测外部依赖，任一不可用返回 503 并说明是哪一项、什么原因。"""
    report = observability.check_dependencies()
    status_code = 200 if report["status"] == "ok" else 503
    return JSONResponse(status_code=status_code, content=report)
