"""全局中间件：统一异常处理 + 请求日志（Ch11 新增）。"""
import time
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("rag.middleware")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """全局异常处理：兜底捕获未处理异常，统一返回标准错误格式。

    注册在 RequestLoggingMiddleware 外层（后注册 = 外层）。
    用户只会看到 {"code": 50000, ...}，不会看到堆栈信息。
    HTTPException 由 FastAPI 自带的 ExceptionMiddleware 提前处理，
    不会到达这里；这里只兜底非 HTTP 异常。
    """

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.exception("未处理异常: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"code": 50000, "message": f"服务器内部错误: {type(exc).__name__}: {str(exc)[:200]}"},
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志：记录每个请求的方法、路径、状态码和耗时。

    注册在 ErrorHandlerMiddleware 内层（先注册 = 内层）。
    每个请求都会输出一行日志，运维可以据此定位慢接口。
    """

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
        return response
