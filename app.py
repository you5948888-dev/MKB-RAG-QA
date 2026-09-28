"""
FastAPI 应用入口。

Ch11 改造：从单体拆分为模块化路由结构。
  - 路由按业务域拆分到 routes/ 目录
  - 全局异常处理和请求日志通过中间件统一管理
  - 敏感配置通过 .env 文件管理
Ch13 补充：新增 tests/ 离线测试与 Docker 部署（Dockerfile / compose.yaml），
  入口本身不再膨胀——测试通过 TestClient 复用本模块的 app 实例。
Ch14 补充：镜像带版本交付（rag-tutorial:0.14.0），运行时以非 root 用户启动，
  应用数据改为命名卷挂载；本入口无需为容器化改动任何代码。
"""
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from config import FRONTEND_DIR
import database
import milvus_ops
import auth
import middleware

# 注册子路由
from routes import kb_routes, document_routes, qa_routes, chat_routes, index_routes, task_routes
from routes import qa_stream_routes  # Ch12 新增：三种流式问答
from routes import observability_routes, health_routes  # 迭代P4：全链路观测与健康检查

# ── 日志配置 ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.app")

# ── FastAPI 实例 ─────────────────────────────────────
app = FastAPI(
    title="RAG 知识库系统",
    description="企业级 RAG 知识库问答系统",
    version="0.14.0",  # Ch14: "0.14.0"（Ch13: "0.13.0"）
)

# 注册中间件（洋葱模型：先注册 RequestLogging 在内存，后注册 ErrorHandler 在外层兜底）
app.add_middleware(middleware.RequestLoggingMiddleware)
app.add_middleware(middleware.ErrorHandlerMiddleware)

# 注册路由
app.include_router(kb_routes.router)
app.include_router(document_routes.router)
app.include_router(qa_routes.router)
app.include_router(chat_routes.router)
app.include_router(index_routes.router)
app.include_router(task_routes.router)
app.include_router(qa_stream_routes.router)  # Ch12 新增
app.include_router(observability_routes.router)  # 迭代P4：观测面板 / 执行详情 / 手动清理
app.include_router(health_routes.router)         # 迭代P4：存活与就绪探针（无 /api 前缀）
app.include_router(auth.router)  # Ch9：认证路由

# 挂载前端静态文件
app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR)), name="ui")


@app.get("/")
def root():
    return RedirectResponse(url="/ui/kb.html")


# ── 启动时初始化数据库、认证系统与 Milvus 集合 ─────────────
@app.on_event("startup")
def on_startup():
    database.init_db()
    # Advanced RAG 第 2 期迁移：旧文档没有"实际入库配置指纹"，用其知识库当前配置补齐。
    # 旧库从未改过分块配置，因此当前配置就是这些文档当初的入库配置。
    try:
        import chunking
        backfilled = 0
        for kb in database.list_kbs(page_size=None)["items"]:
            current_hash = chunking.config_hash(chunking.config_from_kb(kb))
            backfilled += database.backfill_doc_config_hash(kb["kb_id"], current_hash)
        if backfilled:
            logger.info("已为 %d 篇旧文档补齐分块配置指纹", backfilled)
    except Exception as exc:
        logger.warning("补齐分块配置指纹失败（不影响启动）: %s", exc)
    auth.init_auth_tables()
    auth.ensure_default_admin()
    logger.info("认证系统已就绪（默认管理员: admin / 123456）")
    try:
        milvus_ops.ensure_collection()
        milvus_ops.ensure_index_question_collection()  # Ch8 新增
        logger.info("Milvus 集合已就绪")
    except Exception as exc:
        logger.warning("Milvus 集合初始化失败（上传文档时会重试）: %s", exc)
    # Ch10：清理上次 Worker 崩溃留下的僵尸上传任务
    database.cleanup_stale_tasks()
    # Ch11：清理僵尸索引构建任务，避免其永久阻塞删除知识库
    database.cleanup_stale_index_build_tasks()