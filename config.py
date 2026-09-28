"""
项目配置：集中管理路径、数据库、外部服务地址等配置。
Ch11 改造：敏感配置从 .env 文件读取，路径配置保留硬编码。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── 路径配置 ────────────────────────────────────────────
# 项目根目录（code/ 所在目录）
BASE_DIR = Path(__file__).resolve().parent

# 加载项目根目录的 .env（未找到则回退到 os.getenv 的默认值）
load_dotenv(BASE_DIR / ".env")

# 数据目录（存放数据库、上传文件等）
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 上传文件目录
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 前端静态文件目录
FRONTEND_DIR = BASE_DIR / "frontend"

# ── 数据库配置 ──────────────────────────────────────────
# SQLite 数据库文件路径
CHAT_DB_PATH = str(DATA_DIR / "chat_memory.db")

# ── Milvus 配置 ─────────────────────────────────────────
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_tutorial_hybrid")
# 索引问题向量集合（Ch8 新增）
INDEX_QUESTION_COLLECTION = os.getenv("INDEX_QUESTION_COLLECTION", "rag_tutorial_index_questions")

# ── BGE 配置 ────────────────────────────────────────────
# 支持稀疏索引的 BGE-M3 服务（dense + sparse 都从这里取）
BGE_URL = os.getenv("BGE_URL", "http://localhost:9500")
BGE_DENSE_DIM = int(os.getenv("BGE_DENSE_DIM", "1024"))

# ── Pipeline 配置 ──────────────────────────────────────
# RAG Pipeline 功能开关配置
PIPELINE_CONFIG_PATH = str(BASE_DIR / "pipeline_config.json")

# ── LLM 配置 ───────────────────────────────────────────
# 大语言模型 API 地址（OpenAI 兼容格式）
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:9097/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen3-32B-FP8")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))

# ── Celery 配置（Ch10 新增）─────────────────────────────
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
UPLOAD_TASK_QUEUE = os.getenv("UPLOAD_TASK_QUEUE", "upload_queue")

# 任务超时保护（秒）
PARSE_TIMEOUT = int(os.getenv("PARSE_TIMEOUT", "180"))
INDEX_TIMEOUT = int(os.getenv("INDEX_TIMEOUT", "900"))

# ── 服务配置 ────────────────────────────────────────────
# FastAPI 服务地址与端口
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8013"))

# ── 安全配置（Ch9 新增）─────────────────────────────────
# 【安全】签名密钥必须来自 .env，禁止使用代码内置的教学默认值。
# 硬编码密钥或占位符被识别为不安全配置，直接拒绝启动（fail fast）。
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "").strip()
if not AUTH_SECRET_KEY or AUTH_SECRET_KEY in {"replace-with-a-random-secret", "rag_teaching_project_2026"}:
    raise RuntimeError("请在 .env 中设置随机的 AUTH_SECRET_KEY")


# ── LangSmith LLM 追踪（可选，默认关闭）─────────────────
# 放在最后导入：该模块要读 .env 的值才能判断是否开启追踪，因此必须晚于 load_dotenv。
# 它只写环境变量（供 langchain 自动上报用），不参与其它配置，放在文件末尾避免循环依赖。
import langsmith_tracing  # noqa: E402,F401
