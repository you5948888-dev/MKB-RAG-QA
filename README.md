# RAG 知识库问答系统

> RAG knowledge base QA with hybrid retrieval, Advanced RAG and full-link observability.

面向私有知识库的检索增强问答系统：文档解析 → 分块 → 向量化入库 → 混合检索 → Rerank → 上下文组装 → 流式生成与引用溯源。

技术栈：RAG（混合检索 + Rerank + 引用溯源）· Advanced RAG（改写 / 多查询 / HyDE / 多路融合）· Qwen + `langchain-openai` · FastAPI · Milvus · Celery + Redis · SQLite · Docker

> **运行前提**：完整跑通需要自备 Milvus、Redis、BGE（向量化 + Rerank）与 OpenAI 兼容的 LLM 服务。
> **没有这些也能验证**：`pytest tests -q` —— 201 项离线集成测试，不需要任何外部服务。

<!-- 录制演示视频后，把截图放到 docs/images/ 并取消注释
![知识库管理](docs/images/kb.png)
![对话与执行详情](docs/images/chat.png)
![运行观测面板](docs/images/observability.png)
-->

## 功能

- **混合检索**：dense + sparse 加权混合，BGE Rerank 精排，权重与候选规模可配
- **Advanced RAG**：问题改写 / 多查询 / HyDE + 两套多路融合（RRF、min-max 加权），默认关闭、按需开启
- **两类分块**：通用分块 + 父子分块（子块负责检索、父块提供完整上下文），预览与入库结果一致，支持重新索引
- **三种流式输出**：SSE / WebSocket / NDJSON 共用一套事件协议（`citations → token → done | error`）
- **异步任务**：Celery + Redis 任务状态机，含进度、失败回滚、僵尸任务回收、删除保护
- **全链路可观测**：trace/span、回答旁的「执行详情」（改写结果、各路径命中数、融合与 Rerank 变化、首 Token 耗时）、观测面板、健康检查
- **降级可见**：改写 / 多查询 / HyDE / 索引问题检索 / Rerank / 流式生成失败均可降级，且每次降级都记录原因与兜底

## 快速开始

需要 Python 3.12，以及可访问的 Milvus、Redis、BGE 与 LLM 服务。

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置（模板里都是占位符，必须改成自己的地址与密钥）
cp .env.example .env
#    必填：MILVUS_HOST / MILVUS_PORT、BGE_URL、LLM_API_URL / LLM_MODEL_NAME / LLM_API_KEY、
#          CELERY_BROKER_URL、AUTH_SECRET_KEY（随机字符串）

# 3. 启动 Web 与 Worker（两个进程都在项目根目录）
uvicorn app:app --port 8013
celery -A tasks.celery_app worker -Q upload_queue -l info -c 3    # Windows 本机可加 -P gevent
```

| 页面 | 地址 |
|---|---|
| 知识库管理（建库 / 上传 / 分块配置 / 重新索引 / 召回测试） | http://127.0.0.1:8013/ui/kb.html |
| 对话 + 执行详情 | http://127.0.0.1:8013/ui/index.html |
| 运行观测面板 | http://127.0.0.1:8013/ui/observability.html |
| 接口文档 | http://127.0.0.1:8013/docs |

首次启动会自动建表、初始化 Milvus 集合，并创建管理员 `admin`（**初始密码见启动日志**，请登录后立即修改）。

## 测试

```bash
pytest tests -q     # 201 passed，不需要 Milvus / Redis / 模型服务
```

## 容器化部署

```bash
docker build -t rag-tutorial:0.14.0 .
cp .env.docker.example .env.docker                  # 填真实地址与密钥（该文件已被 gitignore）
docker compose --env-file .env.compose up -d        # web / worker / redis 三容器
```

镜像为多阶段构建、以非 root（UID 10001）运行，依赖固定在 `requirements.runtime.txt`，不含测试工具。

## 目录结构

```
app.py config.py auth.py middleware.py models.py database.py milvus_ops.py embedding.py
text_parser.py chunking.py strategy.py advanced_rag.py fusion.py rag_pipeline.py rag_stream.py
tasks.py observability.py langsmith_tracing.py
routes/     9 个业务路由模块（知识库 / 文档 / 问答 / 流式 / 会话 / 任务 / 索引问题 / 观测 / 健康检查）
frontend/   kb.html / index.html / observability.html / http.js（原生 JS，无构建步骤）
tests/      201 项离线集成测试（临时数据库 + 依赖替身 + 网络阻断）
```

## 说明

- Token 长度口径使用 `tiktoken cl100k` 近似（用于控制尺寸，不等于计费 token）；父块不参与向量检索，靠子块命中后回查
- 观测数据不自动清理，提供手动清理接口（默认 dry-run，仅管理员）
- 可选 LangSmith 追踪（默认关闭）：设置 `LANGSMITH_ENABLED=true` 与 `LANGSMITH_API_KEY` 后，LLM 的 prompt 与检索上下文会上传到 LangSmith 云端
- `.env`、`data/`（数据库与上传文件）已加入 gitignore，请勿提交；内置管理员仅供本地体验

## English

A RAG knowledge base QA system for private document collections: parsing → chunking → embedding → hybrid retrieval → rerank → context assembly → streaming generation with citations. It also provides Advanced RAG abilities (query rewriting, multi-query, HyDE, multi-path fusion), parent-child chunking with re-indexing, and full-link observability (tracing, execution details, dashboard, health checks).

Built with FastAPI, Milvus, BGE-M3, Qwen, Celery + Redis and SQLite. The external services (Milvus / Redis / BGE / LLM) are required to run the app, while `pytest tests -q` verifies 201 offline integration tests without any of them.
