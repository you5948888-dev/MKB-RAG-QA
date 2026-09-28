"""知识库管理路由（Ch2-Ch10）。Ch11 改造：从 app.py 拆分为独立模块。"""
import os
import logging

from fastapi import APIRouter, HTTPException, Depends
import database
import milvus_ops
import auth
from auth import ensure_kb_access as _ensure_kb_access
import rag_pipeline
import chunking
import strategy as strategy_module
from models import (
    KnowledgeBaseCreateRequest, KnowledgeBaseRenameRequest, RecallTestRequest,
    ChunkConfigUpdateRequest, ChunkPreviewRequest, ReindexRequest,
    RetrievalStrategyUpdateRequest,
)

logger = logging.getLogger("rag.routes.kb")

router = APIRouter(prefix="/api", tags=["kb"])


@router.get("/kb")
def list_knowledge_bases(
    keyword: str | None = None,
    page: int = 1,
    page_size: int = 10,
    user: dict = Depends(auth.get_current_user),
):
    """获取知识库列表。admin 看到所有 KB 并保留分页/搜索；普通用户只看到自己的（不分页）。
    保持原有响应结构 {items, total}，前端 fetchPage 直接读取 data.items/data.total。"""
    if page < 1 or page_size < 1:
        raise HTTPException(
            status_code=422,
            detail={"code": 42201, "message": "page 和 page_size 必须大于等于 1"},
        )
    if user["role"] == "admin":
        return database.list_kbs(keyword=keyword, page=page, page_size=page_size)
    # 普通用户：取出（支持 keyword 过滤的）全部后，再按归属过滤
    owned = set(auth.get_user_owned_kb_ids(user["user_id"]))
    all_items = database.list_kbs(keyword=keyword, page_size=None)["items"]
    items = [kb for kb in all_items if kb["kb_id"] in owned]
    return {"items": items, "total": len(items)}


@router.get("/kb/{kb_id}")
def get_knowledge_base(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """根据 ID 查询单个知识库，校验访问权限。"""
    _ensure_kb_access(kb_id, user)
    return {"code": 0, "message": "ok", "data": database.get_kb(kb_id)}


@router.post("/kb")
def create_knowledge_base(payload: KnowledgeBaseCreateRequest, user: dict = Depends(auth.get_current_user)):
    """创建知识库，自动绑定当前用户为 owner。

    Advanced RAG 第 1 期：可同时提交分块策略与完整配置。
    取值规则（避免歧义）：传了 `chunk_config` 就以它为准（未出现的项取默认值）；
    没传则用扁平字段 chunk_size / chunk_overlap 生成默认配置。
    """
    raw_config: dict = {}
    if payload.chunk_config is not None:
        raw_config.update(payload.chunk_config)
        if payload.chunk_strategy:
            raw_config.setdefault("strategy", payload.chunk_strategy)
        raw_config.setdefault("chunk_size", payload.chunk_size)
        raw_config.setdefault("chunk_overlap", payload.chunk_overlap)
    else:
        raw_config = {
            "chunk_size": payload.chunk_size,
            "chunk_overlap": payload.chunk_overlap,
        }
        if payload.chunk_strategy:
            raw_config["strategy"] = payload.chunk_strategy

    try:
        config = chunking.validate_config(raw_config)
    except chunking.ChunkConfigError as exc:
        raise HTTPException(status_code=400, detail={"code": 40003, "message": str(exc)})

    kb = database.create_kb(
        name=payload.name,
        description=payload.description or "",
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        chunk_strategy=config["strategy"],
        chunk_config=chunking.canonical_json(config),
    )
    if kb is None:
        raise HTTPException(
            status_code=409,
            detail={"code": 40901, "message": f"知识库名称 '{payload.name}' 已存在"},
        )
    auth.bind_kb_to_user(user["user_id"], kb["kb_id"])
    # 把 JSON 字符串换成结构化配置，前端可直接展示"当前使用的分块配置"
    return {"code": 0, "message": "ok", "data": {**kb, "chunk_config": config}}


@router.put("/kb/{kb_id}")
def rename_knowledge_base(kb_id: str, payload: KnowledgeBaseRenameRequest, user: dict = Depends(auth.get_current_user)):
    """修改知识库名称（前端"修改名称"功能）。改名后不与其它知识库重名。"""
    _ensure_kb_access(kb_id, user)
    existing = [kb for kb in database.list_kbs(page_size=None)["items"]
                if kb["name"] == payload.name and kb["kb_id"] != kb_id]
    if existing:
        raise HTTPException(
            status_code=409,
            detail={"code": 40901, "message": f"知识库名称 '{payload.name}' 已存在"},
        )
    kb = database.update_kb_name(kb_id, payload.name)
    return {"code": 0, "message": "ok", "data": kb}


# ── 分块配置与预览（Advanced RAG 第 1 期）──────────────

@router.get("/kb/{kb_id}/chunk-config")
def get_chunk_config(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """查看知识库当前使用的分块配置（同时返回可选值与默认值，便于前端渲染表单）。"""
    kb = _ensure_kb_access(kb_id, user)
    config = chunking.config_from_kb(kb)
    return {
        "code": 0, "message": "ok",
        "data": {
            "kb_id": kb_id,
            "strategy": config["strategy"],
            "config": config,
            "defaults": chunking.DEFAULT_CONFIG,
            "supported_strategies": list(chunking.STRATEGIES),
            "length_modes": list(chunking.LENGTH_MODES),
            "short_chunk_policies": list(chunking.SHORT_CHUNK_POLICIES),
            "parent_defaults": chunking.DEFAULT_PARENT_CONFIG,
            "child_defaults": chunking.DEFAULT_CHILD_CONFIG,
            "doc_count": len(database.list_documents(kb_id)),
            "config_hash": chunking.config_hash(config),
            "needs_reindex": len(database.list_docs_needing_reindex(kb_id, chunking.config_hash(config))),
        },
    }


@router.put("/kb/{kb_id}/chunk-config")
def update_chunk_config(kb_id: str, payload: ChunkConfigUpdateRequest,
                        user: dict = Depends(auth.get_current_user)):
    """修改知识库分块配置。

    需求 2.3 的硬约束：知识库已有文档时**不允许直接改配置**，否则新配置会与旧索引
    混用。此时返回 409 并附上旧配置、新配置与文档数量，切换动作由第 2 期的
    "重新索引"负责。无文档时直接生效。
    """
    kb = _ensure_kb_access(kb_id, user)
    try:
        config = chunking.validate_config(payload.chunk_config)
    except chunking.ChunkConfigError as exc:
        raise HTTPException(status_code=400, detail={"code": 40003, "message": str(exc)})

    old_config = chunking.config_from_kb(kb)
    docs = database.list_documents(kb_id)
    if docs and not payload.switch_and_reindex:
        # 需求 2.3：必须先让用户看到"已存在索引数据"与旧/新配置对比，再决定是否切换
        raise HTTPException(status_code=409, detail={
            "code": 40904,
            "message": (
                f"知识库已有 {len(docs)} 篇文档，为避免新旧索引混用，"
                "切换分块配置需要重新索引：请确认后带上 switch_and_reindex=true 重新提交，"
                "再调用 POST /api/kb/{kb_id}/reindex 重建索引"
            ),
            "old_config": old_config,
            "new_config": config,
            "doc_count": len(docs),
            "requires_reindex": True,
        })

    updated = database.update_kb_chunk_config(
        kb_id=kb_id,
        chunk_strategy=config["strategy"],
        chunk_config=chunking.canonical_json(config),
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    if updated is None:
        raise HTTPException(status_code=404, detail={"code": 40401, "message": "知识库不存在"})

    # 还没被 Worker 领取的任务直接换到新配置（避免排队的文档仍按旧配置入库）；
    # 已经按旧配置入库过的文档才需要重新索引。
    retargeted = 0
    if docs:
        retargeted = database.retarget_pending_upload_tasks(
            kb_id, chunking.canonical_json(config), config["chunk_size"], config["chunk_overlap"]
        )
    outdated = database.list_docs_needing_reindex(kb_id, chunking.config_hash(config))
    logger.info("更新知识库分块配置: %s（排队任务换配置 %d 个，待重新索引 %d 篇）",
                kb_id, retargeted, len(outdated))
    return {"code": 0, "message": "ok", "data": {
        **updated, "chunk_config": config,
        "old_config": old_config,
        "requires_reindex": bool(outdated),
        "docs_to_reindex": len(outdated),
        "docs_retargeted": retargeted,
    }}


@router.post("/kb/{kb_id}/chunk-preview")
def preview_chunk_config(kb_id: str, payload: ChunkPreviewRequest,
                         user: dict = Depends(auth.get_current_user)):
    """分块预览：展示分块数量、每块长度与具体内容。

    与入库完全共用 `chunking.chunk_text` / `chunk_document`，所以预览结果就是入库结果。
    可携带 `chunk_config` 做**临时覆盖**（不落库），用于调参对比。
    """
    kb = _ensure_kb_access(kb_id, user)
    base_config = chunking.config_from_kb(kb)

    if payload.chunk_config is None:
        config, overridden = base_config, False
    else:
        try:
            config = chunking.validate_config({**base_config, **payload.chunk_config})
        except chunking.ChunkConfigError as exc:
            raise HTTPException(status_code=400, detail={"code": 40003, "message": str(exc)})
        # 只有确实与知识库当前配置不同才算"临时覆盖"
        overridden = config != base_config

    filename = None
    if payload.text is not None:
        built = chunking.build_chunks(payload.text, config)
    elif payload.doc_id:
        doc = database.get_document(payload.doc_id)
        if doc is None or doc["kb_id"] != kb_id:
            raise HTTPException(status_code=404,
                                detail={"code": 40402, "message": "文档不存在或不属于该知识库"})
        if not os.path.exists(doc["path"]):
            raise HTTPException(status_code=404,
                                detail={"code": 40403, "message": "文档原始文件不存在，无法预览"})
        filename = doc["filename"]
        built = chunking.build_document_chunks(doc["path"], config)
    else:
        raise HTTPException(status_code=400,
                            detail={"code": 40004, "message": "请提供 text（文本预览）或 doc_id（文档预览）"})

    return {
        "code": 0, "message": "ok",
        "data": {
            "kb_id": kb_id,
            "doc_id": payload.doc_id,
            "filename": filename,
            "strategy": built["strategy"],
            "total_chunks": len(built["children"]),
            "total_parents": len(built["parents"]),
            "items": built["children"],                        # 可检索叶子块（入库对象）
            "parents": chunking.group_parents(built),           # 父子层级视图
            "config": config,
            "overridden": overridden,
            "config_hash": chunking.config_hash(config),
        },
    }


@router.post("/kb/{kb_id}/reindex")
def reindex_knowledge_base(kb_id: str, payload: ReindexRequest | None = None,
                           user: dict = Depends(auth.get_current_user)):
    """按当前分块配置重新索引文档（Advanced RAG 第 2 期）。

    - 不传 doc_ids：重建所有"配置过期"的文档（实际入库配置 ≠ 当前配置）
    - 传 doc_ids：重建指定文档（用于单篇重试）
    每篇文档创建一条 `task_type=reindex` 任务并投递到 Worker，进度复用上传任务接口查看。
    """
    kb = _ensure_kb_access(kb_id, user)
    config = chunking.config_from_kb(kb)
    config_hash = chunking.config_hash(config)

    outdated = {doc["doc_id"]: doc for doc in database.list_docs_needing_reindex(kb_id, config_hash)}
    if payload and payload.doc_ids:
        targets = []
        for doc_id in payload.doc_ids:
            doc = database.get_document(doc_id)
            if doc is None or doc["kb_id"] != kb_id:
                raise HTTPException(status_code=404,
                                    detail={"code": 40402, "message": f"文档不存在或不属于该知识库: {doc_id}"})
            targets.append(doc)
    else:
        targets = list(outdated.values())

    if not targets:
        raise HTTPException(status_code=400, detail={
            "code": 40005,
            "message": "没有需要重新索引的文档：所有文档都已按当前分块配置入库",
        })

    reindex_tasks = []
    skipped = []
    for doc in targets:
        if not os.path.exists(doc["path"]):
            skipped.append({"doc_id": doc["doc_id"], "filename": doc["filename"],
                            "reason": "原始文件不存在，无法重新索引"})
            continue
        try:
            task = database.create_upload_task(
                kb_id=kb_id, doc_id=doc["doc_id"], filename=doc["filename"],
                filepath=doc["path"], chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                chunk_config=chunking.canonical_json(config), task_type="reindex",
            )
        except ValueError as exc:
            skipped.append({"doc_id": doc["doc_id"], "filename": doc["filename"], "reason": str(exc)})
            continue
        try:
            from tasks import process_upload_task
            process_upload_task.delay(task["task_id"])
        except Exception as exc:
            database.mark_upload_dispatch_unknown(task["task_id"], str(exc))
            skipped.append({"doc_id": doc["doc_id"], "filename": doc["filename"],
                            "reason": f"投递结果待确认，请查询任务状态: {exc}"})
            continue
        reindex_tasks.append({"doc_id": doc["doc_id"], "filename": doc["filename"],
                              "task_id": task["task_id"], "status": "uploaded"})

    logger.info("重新索引: kb=%s 提交 %d 篇，跳过 %d 篇", kb_id, len(reindex_tasks), len(skipped))
    return {"code": 0, "message": "ok", "data": {
        "kb_id": kb_id, "strategy": config["strategy"], "config_hash": config_hash,
        "total": len(targets), "submitted": len(reindex_tasks),
        "reindex_tasks": reindex_tasks, "skipped": skipped,
    }}


@router.get("/kb/{kb_id}/index-status")
def get_index_status(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """索引状态总览：每篇文档实际入库的配置指纹、是否过期、最近任务与进度（第 2 期）。"""
    kb = _ensure_kb_access(kb_id, user)
    config = chunking.config_from_kb(kb)
    current_hash = chunking.config_hash(config)

    items = []
    outdated_count = 0
    for doc in database.list_documents(kb_id):
        doc_hash = doc.get("chunk_config_hash") or ""
        outdated = bool(doc_hash) and doc_hash != current_hash
        if outdated:
            outdated_count += 1
        tasks = database.get_tasks_by_doc_id(doc["doc_id"])
        latest = tasks[0] if tasks else None
        items.append({
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "chunk_count": doc.get("chunk_count", 0),
            "parent_count": doc.get("parent_count", 0),
            "child_count": doc.get("child_count", 0),
            "doc_config_hash": doc_hash,
            "config_outdated": outdated,
            "task": None if latest is None else {
                "task_id": latest["task_id"],
                "task_type": latest.get("task_type", "upload"),
                "status": latest["status"],
                "progress": latest["progress"],
                "step_message": latest["step_message"],
                "current_stage": latest.get("current_stage", ""),
                "parent_count": latest.get("parent_count", 0),
                "child_count": latest.get("child_count", 0),
                "embedded_count": latest.get("embedded_count", 0),
                "retry_count": latest.get("retry_count", 0),
                "error_message": latest.get("error_message", ""),
            },
        })

    return {"code": 0, "message": "ok", "data": {
        "kb_id": kb_id,
        "strategy": config["strategy"],
        "config_hash": current_hash,
        "doc_count": len(items),
        "outdated_count": outdated_count,
        "items": items,
    }}


@router.delete("/kb/{kb_id}")
def delete_knowledge_base(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """删除知识库。
    Ch11：非终态上传/索引构建任务由 database.delete_kb 检测并抛 ValueError → 409，
    路由层不再重复检查，改为 try/except 统一拦截。"""
    _ensure_kb_access(kb_id, user)
    try:
        deleted = database.delete_kb(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": 40401, "message": "知识库不存在"})
    # 清理 Milvus 中的向量（chunk + 索引问题）；失败仅告警，不阻断删除
    try:
        milvus_ops.delete_vectors_by_kb(kb_id)
    except Exception as exc:
        logger.warning("删除 Milvus 向量失败: %s", exc)
    try:
        milvus_ops.delete_index_question_vectors_for_kb(kb_id)  # Ch8 新增
    except Exception as exc:
        logger.warning("删除索引问题向量失败: %s", exc)
    logger.info("删除知识库: %s", kb_id)
    return {"code": 0, "message": "ok", "data": {"kb_id": kb_id, "deleted": True}}


@router.post("/kb/{kb_id}/recall-test")
def recall_test(kb_id: str, payload: RecallTestRequest, user: dict = Depends(auth.get_current_user)):
    """召回测试：检索 → 多路融合 → 可选 Rerank 精排。

    - 支持三种检索模式：dense_only / sparse_only / hybrid；
    - Advanced RAG 第 3 期：可临时覆盖模式与高级参数（改写/多查询/HyDE/融合算法…），
      临时覆盖只影响本次请求，不写库、不改历史记录。
    """
    _ensure_kb_access(kb_id, user)
    mode = payload.retrieval_mode.strip().lower()
    if mode not in {"dense_only", "sparse_only", "hybrid"}:
        raise HTTPException(status_code=400, detail="retrieval_mode 仅支持 dense_only / sparse_only / hybrid")

    overrides = {
        "mode": payload.mode,
        "fusion_method": payload.fusion_method,
        "enable_rewrite": payload.enable_rewrite,
        "enable_multi_query": payload.enable_multi_query,
        "multi_query_count": payload.multi_query_count,
        "enable_hyde": payload.enable_hyde,
        "rerank_top_n": payload.rerank_top_n,
        "max_context_chars": payload.max_context_chars,
    }
    # 召回测试没有会话概念，但问题改写需要对话历史：允许调用方直接传"上一轮问题"
    history = [{"role": "user", "content": text.strip()}
               for text in (payload.history or []) if text and text.strip()]
    try:
        result = rag_pipeline.recall_test(
            kb_id=kb_id, query=payload.query, retrieval_mode=mode,
            top_k=payload.top_k, candidate_k=payload.candidate_k,
            dense_weight=payload.dense_weight, sparse_weight=payload.sparse_weight,
            min_score=payload.min_score or None,   # 0 表示"未指定"，不覆盖策略里的阈值
            enable_rerank=payload.enable_rerank,
            enable_index=payload.enable_index, debug=payload.debug,
            history=history or None,
            strategy=overrides,
        )
    except strategy_module.StrategyConfigError as exc:
        raise HTTPException(status_code=400, detail={"code": 40006, "message": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"召回测试失败: {exc}")
    return {"code": 0, "message": "ok", "data": {"kb_id": kb_id, "query": payload.query, **result}}


# ── 检索策略管理（Advanced RAG 第 3 期）──────────────

@router.get("/kb/{kb_id}/retrieval-strategy")
def get_retrieval_strategy(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """查看知识库当前生效的检索策略（含默认值与是否自定义）。"""
    _ensure_kb_access(kb_id, user)
    raw = database.get_retrieval_strategy(kb_id)
    effective = rag_pipeline.resolve_strategy(kb_id)
    return {"code": 0, "message": "ok", "data": {
        "kb_id": kb_id,
        "customized": bool(raw),
        "strategy": effective,
        "defaults": strategy_module.DEFAULT_STRATEGY,
        "supported": {
            "modes": list(strategy_module.RETRIEVAL_MODES),
            "fusion_methods": list(strategy_module.FUSION_METHODS),
        },
        "strategy_hash": strategy_module.strategy_hash(effective),
    }}


@router.put("/kb/{kb_id}/retrieval-strategy")
def update_retrieval_strategy(kb_id: str, payload: RetrievalStrategyUpdateRequest,
                              user: dict = Depends(auth.get_current_user)):
    """修改检索策略（保存后只影响后续请求，不改动已完成的问答记录）。"""
    _ensure_kb_access(kb_id, user)
    try:
        config = strategy_module.validate_strategy(payload.strategy)
        # 生产级：保存时严格校验，禁止"开了高级能力却停在基础模式"的矛盾配置
        config = strategy_module.ensure_mode_consistency(config)
    except strategy_module.StrategyConfigError as exc:
        raise HTTPException(status_code=400, detail={"code": 40006, "message": str(exc)})

    database.upsert_retrieval_strategy(kb_id, strategy_module.canonical_json(config))
    logger.info("更新知识库检索策略: %s（mode=%s, fusion=%s）",
                kb_id, config["mode"], config["fusion_method"])
    return {"code": 0, "message": "ok", "data": {
        "kb_id": kb_id, "customized": True, "strategy": config,
        "strategy_hash": strategy_module.strategy_hash(config),
    }}


@router.post("/kb/{kb_id}/retrieval-strategy/reset")
def reset_retrieval_strategy(kb_id: str, user: dict = Depends(auth.get_current_user)):
    """恢复默认检索策略（删除自定义配置，回落到全局默认）。"""
    _ensure_kb_access(kb_id, user)
    removed = database.delete_retrieval_strategy(kb_id)
    effective = rag_pipeline.resolve_strategy(kb_id)
    logger.info("恢复默认检索策略: %s（移除自定义=%s）", kb_id, removed)
    return {"code": 0, "message": "ok", "data": {
        "kb_id": kb_id, "customized": False, "strategy": effective,
        "strategy_hash": strategy_module.strategy_hash(effective),
    }}
