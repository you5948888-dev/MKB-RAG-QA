"""
RAG 检索管道：混合检索 + 分数融合 + Rerank 精排。

核心流程：
  1. 将用户查询向量化（dense + sparse）
  2. 在 Milvus 中进行三路检索（dense_only / sparse_only / hybrid）
  3. 保留原始分数观察尺度
  4. 加权融合（hybrid 模式）
  5. 可选 Rerank 精排
  6. 返回排序后的结果
"""
import json
import time
import logging

import database  # Ch8 新增
import embedding
import milvus_ops
import advanced_rag
import fusion as fusion_module
import observability
import langsmith_tracing
import strategy as strategy_module
from config import PIPELINE_CONFIG_PATH

logger = logging.getLogger("rag.pipeline")

# 组装上下文时各块之间的分隔符（其长度也要计入上下文预算）
_CONTEXT_SEPARATOR = "\n\n---\n\n"


def _truncate_text(text, limit: int = 120) -> str:
    """观测详情里的短文本（避免把整段文档写进追踪库）。"""
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


def _summarize_items(items: list[dict], limit: int = 8) -> list[dict]:
    """把最终上下文条目压缩成摘要，用于"执行详情"展示候选与顺序变化。"""
    summary = []
    for item in items[:limit]:
        summary.append({
            "chunk_id": item.get("chunk_id") or item.get("child_chunk_id"),
            "source": item.get("source"),
            "chunk_index": item.get("chunk_index"),
            "parent_index": item.get("parent_index"),
            "sense_index": item.get("sense_index"),
            "final_score": item.get("final_score"),
            "rerank_score": item.get("rerank_score"),
            "hit_paths": item.get("hit_paths"),
            "content_preview": _truncate_text(item.get("child_content") or item.get("content")),
        })
    return summary


def _load_pipeline_config() -> dict:
    """加载 pipeline 配置文件，不存在则返回默认值。"""
    try:
        with open(PIPELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "enable_rerank": True,
            "dense_weight": 0.5,
            "sparse_weight": 0.5,
            "retrieval_top_k": 5,
            "candidate_k": 20,
        }


# ── 分数归一化（Ch6 新增）──────────────────────────────

def _normalize_dense_score(raw_distance: float) -> float:
    """COSINE 相似度截断到 [0,1]，高分更相似。"""
    return max(0.0, min(1.0, float(raw_distance)))


def _normalize_sparse_score(raw_ip: float) -> float:
    """非负 IP 用 x/(1+x) 归一化到 [0,1)，保留相对大小。稀疏命中、dense 未掉时为 0。"""
    raw_ip = max(0.0, float(raw_ip))
    return raw_ip / (1.0 + raw_ip)


# ── 核心检索函数 ──────────────────────────────────────────

def recall_test(
    kb_id: str,
    query: str,
    retrieval_mode: str = "hybrid",
    top_k: int | None = None,
    candidate_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    min_score: float | None = None,
    enable_rerank: bool | None = None,
    enable_index: bool | None = None,  # Ch8 新增
    debug: bool = False,
    # Advanced RAG 第 3 期
    mode: str | None = None,
    strategy: dict | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    执行召回：检索 → 多路融合 → 索引问题融合 → 可选 Rerank → 父子块展开。

    basic 模式（默认）与 Ch8 起的行为完全一致；advanced 模式按知识库检索策略
    组合启用 问题改写 / 多查询 / HyDE / 多路融合（见 fusion.py 与 advanced_rag.py）。

    参数:
        kb_id: 知识库 ID
        query: 用户查询文本
        retrieval_mode: "dense_only" / "sparse_only" / "hybrid"
        top_k: 最终返回结果数（覆盖策略里的 rerank_top_n）
        candidate_k: Milvus 检索候选集大小
        dense_weight / sparse_weight: 路径内 dense/sparse 融合权重
        min_score: 最低分数阈值
        enable_rerank: 是否启用 Rerank 精排
        enable_index: 是否启用索引问题召回融合
        debug: 是否返回调试信息
        mode: 强制指定 basic / advanced（覆盖知识库策略）
        strategy: 请求级策略覆盖（仅本次生效）
        history: 最近对话历史（问题改写需要）

    返回:
        {"items": [...], "total": int, "avg_similarity": float, "debug": dict}
    """
    effective = resolve_strategy(kb_id, override=strategy, mode=mode, explicit={
        "rerank_top_n": top_k,
        "candidate_k": candidate_k,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "enable_rerank": enable_rerank,
        "enable_index": enable_index,
        "min_score": min_score,
    })

    if effective["mode"] == "advanced":
        return _recall_advanced(kb_id=kb_id, query=query, retrieval_mode=retrieval_mode,
                                config=effective, debug=debug, history=history)
    return _recall_basic(kb_id=kb_id, query=query, retrieval_mode=retrieval_mode,
                         config=effective, debug=debug)


def resolve_strategy(kb_id: str, *, override: dict | None = None, mode: str | None = None,
                     explicit: dict | None = None) -> dict:
    """解析生效检索策略：请求临时覆盖 > 知识库策略 > 全局 pipeline_config > 内置默认。

    explicit 里为 None 的键表示"调用方未显式指定"，不参与覆盖——这样
    Ch8 时代的默认行为（未传参就用全局配置）完全保留。
    """
    kb_strategy = None
    raw = database.get_retrieval_strategy(kb_id)
    if raw:
        try:
            kb_strategy = json.loads(raw)
        except json.JSONDecodeError:
            kb_strategy = None
    merged_override: dict = {}
    if override:
        merged_override.update({k: v for k, v in override.items() if v is not None})
    if mode:
        merged_override["mode"] = mode
    if explicit:
        merged_override.update({k: v for k, v in explicit.items() if v is not None})
    return strategy_module.resolve(kb_strategy, merged_override,
                                   global_defaults=_load_pipeline_config())


def _recall_basic(kb_id: str, query: str, retrieval_mode: str, config: dict, debug: bool) -> dict:
    """基础模式检索（Ch8 起的原有流程，参数已由策略解析完毕）。"""
    enable_rerank = config["enable_rerank"]
    enable_index = config["enable_index"]
    sparse_weight = config["sparse_weight"]
    dense_weight = config["dense_weight"]
    candidate_k = config["candidate_k"]
    top_k = config["rerank_top_n"]
    min_score = config["min_score"]
    start = time.time()
    # 本模式内的降级记录（索引问题检索失败 / Rerank 失败），随返回值与 trace 一起暴露，
    # 避免"降级了但执行详情里看不到"（需求：每一次降级都应在执行详情中可见）。
    degradations: list[dict] = []

    # 根据检索模式调整权重
    if retrieval_mode == "dense_only":
        dense_weight, sparse_weight = 1.0, 0.0
    elif retrieval_mode == "sparse_only":
        dense_weight, sparse_weight = 0.0, 1.0

    # 1. 将查询文本向量化
    with observability.span(observability.STAGE_EMBED, input_count=1, query=query) as span:
        emb = embedding.bge_model.embed(query)
        dense_vec = embedding.get_dense_vector(emb)
        sparse_vec = embedding.get_sparse_vector(emb)
        span.set_output(2, parts="dense+sparse")

    # 2. Milvus 检索
    with observability.span(observability.STAGE_RETRIEVE, input_count=1, mode=retrieval_mode) as span:
        hits = milvus_ops.search_vectors(
            kb_id=kb_id,
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            retrieval_mode=retrieval_mode,
            top_k=candidate_k,
        )
        span.set_output(len(hits), path="original")

    # 3. 普通检索：归一化 + 加权融合，存入 chunk_map（以 chunk_id 去重）
    chunk_map: dict[str, dict] = {}
    for hit in hits:
        d_norm = _normalize_dense_score(hit["dense_score"])
        s_norm = _normalize_sparse_score(hit["sparse_score"])

        if retrieval_mode == "dense_only":
            fused = d_norm
        elif retrieval_mode == "sparse_only":
            fused = s_norm
        else:
            fused = dense_weight * d_norm + sparse_weight * s_norm

        chunk_map[hit["pk"]] = {
            "pk": hit["pk"],
            "chunk_id": hit["pk"],
            "source": hit["source"],
            "chunk_index": hit["chunk_index"],
            "content": hit["text"],
            "dense_score": hit["dense_score"],
            "sparse_score": hit["sparse_score"],
            "dense_sim": d_norm,
            "sparse_sim": s_norm,
            "fused_score": fused,
            "hit_source": "chunk",
        }

    # 4. 索引问题检索（可选）：搜索匹配问题 → 获取绑定 chunk → 合并到 chunk_map
    index_hit_count = 0
    index_boost_count = 0
    if enable_index:
        try:
            index_hits = milvus_ops.search_index_questions(kb_id, dense_vec, top_k=candidate_k)
            # 已是 COSINE 相似度，直接过滤低分
            index_hits = [h for h in index_hits if h["score"] >= min_score]
            index_hit_count = len(index_hits)

            if index_hits:
                index_ids = [h["pk"] for h in index_hits]
                links = database.get_index_links_batch(index_ids)
                score_map = {h["pk"]: h["score"] for h in index_hits}

                for link in links:
                    if link["kb_id"] != kb_id:
                        continue
                    if not link.get("chunk_text"):
                        continue
                    chunk_id = link["chunk_id"]
                    index_sim = score_map.get(link["index_id"], 0.0)

                    if chunk_id not in chunk_map:
                        # 普通检索未命中的 chunk，由索引问题补充
                        index_boost_count += 1
                        chunk_map[chunk_id] = {
                            "pk": chunk_id,
                            "chunk_id": chunk_id,
                            "source": link.get("source", ""),
                            "chunk_index": link.get("chunk_index", 0),
                            "content": link.get("chunk_text", ""),
                            "dense_score": None,
                            "sparse_score": None,
                            "dense_sim": None,
                            "sparse_sim": None,
                            "fused_score": index_sim,
                            "hit_source": "index_question",
                            # Ch8 课后题:记录「索引问题命中」证据，供重排后冲突裁决使用
                            "_index_hit": True,
                            "_index_sim": index_sim,
                        }
                    else:
                        # 冲突:同一 chunk 已被普通召回又经索引问题命中。
                        # 在这里先只记录索引证据，不立即改 source/分数——
                        # 谁更可信、以谁排序，统一交给「重排后冲突裁决」（见第5步），
                        # 避免用重排前的融合旧分做武断的提前替换。
                        chunk_map[chunk_id]["_index_hit"] = True
                        chunk_map[chunk_id]["_index_sim"] = max(
                            chunk_map[chunk_id].get("_index_sim", 0.0), index_sim
                        )
        except Exception as exc:
            # 需求六：索引问题检索失败 → 继续使用普通检索；且必须作为一次降级可见
            logger.warning("索引问题检索失败，使用普通召回: %s", exc)
            reason = f"索引问题检索失败: {type(exc).__name__}: {exc}"
            degradations.append({
                "stage": observability.STAGE_RETRIEVE,
                "reason": reason,
                "fallback": "继续使用普通召回（仅索引问题路径失败）",
            })
            observability.degrade(observability.STAGE_RETRIEVE, reason,
                                  "继续使用普通召回（仅索引问题路径失败）")

    # 观测：多路召回 + 索引问题融合的中间结果
    observability.mark(observability.STAGE_FUSE,
                       input_count=len(hits) + index_hit_count,
                       output_count=len(chunk_map),
                       index_question_hits=index_hit_count,
                       index_boosted=index_boost_count,
                       method="weighted_normalize")

    # 5. Rerank 精排
    nodes = list(chunk_map.values())
    rerank_applied = False
    if enable_rerank and nodes:
        with observability.span(observability.STAGE_RERANK, input_count=len(nodes)) as span:
            nodes, rerank_applied = _rerank_nodes(nodes, query)
            span.set_output(len(nodes), applied=rerank_applied)
            if not rerank_applied:
                # Rerank 失败不算致命：退化为融合分数排序（需求允许的降级）
                span.status = observability.STATUS_DEGRADED
                span.detail["fallback"] = "使用融合分数排序"
                reason = "Rerank 未生效（服务不可用或返回不完整）"
                degradations.append({"stage": observability.STAGE_RERANK, "reason": reason,
                                     "fallback": "使用融合分数排序"})
                observability.note_degradation(observability.STAGE_RERANK, reason,
                                               "使用融合分数排序")

    # 5.1 冲突裁决（Ch8 课后题 3）：同一 chunk 被普通召回 + 索引问题命中时，
    # 只保留一条，决定「更可信来源」并统一用重排分排序。
    #
    # 策略与理由：
    #   索引问题是针对 chunk 编写的高质量「问题锚点」，命中它代表该 chunk 与
    #   用户语义强对齐，因此本身是比普通向量召回更可信的**来源证据**。
    #   但为了不武断，冲突时以**重排语义相关性**作最终裁决：
    #   - 若该冲突节点的 rerank_score 达到池内均值以上 -> 说明内容确实是答案，
    #     索引命中证据被印证，标记为更可信的 index_question；
    #   - 否则说明普通向量召回已足够可信，标记 chunk。
    #   纯索引补充、普通召回根本没命中的节点天然只有索引一个来源，恒标记 index_question。
    #   这样最终排序以 rerank_score 为准，hit_source 也体现"更可信来源"。
    conflict_threshold = None
    if rerank_applied and nodes:
        _scores = [n["rerank_score"] for n in nodes]
        conflict_threshold = sum(_scores) / len(_scores)
    for node in nodes:
        node["final_score"] = (
            node["rerank_score"] if rerank_applied else node["fused_score"]
        )
        if node.get("_index_hit"):
            if node["dense_sim"] is None:
                # 纯索引补充来源，无冲突，恒标 index_question
                node["hit_source"] = "index_question"
            elif rerank_applied:
                # 冲突节点：重排分 ≥ 均值则索引证据更可信
                node["hit_source"] = (
                    "index_question" if node["rerank_score"] >= conflict_threshold else "chunk"
                )
            else:
                # 未重排时退化：索引证据强于融合分才标 index_question
                node["hit_source"] = (
                    "index_question" if node["_index_sim"] >= node["fused_score"] else "chunk"
                )
    nodes.sort(key=lambda x: x["final_score"], reverse=True)

    items = []
    for idx, node in enumerate(nodes[:top_k], start=1):
        final_score = node["final_score"]
        if final_score < min_score:
            continue
        items.append({
            "rank": idx,
            "chunk_id": node.get("chunk_id", ""),   # 父子分块回查父块的关键字段
            "source": node["source"],
            "chunk_index": node["chunk_index"],
            "content": node["content"],
            "dense_score": round(node["dense_score"], 6) if node["dense_score"] is not None else None,
            "sparse_score": round(node["sparse_score"], 6) if node["sparse_score"] is not None else None,
            "dense_sim": round(node["dense_sim"], 4) if node["dense_sim"] is not None else None,
            "sparse_sim": round(node["sparse_sim"], 4) if node["sparse_sim"] is not None else None,
            "fused_score": round(node["fused_score"], 4),
            "final_score": round(final_score, 4),
            "rerank_score": round(node.get("rerank_score", 0), 4)
                if rerank_applied else None,
            "hit_source": node.get("hit_source", "chunk"),  # Ch8
        })

    # 5.2 父子分块：命中子块 → 回查父块完整上下文 → 同一父块去重（第 2 期）
    items = _expand_parent_context(items)

    elapsed_ms = round((time.time() - start) * 1000, 2)
    sim_values = [item["final_score"] for item in items]
    avg_similarity = round(sum(sim_values) / len(sim_values), 4) if sim_values else 0.0

    # 观测：把本次问答的关键中间结果写进执行详情
    recorder = observability.current()
    if recorder is not None:
        recorder.set_result(
            config_snapshot=config,
            recall_empty=(len(items) == 0),
            mode="basic",
            after_merge=len(chunk_map),
            after_filter=len(items),
            rerank_applied=rerank_applied,
            retrieval_hits=len(hits),
            top_items=_summarize_items(items),
        )

    debug_info = {}
    if debug:
        debug_info = {
            "pipeline": {
                "mode": "basic",
                "retrieval_mode": retrieval_mode,
                "enable_rerank": enable_rerank,
                "enable_index": enable_index,
                "rerank_applied": rerank_applied,
            },
            "config_snapshot": {
                "top_k": top_k,
                "candidate_k": candidate_k,
                "min_score": min_score,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "max_context_chars": config.get("max_context_chars"),
            },
            "retrieval": {
                "milvus_chunk_hits": len(hits),
                "index_question_hits": index_hit_count,
                "index_boosted": index_boost_count,
                "after_merge": len(chunk_map),
                "after_filter": len(items),
            },
            "latency_ms": elapsed_ms,
        }

    return {
        "items": items,
        "total": len(items),
        "avg_similarity": avg_similarity,
        "debug": debug_info,
        "degradations": degradations,   # 基础模式的降级也回传（此前恒为空，执行详情看不到）
        "config_snapshot": {   # 本次生效的配置（供响应/观测/执行详情展示）
            "mode": "basic",
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "min_score": min_score,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "enable_rerank": enable_rerank,
            "enable_index": enable_index,
            "max_context_chars": config.get("max_context_chars"),
            # 以下字段基础模式不使用，但属于"生效策略"的一部分，快照里一并回显便于排查
            "fusion_method": config.get("fusion_method"),
            "per_path_k": config.get("per_path_k"),
            "total_candidate_cap": config.get("total_candidate_cap"),
        },
    }


def _search_path_hits(kb_id: str, query_text: str, *, retrieval_mode: str, config: dict) -> list[dict]:
    """单条查询的一次向量检索 → 融合输入。

    路径内的 dense/sparse 仍用原有加权归一化（COSINE 截断 + x/(1+x)），
    因此"路径内分数可比"，路径之间再交给 RRF / min-max 处理。
    """
    emb = embedding.bge_model.embed(query_text)
    dense_vec = embedding.get_dense_vector(emb)
    sparse_vec = embedding.get_sparse_vector(emb)
    hits = milvus_ops.search_vectors(
        kb_id=kb_id, dense_vec=dense_vec, sparse_vec=sparse_vec,
        retrieval_mode=retrieval_mode, top_k=config["candidate_k"],
    )
    dense_weight, sparse_weight = config["dense_weight"], config["sparse_weight"]
    if retrieval_mode == "dense_only":
        dense_weight, sparse_weight = 1.0, 0.0
    elif retrieval_mode == "sparse_only":
        dense_weight, sparse_weight = 0.0, 1.0

    results = []
    for hit in hits:
        d_norm = _normalize_dense_score(hit["dense_score"])
        s_norm = _normalize_sparse_score(hit["sparse_score"])
        if retrieval_mode == "dense_only":
            score = d_norm
        elif retrieval_mode == "sparse_only":
            score = s_norm
        else:
            score = dense_weight * d_norm + sparse_weight * s_norm
        results.append({
            "chunk_id": hit["pk"],
            "source": hit["source"],
            "chunk_index": hit["chunk_index"],
            "content": hit["text"],
            "score": score,
            "dense_score": hit["dense_score"],
            "sparse_score": hit["sparse_score"],
            "dense_sim": d_norm,
            "sparse_sim": s_norm,
        })
    return results


def _index_question_hits(kb_id: str, query_text: str, config: dict) -> list[dict]:
    """索引问题检索路径：命中索引问题 → 取绑定 chunk 作为该路径候选。"""
    emb = embedding.bge_model.embed(query_text)
    dense_vec = embedding.get_dense_vector(emb)
    matched = milvus_ops.search_index_questions(kb_id, dense_vec, top_k=config["candidate_k"])
    matched = [item for item in matched if item["score"] >= config["min_score"]]
    if not matched:
        return []
    score_map = {item["pk"]: item["score"] for item in matched}
    links = database.get_index_links_batch([item["pk"] for item in matched])
    hits = []
    for link in links:
        if link.get("kb_id") != kb_id or not link.get("chunk_text"):
            continue
        hits.append({
            "chunk_id": link["chunk_id"],
            "source": link.get("source", ""),
            "chunk_index": link.get("chunk_index", -1),
            "content": link["chunk_text"],
            "score": score_map.get(link["index_id"], 0.0),
        })
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits


def _recall_advanced(kb_id: str, query: str, retrieval_mode: str, config: dict,
                     debug: bool, history: list[dict] | None = None) -> dict:
    """高级模式：问题改写 / 多查询 / HyDE → 多路召回 → 融合 → Rerank → 父子块展开。

    降级约定：任一前置能力或某条检索路径失败，只把该路径记为 0 命中，
    不影响其它路径；原问题始终参与检索。
    """
    start = time.time()
    plan = advanced_rag.build_plan(query, history=history, config=config)
    # 检索路径级降级（索引问题检索失败 / Rerank 失败），会合并进返回的 degradations
    path_degradations: list[dict] = []

    path_hits: dict[str, list[dict]] = {}
    for path_id, query_text in plan.search_queries():
        try:
            with observability.span(observability.STAGE_RETRIEVE, input_count=1,
                                    path=path_id, query=_truncate_text(query_text)) as span:
                path_hits[path_id] = _search_path_hits(kb_id, query_text,
                                                       retrieval_mode=retrieval_mode, config=config)
                span.set_output(len(path_hits[path_id]), path=path_id)
        except Exception as exc:
            logger.warning("检索路径 %s 失败，已跳过: %s", path_id, exc)
            path_hits[path_id] = []
    if config["enable_index"]:
        # 注意：捕获点放在 span **内部**——异常若穿出 `with`，该阶段会被记成 failed（语义是"这一步挂了"），
        # 而这里实际是"这一路降级、整体继续"，所以要在 span 内接住并显式标成 degraded。
        with observability.span(observability.STAGE_RETRIEVE, input_count=1,
                                path="index_question") as span:
            try:
                path_hits["index_question"] = _index_question_hits(kb_id, plan.original, config)
                span.set_output(len(path_hits["index_question"]), path="index_question")
            except Exception as exc:
                logger.warning("索引问题检索失败，继续使用普通检索: %s", exc)
                path_hits["index_question"] = []
                reason = f"索引问题检索失败: {type(exc).__name__}: {exc}"
                fallback = "继续使用普通召回（仅索引问题路径失败）"
                path_degradations.append({"stage": observability.STAGE_RETRIEVE,
                                          "reason": reason, "fallback": fallback})
                span.status = observability.STATUS_DEGRADED
                span.set_output(0, path="index_question")
                span.detail["error"] = reason
                span.detail["fallback"] = fallback
                observability.note_degradation(observability.STAGE_RETRIEVE, reason, fallback)

    # 融合：去重 + 统一排序 + 每路配额 + 总量上限（进入 Rerank 前控制规模）
    nodes, fusion_debug = fusion_module.fuse(
        path_hits, method=config["fusion_method"],
        per_path_k=config["per_path_k"], total_cap=config["total_candidate_cap"],
    )
    observability.mark(observability.STAGE_FUSE,
                       input_count=sum(len(hits) for hits in path_hits.values()),
                       output_count=len(nodes),
                       method=config["fusion_method"],
                       path_hits={path: len(hits) for path, hits in path_hits.items()},
                       dedup_removed=fusion_debug.get("dedup_removed"))

    rerank_applied = False
    if config["enable_rerank"] and nodes:
        with observability.span(observability.STAGE_RERANK, input_count=len(nodes)) as span:
            before_order = [node["chunk_id"] for node in nodes]
            nodes, rerank_applied = _rerank_nodes(nodes, query)
            after_order = sorted(nodes, key=lambda item: item.get("rerank_score", 0), reverse=True)
            span.set_output(len(nodes), applied=rerank_applied,
                            order_changed=before_order[:10] != [n["chunk_id"] for n in after_order][:10])
            if not rerank_applied:
                span.status = observability.STATUS_DEGRADED
                span.detail["fallback"] = "使用融合分数排序"
                reason = "Rerank 未生效（服务不可用或返回不完整）"
                path_degradations.append({"stage": observability.STAGE_RERANK, "reason": reason,
                                          "fallback": "使用融合分数排序"})
                observability.note_degradation(observability.STAGE_RERANK, reason,
                                               "使用融合分数排序")

    for node in nodes:
        node["final_score"] = node.get("rerank_score", 0.0) if rerank_applied else node["fused_score"]
        paths = node.get("hit_paths") or ["chunk"]
        node["hit_source"] = "multi_path" if len(paths) > 1 else paths[0]
    nodes.sort(key=lambda item: item["final_score"], reverse=True)

    items = []
    for idx, node in enumerate(nodes[:config["rerank_top_n"]], start=1):
        if node["final_score"] < config["min_score"]:
            continue
        items.append({
            "rank": idx,
            "chunk_id": node["chunk_id"],
            "source": node["source"],
            "chunk_index": node["chunk_index"],
            "content": node["content"],
            "dense_score": round(node["dense_score"], 6) if node.get("dense_score") is not None else None,
            "sparse_score": round(node["sparse_score"], 6) if node.get("sparse_score") is not None else None,
            "dense_sim": round(node["dense_sim"], 4) if node.get("dense_sim") is not None else None,
            "sparse_sim": round(node["sparse_sim"], 4) if node.get("sparse_sim") is not None else None,
            "fused_score": round(node["fused_score"], 4),
            "final_score": round(node["final_score"], 4),
            "rerank_score": round(node.get("rerank_score", 0), 4) if rerank_applied else None,
            "hit_source": node["hit_source"],
            "hit_paths": node.get("hit_paths", []),
            "path_ranks": node.get("path_ranks", {}),
        })

    items = _expand_parent_context(items)

    # 观测：高级模式的完整过程（问题计划、各路命中、融合、降级）
    recorder = observability.current()
    if recorder is not None:
        for degradation in plan.degradations:
            recorder.degrade(degradation.get("stage", ""), degradation.get("reason", ""),
                             degradation.get("fallback", ""))
        recorder.set_result(
            config_snapshot=config,
            rewritten_question=plan.rewritten,
            expansions=plan.multi_queries,
            hyde_answer=plan.hyde_answer,
            recall_empty=(len(items) == 0),
            mode="advanced",
            fusion_method=config["fusion_method"],
            fusion={key: value for key, value in fusion_debug.items() if key != "path_stats"},
            path_hits={path: len(hits) for path, hits in path_hits.items()},
            after_merge=len(nodes),
            after_filter=len(items),
            rerank_applied=rerank_applied,
            top_items=_summarize_items(items),
        )

    elapsed_ms = round((time.time() - start) * 1000, 2)
    sim_values = [item["final_score"] for item in items]
    avg_similarity = round(sum(sim_values) / len(sim_values), 4) if sim_values else 0.0

    debug_info = {}
    if debug:
        debug_info = {
            "pipeline": {
                "mode": "advanced",
                "retrieval_mode": retrieval_mode,
                "enable_rerank": config["enable_rerank"],
                "enable_index": config["enable_index"],
                "fusion_method": config["fusion_method"],
                "rerank_applied": rerank_applied,
            },
            "query_plan": plan.debug(),
            "fusion": fusion_debug,
            "path_hits": {path: len(hits) for path, hits in path_hits.items()},
            "config_snapshot": {
                "per_path_k": config["per_path_k"],
                "total_candidate_cap": config["total_candidate_cap"],
                "candidate_k": config["candidate_k"],
                "rerank_top_n": config["rerank_top_n"],
                "min_score": config["min_score"],
                "dense_weight": config["dense_weight"],
                "sparse_weight": config["sparse_weight"],
                "max_context_chars": config["max_context_chars"],
            },
            "retrieval": {
                "search_queries": [text for _, text in plan.search_queries()],
                "before_fusion": sum(len(hits) for hits in path_hits.values()),
                "after_merge": len(nodes),
                "after_filter": len(items),
            },
            "latency_ms": elapsed_ms,
        }

    return {
        "items": items,
        "total": len(items),
        "avg_similarity": avg_similarity,
        "debug": debug_info,
        "query_plan": plan.debug(),          # 调试信息之外单独回传，供流式与观测使用
        # 前置能力降级（改写/多查询/HyDE）+ 检索路径降级（索引问题检索/Rerank）
        "degradations": list(plan.degradations) + path_degradations,
        "config_snapshot": dict(config),      # 本次实际生效的完整策略快照
        "fusion": fusion_debug,
    }


def _expand_parent_context(items: list[dict]) -> list[dict]:
    """父子分块：把命中的**子块**映射到父块，并按父块去重（Advanced RAG 第 2 期）。

    规则（对应需求 2.2）：
      - 检索单元是较小的子块（向量库里只有子块）；
      - 生成回答时提供子块所属**父块的完整上下文**（写入 item["content"]）；
      - 同一父块被多个子块命中时只保留一条（取 final_score 最高者作代表），
        其余子块记录在 merged_children 里，避免重复把同一段内容喂给大模型；
      - 引用可定位到：原始文档 source + 父块序号 + 命中的子块序号/内容。

    通用分块（文档没有父块）时原样返回。
    """
    rows_by_id = database.get_chunks_by_ids([item.get("chunk_id") for item in items])
    if not rows_by_id:
        return items
    parent_ids = sorted({
        row.get("parent_chunk_id") for row in rows_by_id.values() if row.get("parent_chunk_id")
    })
    if not parent_ids:
        return items  # 通用分块：无需回查父块
    parents = database.get_chunks_by_ids(parent_ids)

    buckets: dict[str, list[dict]] = {}
    expanded: list[dict] = []
    for item in items:
        row = rows_by_id.get(item.get("chunk_id") or "")
        parent_id = (row or {}).get("parent_chunk_id") or ""
        if not parent_id:
            expanded.append(item)
            continue
        entry = dict(item)
        entry.update({
            "parent_chunk_id": parent_id,
            "parent_index": (row or {}).get("parent_index", -1),
            "sense_index": (row or {}).get("sense_index", 0),
            "child_chunk_id": item.get("chunk_id"),
            "child_content": item["content"],
        })
        buckets.setdefault(parent_id, []).append(entry)

    for parent_id, entries in buckets.items():
        entries.sort(key=lambda e: (e.get("final_score") or 0), reverse=True)
        best = entries[0]
        parent_row = parents.get(parent_id) or {}
        parent_content = parent_row.get("content") or best.get("content") or ""
        best["parent_content"] = parent_content
        best["content"] = parent_content          # 喂给大模型的是父块完整上下文
        best["length"] = parent_row.get("length") or len(parent_content)
        best["parent_chunk_index"] = parent_row.get("chunk_index", best.get("parent_index"))
        best["hit_child_count"] = len(entries)
        best["merged_children"] = [
            {
                "chunk_id": entry.get("child_chunk_id"),
                "sense_index": entry.get("sense_index"),
                "content": entry.get("child_content"),
                "final_score": entry.get("final_score"),
            }
            for entry in entries
        ]
        expanded.append(best)

    expanded.sort(key=lambda e: (e.get("final_score") or 0), reverse=True)
    for rank, entry in enumerate(expanded, start=1):
        entry["rank"] = rank
    return expanded


# ── Rerank 精排 ──────────────────────────────────────────

def _rerank_nodes(nodes: list[dict], query: str) -> tuple[list[dict], bool]:
    """
    使用 BGE Rerank 对检索结果重新排序。

    参数:
        nodes: 检索结果列表
        query: 用户查询文本

    返回:
        (重排后的 nodes, 是否成功执行 rerank)
    """
    try:
        contents = [n["content"] for n in nodes]
        rerank_results = embedding.bge_model.rerank(query, contents, top_k=len(nodes))

        if not rerank_results:
            return nodes, False

        # 先验证完整返回，避免部分 rerank 分数与原融合分数混排。
        scores = {}
        for item in rerank_results:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(nodes):
                scores[idx] = float(item["relevance_score"])
        if len(scores) != len(nodes):
            return nodes, False
        # 只写 rerank_score，不覆盖 fused_score（final_score 在 recall_test 里统一计算）
        for idx, score in scores.items():
            nodes[idx]["rerank_score"] = score
        return nodes, True
    except Exception as exc:
        print(f"[WARNING] Rerank 失败，使用原始分数排序: {exc}")
        return nodes, False


# ── RAG 问答（Ch6 新增）─────────────────────────────────

def prepare_context(kb_id: str, question: str, session_id: str | None, context_turns: int = 3,
                    strategy: dict | None = None) -> dict:
    """
    阶段一：检索 + 拼装上下文（普通问答与三种流式共用）。

    Ch12 新增：将 rag_query 中的"会话处理 + 保存用户消息 + 检索 + 拼接 messages"
    拆为独立的 prepare_context，供同步 invoke 与流式 stream_tokens_generator 复用。

    用户消息的保存与 assistant 回答是否成功是两件事：生成失败时不把错误文本写入。
    空召回（无相关文档）返回 messages=None、citations=[]，由生成器发出固定提示并正常 done。

    返回:
        {
            "session_id": str,        # 会话 ID
            "messages": list | None,  # 传给 LLM 的消息列表；空召回时为 None
            "citations": list[dict],  # 引用来源（空召回时为 []）
        }
    """
    import database
    from embedding import llm_client, SystemMessage, HumanMessage
    from langchain_core.messages import AIMessage

    # 1. 获取或创建会话
    session = database.ensure_session(session_id, kb_id)
    sid = session["id"]

    # 2. 首次提问时自动设置会话标题
    if session["title"] == "新对话":
        database.update_session_title(sid, question[:20])

    # 3. 先加载历史，再保存当前问题
    history = database.get_session_messages(sid)
    # 最多保留最近 N * 2 条历史消息
    if context_turns > 0:
        max_messages = context_turns * 2
        history = history[-max_messages:] if len(history) > max_messages else history

    # 4. 保存当前用户消息
    database.save_message(sid, "user", question)

    # 5. 检索相关文档（高级模式会用到历史做问题改写，因此把 history 传进去）
    retrieval = recall_test(
        kb_id=kb_id,
        query=question,
        retrieval_mode="hybrid",
        history=history,
        strategy=strategy,
    )
    extra = {
        "strategy_snapshot": retrieval.get("config_snapshot"),
        "query_plan": retrieval.get("query_plan"),
        "degradations": retrieval.get("degradations", []),
        "fusion": retrieval.get("fusion"),
    }

    items = retrieval["items"]
    if not items:
        # 空召回：不发消息给 LLM，也不保存 assistant；由生成器发固定提示
        return {"session_id": sid, "messages": None, "citations": [], **extra}

    # 6. 拼接上下文 + 构建引用
    #    父子分块时 item["content"] 已是父块完整上下文（见 _expand_parent_context），
    #    引用里同时保留"命中的子块"，这样引用可定位到 文档 + 父块 + 子块。
    context_parts = []
    citations = []
    # 最终上下文总长度受策略约束（max_context_chars）：超出预算就截断/丢弃后面的块，
    # 引用只保留真正进入上下文的那几条，保证"上下文 ↔ 引用来源"一一对应。
    max_context_chars = int((retrieval.get("config_snapshot") or {}).get("max_context_chars") or 6000)
    used_chars = 0
    _context_started = time.time()
    for item in items:
        if item.get("parent_chunk_id"):
            label = (f"[来源: {item['source']}, 父块 #{item.get('parent_index')}"
                     f"（命中子块 #{item.get('sense_index')}）]")
        else:
            label = f"[来源: {item['source']}, 块 #{item['chunk_index']}]"
        piece = f"{label}\n{item['content']}"
        # 块之间还要拼分隔符，预算必须把它算进去，否则总长度会超出上限
        overhead = len(_CONTEXT_SEPARATOR) if context_parts else 0
        remaining = max_context_chars - used_chars - overhead
        if remaining <= 0:
            break
        if len(piece) > remaining:
            piece = piece[:remaining]
        context_parts.append(piece)
        used_chars += overhead + len(piece)

        citation = {
            "source": item["source"],
            "chunk_index": item["chunk_index"],
            "content": item["content"],
            "fused_score": item["fused_score"],
            "rerank_score": item["rerank_score"],
            "final_score": item["final_score"],
        }
        if item.get("parent_chunk_id"):
            citation.update({
                "parent_chunk_id": item["parent_chunk_id"],
                "parent_index": item.get("parent_index"),
                "child_chunk_id": item.get("child_chunk_id"),
                "child_sense_index": item.get("sense_index"),
                "child_content": item.get("child_content"),
                "hit_child_count": item.get("hit_child_count", 1),
            })
        citations.append(citation)
    context = _CONTEXT_SEPARATOR.join(context_parts)
    # 观测：父块回查与上下文组装（含父子块数量与预算使用情况）
    observability.mark(
        observability.STAGE_CONTEXT,
        duration_ms=int((time.time() - _context_started) * 1000),
        input_count=len(items),
        output_count=len(citations),
        context_chars=used_chars,
        max_context_chars=max_context_chars,
        parent_blocks=sum(1 for citation in citations if citation.get("parent_chunk_id")),
        hit_children=sum(int(citation.get("hit_child_count") or 1) for citation in citations),
    )

    # 7. 构建 LLM 消息列表（Ch7 改进：加入历史对话）
    system_prompt = (
        "你是企业知识库问答助手。请仅根据给定的检索内容回答，不能编造。"
        "回答应简洁、专业，必要时用要点列出。"
        "如果检索内容不足以回答，请说明信息不足。"
        "如果用户的问题是前文对话的延续（如'它还有什么特点'），请结合历史对话和检索内容综合回答。"
    )
    messages = [SystemMessage(content=system_prompt)]

    # 加入历史对话
    for msg in history:
        messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                        else AIMessage(content=msg["content"]))

    # 加入当前问题
    user_prompt = f"参考内容：\n{context}\n\n用户问题：{question}"
    messages.append(HumanMessage(content=user_prompt))

    return {"session_id": sid, "messages": messages, "citations": citations, **extra}


def rag_query(kb_id: str, question: str, session_id: str | None = None, context_turns: int = 3,
              strategy: dict | None = None) -> dict:
    """
    RAG 问答核心流程：检索 → 拼接上下文 → LLM 生成 → 保存消息。

    Ch7 新增：多轮上下文支持，将最近 N 轮对话历史作为上下文传给 LLM。
    Ch12 重构：检索与拼上下文复用 prepare_context()，此处仅负责 invoke 生成。

    参数:
        kb_id: 知识库 ID
        question: 用户问题
        session_id: 会话 ID（None 则新建）
        context_turns: 上下文轮数（默认 3，即最近 3 轮 user+assistant 对话）

    返回:
        {
            "answer": str,           # LLM 生成的回答
            "citations": list[dict],  # 引用来源
            "session_id": str,        # 会话 ID
        }
    """
    import database
    from embedding import llm_client

    ctx = prepare_context(kb_id, question, session_id, context_turns=context_turns,
                          strategy=strategy)
    sid = ctx["session_id"]
    citations = ctx["citations"]
    messages = ctx["messages"]
    extra = {
        "strategy_snapshot": ctx.get("strategy_snapshot"),
        "query_plan": ctx.get("query_plan"),
        "degradations": ctx.get("degradations", []),
    }

    # 空召回：同步模式直接发固定提示并保存
    if messages is None:
        msg = "未检索到相关内容，请先上传文档到知识库。"
        database.save_message(sid, "assistant", msg)
        return {"answer": msg, "citations": [], "session_id": sid, **extra}

    # 调用 LLM 生成回答
    try:
        # LangSmith 未开启时原样返回 llm_client（零开销）
        response = langsmith_tracing.traced_llm(llm_client, mode="basic").invoke(messages)
        answer = response.content.strip()
    except Exception as exc:
        answer = f"大模型调用失败: {exc}"

    # 保存助手回答
    database.save_message(sid, "assistant", answer, citations)

    return {
        "answer": answer,
        "citations": citations,
        "session_id": sid,
        **extra,
    }


# ── 索引问题检索（Ch8 新增）─────────────────────────────

def recall_with_index_questions(kb_id: str, query: str, top_k: int = 5, min_score: float = 0.5,
                                enable_rerank: bool | None = None) -> dict:
    """
    索引问题检索流程（Milvus 向量检索）：
    1. 将查询文本向量化
    2. 在 Milvus 索引问题集合中进行 COSINE 向量检索
    3. 获取匹配的索引问题绑定的 chunk
    4. 【Ch8 课后题 3】对命中的相关 chunk 做与普通召回一致的 Rerank 精排，
       以 relevance_score 作为最终排序依据，保持行为与 recall_test 一致。

    返回:
        {"items": [...], "total": int, "matched_questions": int}
    """
    # 1. 将查询文本向量化
    emb = embedding.bge_model.embed(query)
    dense_vec = embedding.get_dense_vector(emb)

    # 2. 在 Milvus 中检索匹配的索引问题（候选略多于最终 top_k，过滤低分后更精准）
    matched = milvus_ops.search_index_questions(kb_id, dense_vec, top_k=top_k * 3)
    # 过滤低分结果（min_score 可由前端调整，默认 0.5）
    matched = [m for m in matched if m["score"] >= min_score]

    if not matched:
        return {"items": [], "total": 0, "matched_questions": 0, "matched_index_questions": []}

    # 3. 获取匹配的索引问题 ID
    index_ids = [m["pk"] for m in matched]

    # 4. 获取绑定的 chunk
    links = database.get_index_links_batch(index_ids)

    # 5. 构建分数映射
    score_map = {m["pk"]: m["score"] for m in matched}

    # 6. 组装结果（去重）
    seen = set()
    items = []

    for link in links:
        if link["kb_id"] != kb_id:
            continue
        key = f"{link['doc_id']}_{link['chunk_index']}"
        if key in seen:
            continue
        seen.add(key)

        items.append({
            "index_id": link["index_id"],
            "doc_id": link["doc_id"],
            "source": link["source"],
            "chunk_index": link["chunk_index"],
            "content": link["chunk_text"],
            "similarity": score_map.get(link["index_id"], 0.0),
            "hit_source": "index_question",
        })

    # 7. 【Ch8 课后题 3】与普通召回一致做 Rerank 精排：把命中的相关 chunk 文本与
    #    查询交给 rerank，用 relevance_score 作为最终排序依据（复用共享帮助函数
    #    _rerank_nodes，与 recall_test 行为一致，否则重排失败时按 similarity 排序）。
    config = _load_pipeline_config()
    if enable_rerank is None:
        enable_rerank = config.get("enable_rerank", True)
    rerank_applied = False
    if enable_rerank and items:
        items, rerank_applied = _rerank_nodes(items, query)
    for it in items:
        it["final_score"] = it["rerank_score"] if rerank_applied else it["similarity"]
        # 供前端可选展示：重排分保留原始字段，不覆盖 similarity
    items.sort(key=lambda x: x["final_score"], reverse=True)
    # 命中的索引问题清单（供前端"查看命中了哪些索引问题"展示）
    matched_index_questions = [
        {"index_id": m["pk"], "question": m.get("question", ""), "similarity": round(m["score"], 4)}
        for m in matched
    ]
    matched_index_questions.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "items": items[:top_k],
        "total": len(items[:top_k]),
        "matched_questions": len(index_ids),
        "matched_index_questions": matched_index_questions,
        "rerank_applied": rerank_applied,
    }