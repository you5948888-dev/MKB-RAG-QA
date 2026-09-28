"""
多路召回结果融合（Advanced RAG 迭代 · 第 3 期）。

需求："记录每个结果来自哪条检索路径、合并去重、统一排序、限制每路与总量、
避免直接比较量纲不同的原始分数、进入 Rerank 前控制候选规模"。

提供两种可选算法（策略里用 fusion_method 切换）：

1. **RRF（Reciprocal Rank Fusion，默认）**
   `score = Σ w_path / (k + rank_path)`（k=60）。
   - 只看**名次**，天然不比较不同路径的分数量纲（余弦相似度、内积、稀疏权重、
     索引问题相似度都不必归一化到同一尺度）；
   - 对单条路径的分数分布不敏感，离群值影响小；
   - 代价：忽略分数的绝对大小（两路都排第 1 与一路第 1 一路第 10 无法区分强弱），
     所以实现里保留 path_scores 供调试查看。

2. **weighted（路径内 min-max 归一化后加权求和）**
   `score = Σ w_path * norm_path(score)`。
   - 保留了"同一路径内分数差距"的信息，更贴近直觉；
   - 但 min-max 受该路径候选的分部影响（min==max 时统一定为 1.0），
     路径之间仍然只比较"相对位置"，不适合跨路径比较原始量纲。

两者都会：按 chunk_id 去重、每路取前 per_path_k、融合后截断到 total_candidate_cap。
"""
from typing import Iterable

DEFAULT_RRF_K = 60


def normalize_hit(hit: dict) -> dict:
    """把不同检索路径的命中统一成同一结构（chunk_id / source / content / score）。"""
    chunk_id = str(hit.get("chunk_id") or hit.get("pk") or "")
    content = hit.get("content")
    if content is None:
        content = hit.get("text") or ""
    score = hit.get("score")
    if score is None:
        score = hit.get("fused_score") or 0.0
    return {
        "chunk_id": chunk_id,
        "source": hit.get("source", ""),
        "chunk_index": hit.get("chunk_index", -1),
        "content": content,
        "score": float(score),
        "dense_score": hit.get("dense_score"),
        "sparse_score": hit.get("sparse_score"),
        "dense_sim": hit.get("dense_sim"),
        "sparse_sim": hit.get("sparse_sim"),
    }


def _min_max(values: list[float]) -> list[float]:
    """路径内 min-max 归一化；全部相等时统一定为 1.0（避免除零与全 0）。"""
    low, high = min(values), max(values)
    span = high - low
    if span <= 1e-9:
        return [1.0 for _ in values]
    return [(value - low) / span for value in values]


def fuse(path_hits: dict[str, Iterable[dict]], *, method: str = "rrf", k: int = DEFAULT_RRF_K,
         weights: dict[str, float] | None = None, path_weights: dict[str, float] | None = None,
         per_path_k: int = 20, total_cap: int = 60) -> tuple[list[dict], dict]:
    """融合多路召回结果。

    参数:
        path_hits: {path_id: [hit, ...]}，每个 hit 是**该路径内已排好序**的命中
        method: "rrf" | "weighted"
        weights / path_weights: 路径权重（默认 1.0，预留调参入口）
        per_path_k: 每条路径最多取多少候选
        total_cap: 融合后保留多少候选（进入 Rerank 前控制规模）

    返回: (融合后的节点列表（按 fused_score 降序）, debug 信息)
    """
    if method not in ("rrf", "weighted"):
        raise ValueError(f"不支持的融合算法: {method!r}")

    weight_map = dict(weights or path_weights or {})
    nodes: dict[str, dict] = {}
    path_stats: dict[str, dict] = {}
    raw_total = 0

    for path_id, hits in (path_hits or {}).items():
        ranked = [normalize_hit(hit) for hit in (hits or [])]
        raw_total += len(ranked)
        ranked = [hit for hit in ranked if hit["chunk_id"]][:max(per_path_k, 1)]
        path_stats[path_id] = {"hits": len(hits or []), "used": len(ranked)}
        if not ranked:
            continue

        weight = float(weight_map.get(path_id, 1.0))
        normalized = _min_max([hit["score"] for hit in ranked]) if method == "weighted" else []

        for rank, hit in enumerate(ranked, start=1):
            node = nodes.get(hit["chunk_id"])
            if node is None:
                node = {
                    **hit,
                    "fused_score": 0.0,
                    "hit_paths": [],
                    "path_ranks": {},
                    "path_scores": {},
                }
                nodes[hit["chunk_id"]] = node
            node["hit_paths"].append(path_id)
            node["path_ranks"][path_id] = rank
            node["path_scores"][path_id] = round(hit["score"], 6)
            if method == "rrf":
                node["fused_score"] += weight / (k + rank)
            else:
                node["fused_score"] += weight * normalized[rank - 1]
            # 保留"最相关那条路径"的分量信息，便于引用展示与排查
            if node.get("dense_sim") is None and hit.get("dense_sim") is not None:
                node["dense_sim"] = hit["dense_sim"]
                node["sparse_sim"] = hit["sparse_sim"]

    ordered = sorted(nodes.values(), key=lambda item: item["fused_score"], reverse=True)
    kept = ordered[:max(total_cap, 1)]
    for rank, node in enumerate(kept, start=1):
        node["fused_rank"] = rank
        node["fused_score"] = round(node["fused_score"], 6)

    debug = {
        "method": method,
        "rrf_k": k if method == "rrf" else None,
        "path_stats": path_stats,
        "raw_hits": raw_total,
        "unique_candidates": len(nodes),
        "kept_candidates": len(kept),
        "dedup_removed": raw_total - len(nodes),
        "per_path_k": per_path_k,
        "total_candidate_cap": total_cap,
    }
    return kept, debug
