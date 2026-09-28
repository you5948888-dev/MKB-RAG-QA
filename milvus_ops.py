"""
Milvus 向量数据库操作层。
本章功能：集合创建、向量写入（upsert）、向量删除。
"""
from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    BGE_DENSE_DIM,
    INDEX_QUESTION_COLLECTION,  # Ch8 新增
)


def get_milvus_client():
    """获取 Milvus Collection 对象，并确保已连接。"""
    from pymilvus import connections, Collection
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    return Collection(MILVUS_COLLECTION)


def ensure_collection() -> None:
    """确保 hybrid 集合存在，不存在则创建。"""
    from pymilvus import (
        connections, utility, FieldSchema, CollectionSchema, DataType, Collection,
    )

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(MILVUS_COLLECTION):
        return

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=BGE_DENSE_DIM),
        FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields=fields, description="RAG Tutorial hybrid collection")
    coll = Collection(name=MILVUS_COLLECTION, schema=schema)

    # dense 向量：HNSW 索引 + COSINE 度量
    coll.create_index(
        field_name="dense",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    # sparse 向量：稀疏倒排索引 + IP 度量
    coll.create_index(
        field_name="sparse",
        index_params={
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2},
        },
    )


def upsert_vectors(rows: list[dict]) -> None:
    """批量写入向量到 Milvus。"""
    if not rows:
        return
    ensure_collection()
    client = get_milvus_client()
    batch_size = 32
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        client.upsert(batch)
    client.flush()


def delete_vectors_by_pk(pks: list[str]) -> None:
    """按主键批量删除向量。"""
    if not pks:
        return
    ensure_collection()
    client = get_milvus_client()
    for start in range(0, len(pks), 1024):
        batch = pks[start : start + 1024]
        client.delete(f"pk in {batch}")
    client.flush()


def delete_vectors_by_kb(kb_id: str) -> None:
    """按知识库 ID 删除该库下所有向量。"""
    ensure_collection()
    client = get_milvus_client()
    client.delete(f'kb_id == "{kb_id}"')
    client.flush()


def delete_vectors_for_doc(kb_id: str, filename: str) -> None:
    """按知识库 + 文件名删除该文档的所有向量（Ch10 新增）。

    用于任务失败回滚（_cleanup_inserted_vectors）和重建索引时清除旧版本向量。
    source 字段存的就是原始文件名，因此按 (kb_id, source) 精确定位该文档的向量。
    """
    ensure_collection()
    client = get_milvus_client()
    client.delete(f'kb_id == "{kb_id}" and source == "{filename}"')
    client.flush()


def search_vectors(
    kb_id: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    retrieval_mode: str = "hybrid",
    top_k: int = 20,
) -> list[dict]:
    """
    在指定知识库内进行向量检索。

    参数:
        kb_id: 知识库 ID（用于过滤）
        dense_vec: dense 查询向量（1024 维）
        sparse_vec: sparse 查询向量（{维度索引: 权重值}）
        retrieval_mode: 检索模式 — "dense_only" / "sparse_only" / "hybrid"
        top_k: 返回结果数量

    返回:
        [{"pk": str, "source": str, "chunk_index": int, "text": str,
          "dense_score": float, "sparse_score": float}, ...]
    """
    from pymilvus import connections, Collection

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    ensure_collection()
    coll = Collection(MILVUS_COLLECTION)
    coll.load()

    expr = f'kb_id == "{kb_id}"'
    output_fields = ["pk", "source", "chunk_index", "text"]

    dense_hits = []
    sparse_hits = []

    if retrieval_mode in {"dense_only", "hybrid"}:
        dense_hits = coll.search(
            data=[dense_vec],
            anns_field="dense",
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
            param={"metric_type": "COSINE", "params": {"ef": 128}},
        )[0]

    if retrieval_mode in {"sparse_only", "hybrid"}:
        sparse_hits = coll.search(
            data=[sparse_vec],
            anns_field="sparse",
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
            param={"metric_type": "IP", "params": {}},
        )[0]

    # 合并两路结果，以 pk 去重
    merged: dict[str, dict] = {}
    for h in dense_hits:
        pk = str(h.id)
        merged[pk] = {
            "pk": pk,
            "source": h.entity.get("source"),
            "chunk_index": h.entity.get("chunk_index"),
            "text": h.entity.get("text"),
            "dense_score": float(h.distance),
            "sparse_score": 0.0,
        }
    for h in sparse_hits:
        pk = str(h.id)
        if pk in merged:
            merged[pk]["sparse_score"] = float(h.distance)
        else:
            merged[pk] = {
                "pk": pk,
                "source": h.entity.get("source"),
                "chunk_index": h.entity.get("chunk_index"),
                "text": h.entity.get("text"),
                "dense_score": 0.0,
                "sparse_score": float(h.distance),
            }

    return list(merged.values())


# ── 索引问题集合（Ch8 新增）──────────────────────────────

def ensure_index_question_collection() -> None:
    """确保索引问题集合存在，不存在则创建。仅使用 dense 向量（COSINE）。"""
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(INDEX_QUESTION_COLLECTION):
        return

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=BGE_DENSE_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="索引问题向量集合")
    coll = Collection(name=INDEX_QUESTION_COLLECTION, schema=schema)

    coll.create_index(
        field_name="dense",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )


def upsert_index_question_vector(index_id: str, kb_id: str, question: str, dense_vec: list[float]) -> None:
    """将索引问题的向量写入 Milvus。

    【易错】不能复用 get_milvus_client()——它返回的是主集合 rag_tutorial_hybrid 的
    Collection 对象，且 ORM 风格 upsert 不接受 collection_name 参数。
    必须显式打开索引问题集合再写入。
    """
    from pymilvus import connections, Collection

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    ensure_index_question_collection()
    coll = Collection(INDEX_QUESTION_COLLECTION)
    coll.upsert([{
        "pk": index_id,
        "kb_id": kb_id,
        "question": question,
        "dense": dense_vec,
    }])
    coll.flush()


def delete_index_question_vector(index_id: str) -> None:
    """删除索引问题在 Milvus 中的向量。"""
    try:
        from pymilvus import connections, Collection
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        ensure_index_question_collection()
        coll = Collection(INDEX_QUESTION_COLLECTION)
        coll.load()
        coll.delete(expr=f'pk == "{index_id}"')
    except Exception as exc:
        print(f"[WARNING] 删除索引问题向量失败: {exc}")


def delete_index_question_vectors_for_kb(kb_id: str) -> None:
    """删除某个知识库的所有索引问题向量。"""
    try:
        from pymilvus import connections, Collection
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        ensure_index_question_collection()
        coll = Collection(INDEX_QUESTION_COLLECTION)
        coll.load()
        coll.delete(expr=f'kb_id == "{kb_id}"')
    except Exception as exc:
        print(f"[WARNING] 删除索引问题向量失败: {exc}")


def delete_index_question_vectors_by_ids(index_ids: list[str]) -> None:
    """按主键批量删除索引问题向量（Ch11 新增：清理索引问题时调用）。"""
    if not index_ids:
        return
    try:
        from pymilvus import connections, Collection
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        ensure_index_question_collection()
        coll = Collection(INDEX_QUESTION_COLLECTION)
        coll.load()
        for start in range(0, len(index_ids), 200):
            batch = index_ids[start : start + 200]
            coll.delete(expr=f"pk in {batch}")
    except Exception as exc:
        print(f"[WARNING] 删除索引问题向量失败: {exc}")


def search_index_questions(kb_id: str, dense_vec: list[float], top_k: int = 5) -> list[dict]:
    """
    在索引问题集合中进行向量检索。

    返回:
        [{"pk": str, "question": str, "score": float}, ...]
        score 为截断到 [0, 1] 的 COSINE 相似度，越大越相似
    """
    from pymilvus import connections, Collection

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    ensure_index_question_collection()
    coll = Collection(INDEX_QUESTION_COLLECTION)
    coll.load()

    expr = f'kb_id == "{kb_id}"'
    output_fields = ["pk", "question"]

    hits = coll.search(
        data=[dense_vec],
        anns_field="dense",
        limit=top_k,
        expr=expr,
        output_fields=output_fields,
        param={"metric_type": "COSINE", "params": {"ef": 64}},
    )[0]

    results = []
    for h in hits:
        # distance 是 SDK 字段名；COSINE 指标下它就是相似度。
        similarity = max(0.0, min(1.0, float(h.distance)))
        results.append({
            "pk": str(h.id),
            "question": h.entity.get("question", ""),
            "score": similarity,
        })
    return results