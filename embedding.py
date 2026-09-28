"""
BGE Embedding 服务封装。
本章只使用 embed（生成 dense + sparse 向量），rerank 功能在 Ch5 引入。
服务地址用支持稀疏索引的 50183（python 容器服务，返回 base64+pickle）。
"""
import base64
import pickle

import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import BGE_URL, LLM_API_URL, LLM_MODEL_NAME, LLM_API_KEY, LLM_REQUEST_TIMEOUT


class BgeModel:
    """BGE Embedding 服务客户端。"""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        # 复用连接，避免每个请求都新建 TCP 连接（批量建索引时性能差异明显）
        self.session = requests.Session()

    def embed(self, text: str) -> dict:
        """调用 BGE 服务生成向量。

        返回: {"dense": list[list[float]], "sparse": dict}
        - dense: 1024 维 float 列表（外层多包一层数组）
        - sparse: 稀疏向量（索引:权重）
        """
        url = f"{self.url}/api/emb/embedding"
        # 批量建索引需要较长等待，超时放宽到 120s
        response = self.session.post(url, json={"content": text}, timeout=120)
        response.raise_for_status()
        data = response.json()["data"]
        data["dense"] = pickle.loads(base64.b64decode(data["dense"]))
        data["sparse"] = pickle.loads(base64.b64decode(data["sparse"]))
        return data

    def rerank(self, query: str, contents: list[str], top_k: int = 20) -> list[dict]:
        """
        调用 BGE 服务对检索结果进行精排。

        参数:
            query: 用户查询文本
            contents: 候选文本列表
            top_k: 返回前 k 个结果

        返回:
            [{"index": int, "relevance_score": float, "text": str}, ...]
            按相关性分数降序排列
        """
        url = f"{self.url}/api/emb/rerank"
        response = self.session.post(
            url, json={"query": query, "contents": contents, "top_k": top_k}, timeout=120
        )
        results = response.json().get("data", [])
        # Ch6 统一 BGE Rerank 返回契约：有些服务返回 score，有些返回 relevance_score。
        # 在此归一化，后续代码只认 relevance_score 一种字段名。
        normalized_results = []
        for item in results:
            normalized_item = dict(item)
            if "relevance_score" not in normalized_item and "score" in normalized_item:
                normalized_item["relevance_score"] = normalized_item["score"]
            normalized_results.append(normalized_item)
        return normalized_results


# 全局实例
bge_model = BgeModel(BGE_URL)


def get_dense_vector(embedding: dict) -> list[float]:
    """从 embed 结果中提取 dense 向量（1024 维）。"""
    return embedding["dense"][0]


def get_sparse_vector(embedding: dict) -> dict[int, float]:
    """从 embed 结果中提取 sparse 向量。"""
    return embedding["sparse"]


# ── LLM 客户端（Ch6 新增）─────────────────────────────

llm_client = ChatOpenAI(
    base_url=LLM_API_URL,
    model=LLM_MODEL_NAME,
    api_key=LLM_API_KEY,
    request_timeout=LLM_REQUEST_TIMEOUT,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)