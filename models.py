"""
Pydantic 请求/响应模型。
FastAPI 使用 Pydantic 自动校验请求参数——类型不对、缺少必填字段时
会自动返回 422 错误，不用我们手动写 if 判断。
"""
from pydantic import BaseModel, Field


class KnowledgeBaseCreateRequest(BaseModel):
    """创建知识库的请求体"""

    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: str | None = Field(None, max_length=500, description="知识库描述")
    chunk_size: int = Field(800, ge=1, le=20000, description="分块最大长度（长度口径见 chunk_config）")
    chunk_overlap: int = Field(100, ge=0, le=19999, description="分块重叠长度，必须小于分块最大长度")
    # Advanced RAG 第 1 期：可选分块策略与完整配置（chunk_config 优先，逐项覆盖扁平字段）
    chunk_strategy: str | None = Field(None, description="分块策略: general / parent_child（后者后置开放）")
    chunk_config: dict | None = Field(None, description="完整分块配置（分隔符/长度口径/短块策略等）")


class ChunkConfigUpdateRequest(BaseModel):
    """修改知识库分块配置的请求体（Advanced RAG 第 1 期）"""

    chunk_config: dict = Field(..., description="完整分块配置对象")
    switch_and_reindex: bool = Field(
        False,
        description="知识库已有文档时，确认切换配置（随后需调用重新索引；否则返回 409）",
    )


class ReindexRequest(BaseModel):
    """重新索引请求（Advanced RAG 第 2 期）"""

    doc_ids: list[str] | None = Field(
        None, description="指定要重新索引的文档 ID；省略表示全部配置过期的文档"
    )


class ChunkPreviewRequest(BaseModel):
    """分块预览请求：给 text 预览文本，给 doc_id 预览已上传文档（二者取其一）"""

    text: str | None = Field(None, description="待预览的文本")
    doc_id: str | None = Field(None, description="待预览的文档 ID")
    chunk_config: dict | None = Field(None, description="临时覆盖的分块配置（仅影响本次预览）")


class KnowledgeBaseRenameRequest(BaseModel):
    """修改知识库名称的请求体"""

    name: str = Field(..., min_length=1, max_length=100, description="新名称")


class RecallTestRequest(BaseModel):
    """召回测试请求"""

    query: str = Field(..., min_length=1, description="查询文本")
    retrieval_mode: str = Field("hybrid", description="检索模式: dense_only / sparse_only / hybrid")
    top_k: int | None = Field(None, ge=1, le=20, description="返回结果数量；省略时使用策略配置")
    candidate_k: int | None = Field(None, ge=1, le=50, description="候选集数量；省略时使用策略配置")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="最低相似度阈值（0 表示不额外过滤）")
    dense_weight: float | None = Field(None, ge=0.0, le=1.0, description="Dense 向量权重")
    sparse_weight: float | None = Field(None, ge=0.0, le=1.0, description="Sparse 向量权重")
    enable_rerank: bool | None = Field(None, description="是否启用 Rerank 精排")
    enable_index: bool | None = Field(None, description="是否启用索引问题召回融合")  # Ch8 新增
    debug: bool = Field(False, description="是否返回调试信息")
    # ── Advanced RAG 第 3 期：本次调用的临时覆盖（只影响本次请求）──
    mode: str | None = Field(None, description="basic / advanced，覆盖知识库策略")
    history: list[str] | None = Field(
        None, description="上一轮问题（按时间顺序，用于问题改写；不传则改写拿不到历史）")
    fusion_method: str | None = Field(None, description="rrf / weighted，覆盖知识库策略")
    enable_rewrite: bool | None = Field(None, description="是否启用问题改写")
    enable_multi_query: bool | None = Field(None, description="是否启用多查询扩展")
    multi_query_count: int | None = Field(None, ge=1, le=8, description="扩展问题数量")
    enable_hyde: bool | None = Field(None, description="是否启用 HyDE 假设答案检索")
    rerank_top_n: int | None = Field(None, ge=1, le=50, description="Rerank 后最终保留数量")
    max_context_chars: int | None = Field(None, ge=200, le=50000, description="上下文总长度上限")


class RetrievalStrategyUpdateRequest(BaseModel):
    """修改知识库检索策略（Advanced RAG 第 3 期）"""

    strategy: dict = Field(..., description="完整检索策略对象")


class CleanupTracesRequest(BaseModel):
    """手动清理观测数据（Advanced RAG 第 4 期，仅管理员）"""

    mode: str = Field("older_than",
                      description="清理模式: all / older_than / range（最近N小时内）/ trace_ids")
    days: int = Field(7, ge=1, description="older_than 模式保留最近多少天")
    hours: int | None = Field(None, ge=1, le=8760,
                              description="range 模式：清理最近 N 小时内的记录（对应“最近1小时/24小时”这类快捷范围）")
    trace_ids: list[str] | None = Field(None, description="trace_ids 模式下要删除的追踪编号（支持批量）")
    dry_run: bool = Field(True, description="默认只统计将删除的条数，不真正删除")


# ── 索引问题请求模型（Ch8 新增）─────────────────────────

class IndexQuestionCreateRequest(BaseModel):
    """创建索引问题"""
    question: str = Field(..., min_length=1, max_length=500, description="索引问题")


class IndexQuestionUpdateRequest(BaseModel):
    """修改索引问题"""
    question: str = Field(..., min_length=1, max_length=500, description="索引问题")


class IndexQuestionLinkItem(BaseModel):
    """索引问题绑定的单个 chunk"""
    chunk_id: str = Field(..., description="Chunk ID（格式: {doc_id}_{chunk_index}）")


class IndexQuestionLinkRequest(BaseModel):
    """批量绑定索引问题到 chunk"""
    items: list[IndexQuestionLinkItem] = Field(..., min_length=1, description="绑定的 chunk 列表")


class IndexRetrieveRequest(BaseModel):
    """索引检索请求"""
    query: str = Field(..., min_length=1, description="查询文本")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")
    min_score: float = Field(0.5, description="命中索引问题的相似度阈值（越低命中越多）")


class UnlinkRequest(BaseModel):
    """解除绑定请求"""
    chunk_id: str = Field(..., description="Chunk ID")


class IndexBuildRequest(BaseModel):
    """自动生成索引问题请求"""
    prompt: str | None = Field(None, max_length=2000, description="自定义 prompt 模板，{data} 会被替换为 chunk 内容")


class CleanupRequest(BaseModel):
    """清理索引问题请求（Ch11 新增）。"""
    mode: str = Field("orphan", description="清理模式: orphan（无关联）或 all（全部）")


class QueryRequest(BaseModel):
    """RAG 问答请求"""

    question: str = Field(..., min_length=1, description="用户问题")
    kb_id: str = Field(..., min_length=1, description="知识库 ID")
    session_id: str | None = Field(None, description="会话 ID（不传则新建）")
    # Advanced RAG 第 3 期：本次问答的临时策略覆盖（不写库，不影响历史记录）
    mode: str | None = Field(None, description="basic / advanced，覆盖知识库策略")
    strategy: dict | None = Field(None, description="临时覆盖的检索策略字段")


class FeedbackRequest(BaseModel):
    """问答反馈请求"""

    answer_id: str = Field(..., description="回答 ID")
    score: int = Field(..., ge=1, le=5, description="评分 1-5")
    text: str | None = Field(None, description="反馈文字")


class RenameSessionRequest(BaseModel):
    """重命名会话请求（Ch7 新增）"""

    title: str = Field(..., min_length=1, max_length=100, description="新标题")


# ── 认证请求模型（Ch9 新增）─────────────────────────

class UserRegisterRequest(BaseModel):
    """注册请求"""

    username: str = Field(..., min_length=1, max_length=32, description="用户名")
    password: str = Field(..., min_length=4, description="密码")
    display_name: str | None = Field(None, max_length=50, description="显示名称")


class UserLoginRequest(BaseModel):
    """登录请求"""

    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")


class ChangePasswordRequest(BaseModel):
    """修改密码请求"""

    old_password: str = Field(..., min_length=1, description="旧密码")
    new_password: str = Field(..., min_length=4, description="新密码")