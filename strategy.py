"""
检索策略（Advanced RAG 迭代 · 第 3 期）。

职责：
  - 定义策略的默认值、合法取值与校验（非法参数给明确中文原因）；
  - 解析"生效配置"的优先级：**请求临时覆盖 > 知识库策略 > 全局 pipeline_config.json**；
  - 提供配置快照（写入问答响应/追踪），保证"配置改动只影响后续请求"。

策略不落在这里：知识库策略存在 `kb_retrieval_strategies` 表（database 层读写），
本模块只负责语义（默认值/校验/合并）。
"""
import hashlib
import json
import logging

logger = logging.getLogger("rag.strategy")

RETRIEVAL_MODES = ("basic", "advanced")
FUSION_METHODS = ("rrf", "weighted")
REWRITE_MODES = ("off", "history")     # 目前只有"按最近对话历史改写"一种策略
MAX_MULTI_QUERY_COUNT = 8

# 只在"高级模式"下才会执行的能力（改写/多查询/HyDE）。
# 这些能力与 mode 必须一致，否则会出现"勾了开关却没生效"的静默失效。
ADVANCED_ABILITY_KEYS = ("enable_rewrite", "enable_multi_query", "enable_hyde")

DEFAULT_STRATEGY = {
    # 流程开关
    "mode": "basic",                 # basic=保持原有检索流程；advanced=启用高级能力组合
    "enable_rewrite": False,         # 问题改写（根据最近对话历史补全指代）
    "enable_multi_query": False,     # 多查询扩展
    "multi_query_count": 3,          # 扩展问题数量（不含原问题）
    "enable_hyde": False,            # 假设答案检索
    "hyde_max_chars": 500,           # 假设答案长度上限（控制调用成本）
    # 检索与融合
    "dense_weight": 0.5,
    "sparse_weight": 0.5,
    "enable_index": True,            # 索引问题召回
    "fusion_method": "rrf",          # rrf=按名次融合；weighted=路径内归一化后加权
    "per_path_k": 20,                # 每条检索路径的候选上限
    "total_candidate_cap": 60,       # 进入 Rerank 前的总候选上限
    "candidate_k": 20,               # 单次向量检索候选数
    # 精排与上下文
    "enable_rerank": True,
    "rerank_top_n": 5,               # Rerank 后最终保留数量
    "min_score": 0.0,                # 最低相关度（低于则过滤）
    "max_context_chars": 6000,       # 最终上下文总长度上限（字符口径）
}

_INT_KEYS = {
    "multi_query_count": (1, MAX_MULTI_QUERY_COUNT),
    "hyde_max_chars": (50, 4000),
    "per_path_k": (1, 100),
    "total_candidate_cap": (1, 500),
    "candidate_k": (1, 100),
    "rerank_top_n": (1, 50),
}
_FLOAT_KEYS = {
    "dense_weight": (0.0, 1.0),
    "sparse_weight": (0.0, 1.0),
    "min_score": (0.0, 1.0),
}


class StrategyConfigError(ValueError):
    """策略参数非法（路由层捕获后返回 400 + 明确提示）。"""


def _as_bool(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "是"}


def _as_int(value, default: int, key: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise StrategyConfigError(f"{key} 必须是整数，当前值: {value!r}")


def _as_float(value, default: float, key: str) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise StrategyConfigError(f"{key} 必须是数字，当前值: {value!r}")


def normalize_strategy(raw: dict | None = None) -> dict:
    """归一化（类型 + 默认值），不做范围校验。"""
    source = dict(raw or {})
    config = {
        "mode": str(source.get("mode") or DEFAULT_STRATEGY["mode"]).strip().lower(),
        "enable_rewrite": _as_bool(source.get("enable_rewrite"), DEFAULT_STRATEGY["enable_rewrite"]),
        "enable_multi_query": _as_bool(source.get("enable_multi_query"), DEFAULT_STRATEGY["enable_multi_query"]),
        "multi_query_count": _as_int(source.get("multi_query_count"),
                                     DEFAULT_STRATEGY["multi_query_count"], "multi_query_count"),
        "enable_hyde": _as_bool(source.get("enable_hyde"), DEFAULT_STRATEGY["enable_hyde"]),
        "hyde_max_chars": _as_int(source.get("hyde_max_chars"),
                                  DEFAULT_STRATEGY["hyde_max_chars"], "hyde_max_chars"),
        "dense_weight": _as_float(source.get("dense_weight"), DEFAULT_STRATEGY["dense_weight"], "dense_weight"),
        "sparse_weight": _as_float(source.get("sparse_weight"), DEFAULT_STRATEGY["sparse_weight"], "sparse_weight"),
        "enable_index": _as_bool(source.get("enable_index"), DEFAULT_STRATEGY["enable_index"]),
        "fusion_method": str(source.get("fusion_method") or DEFAULT_STRATEGY["fusion_method"]).strip().lower(),
        "per_path_k": _as_int(source.get("per_path_k"), DEFAULT_STRATEGY["per_path_k"], "per_path_k"),
        "total_candidate_cap": _as_int(source.get("total_candidate_cap"),
                                       DEFAULT_STRATEGY["total_candidate_cap"], "total_candidate_cap"),
        "candidate_k": _as_int(source.get("candidate_k"), DEFAULT_STRATEGY["candidate_k"], "candidate_k"),
        "enable_rerank": _as_bool(source.get("enable_rerank"), DEFAULT_STRATEGY["enable_rerank"]),
        "rerank_top_n": _as_int(source.get("rerank_top_n"), DEFAULT_STRATEGY["rerank_top_n"], "rerank_top_n"),
        "min_score": _as_float(source.get("min_score"), DEFAULT_STRATEGY["min_score"], "min_score"),
        "max_context_chars": _as_int(source.get("max_context_chars"),
                                     DEFAULT_STRATEGY["max_context_chars"], "max_context_chars"),
    }
    if config["max_context_chars"] < 200:
        config["max_context_chars"] = DEFAULT_STRATEGY["max_context_chars"]
    return config


def validate_strategy(raw: dict | None = None) -> dict:
    """归一化 + 校验，返回可直接使用的策略；非法时抛 StrategyConfigError。"""
    config = normalize_strategy(raw)

    if config["mode"] not in RETRIEVAL_MODES:
        raise StrategyConfigError(
            f"检索模式仅支持 {' / '.join(RETRIEVAL_MODES)}，当前值: {config['mode']!r}"
        )
    if config["fusion_method"] not in FUSION_METHODS:
        raise StrategyConfigError(
            f"融合算法仅支持 {' / '.join(FUSION_METHODS)}，当前值: {config['fusion_method']!r}"
        )
    for key, (low, high) in _INT_KEYS.items():
        if not low <= config[key] <= high:
            raise StrategyConfigError(f"{key} 需在 {low}-{high} 之间，当前值: {config[key]}")
    for key, (low, high) in _FLOAT_KEYS.items():
        if not low <= config[key] <= high:
            raise StrategyConfigError(f"{key} 需在 {low}-{high} 之间，当前值: {config[key]}")
    if config["per_path_k"] > config["total_candidate_cap"] * 3:
        raise StrategyConfigError("per_path_k 过大（相对总候选上限），会显著增加延迟")
    if config["dense_weight"] == 0 and config["sparse_weight"] == 0:
        raise StrategyConfigError("dense_weight 与 sparse_weight 不能同时为 0")
    if config["rerank_top_n"] > config["total_candidate_cap"]:
        raise StrategyConfigError("rerank_top_n 不能大于 total_candidate_cap")
    return config


def has_advanced_ability(config: dict | None) -> bool:
    """是否开启了"只在高级模式下生效"的能力（改写/多查询/HyDE）。"""
    return any(bool((config or {}).get(key)) for key in ADVANCED_ABILITY_KEYS)


def ensure_mode_consistency(config: dict) -> dict:
    """保存策略时的**严格校验**（fail fast）。

    规则：开启改写/多查询/HyDE 任一项时，运行模式必须是 advanced——
    否则这些开关不会生效，用户会以为功能坏了（生产环境最怕"配置被接受但被忽略"）。
    历史脏数据的运行时兜底见 resolve()。
    """
    if config.get("mode") != "advanced" and has_advanced_ability(config):
        enabled = "、".join(key for key in ADVANCED_ABILITY_KEYS if config.get(key))
        raise StrategyConfigError(
            f"已开启高级能力（{enabled}），运行模式必须是 advanced："
            "这些能力只在高级模式下生效；若只想做基础检索对比，请先关闭能力开关"
        )
    return config


def canonical_json(config: dict) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def strategy_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


_inconsistent_warned = False


def _warn_inconsistent_config() -> None:
    """历史脏数据只告警一次，避免每个请求都刷日志。"""
    global _inconsistent_warned
    if _inconsistent_warned:
        return
    _inconsistent_warned = True
    logger.warning(
        "检测到矛盾的历史检索策略（mode=basic 但已开启高级能力），本次按高级模式执行；"
        "请在「检索策略」弹窗里重新保存一次以消除该状态"
    )


def resolve(kb_strategy: dict | None = None, override: dict | None = None,
            global_defaults: dict | None = None) -> dict:
    """解析生效策略，优先级：请求临时覆盖 > 知识库策略 > 全局配置 > 内置默认值。

    global_defaults 来自 pipeline_config.json（Ch8 起就存在的全局开关），
    只映射其中语义相同的字段，避免两套配置互相覆盖。
    """
    merged: dict = {}
    if global_defaults:
        for key in ("enable_rerank", "dense_weight", "sparse_weight",
                    "retrieval_top_k", "candidate_k"):
            if key not in global_defaults:
                continue
            target = "rerank_top_n" if key == "retrieval_top_k" else key
            merged[target] = global_defaults[key]
    if kb_strategy:
        merged.update({k: v for k, v in kb_strategy.items() if v is not None})
    if override:
        merged.update({k: v for k, v in override.items() if v is not None})

    # 运行模式与高级能力的一致性：
    # - 请求级**显式**指定 mode（召回测试做基础/高级对比）→ 完全尊重调用方，不改写；
    # - 保存接口已用 ensure_mode_consistency 禁止产生矛盾配置，这里只兜底**历史脏数据**：
    #   若仍是"基础模式 + 已开高级能力"，则提升为高级模式，并在快照里标记 mode_adjusted，
    #   同时打一条告警（显式可审计，不静默改变行为）。
    override_mode = (override or {}).get("mode")
    mode_adjusted = False
    if override_mode:
        merged["mode"] = override_mode
    elif merged.get("mode") != "advanced" and has_advanced_ability(merged):
        merged["mode"] = "advanced"
        mode_adjusted = True
        _warn_inconsistent_config()

    config = validate_strategy(merged)
    if mode_adjusted:
        config["mode_adjusted"] = True
    return config
