"""
分块引擎（Advanced RAG 迭代 · P1 通用分块 + P2 父子分块）。

设计要点：
1. **唯一入口**：文档预览、上传前预览、Worker 实际入库全部调用本模块的
   `chunk_text` / `chunk_document`，因此"预览结果 == 入库结果"是结构性保证，
   而不是靠两处参数人工对齐（这是本章验收场景 1 的核心）。
2. **配置可冻结**：知识库创建时把完整配置存进 `knowledge_bases.chunk_config`，
   上传时再把当时的配置快照写进 `upload_tasks.chunk_config`，Worker 只认任务里的
   快照——避免"排队等待期间改了配置"导致入库配置与预览时不一致。
3. **长度口径可切换**：`char` 用 `len()`；`token` 用 tiktoken 的 cl100k_base。
   【注意】cl100k 只是**尺寸近似口径**（BGE-M3 的真实 tokenizer 与之不同），
   用于控制分块尺寸与成本，不代表模型实际计费 token 数。
4. **父子分块**（P2）：父块与子块各自独立一套参数（`parent` / `child` 子字典），
   先按父块参数切出父块，再把每个父块按子块参数切成子块。入库时**只有子块进 Milvus**
   （检索单元），父块只落 SQLite（生成回答时的完整上下文来源）；
   顶层同时镜像子块参数与扁平字段，旧前端与知识库扁平列继续可用。
"""
import json
import hashlib
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("rag.chunking")

# ── 常量 ────────────────────────────────────────────────

STRATEGIES = ("general", "parent_child")
LENGTH_MODES = ("char", "token")
SHORT_CHUNK_POLICIES = ("merge", "drop", "keep")

# 通用分块的默认值（与项目历史行为保持一致：800/100 + 默认分隔符优先级）
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""]

DEFAULT_CONFIG = {
    "strategy": "general",
    "separators": list(DEFAULT_SEPARATORS),
    "chunk_size": 800,
    "chunk_overlap": 100,
    "length_mode": "char",
    "keep_separator": False,
    "filter_whitespace": True,
    "short_chunk_policy": "keep",
    "min_length": 0,
}

# 父子分块默认值（第 2 期）：
# 父块只用于"给大模型完整上下文"，不参与向量检索，所以默认更长、保留段落分隔符；
# 子块是检索单元，默认更小并带重叠。
DEFAULT_PARENT_CONFIG = {
    "separators": ["\n\n", "\n"],
    "chunk_size": 2000,
    "chunk_overlap": 0,          # 父块重叠没有意义（父块之间不共享上下文）
    "length_mode": "char",
    "keep_separator": True,      # 保留段落分隔符，父块文本更贴近原文
    "filter_whitespace": True,
    "short_chunk_policy": "keep",
    "min_length": 0,
}

DEFAULT_CHILD_CONFIG = {
    "separators": list(DEFAULT_SEPARATORS),
    "chunk_size": 300,
    "chunk_overlap": 50,
    "length_mode": "char",
    "keep_separator": False,
    "filter_whitespace": True,
    "short_chunk_policy": "merge",   # 过短子块并回相邻块，避免无意义碎片进入向量库
    "min_length": 30,
}

MAX_CHUNK_SIZE = 20000
MAX_SEPARATORS = 20


class ChunkConfigError(ValueError):
    """分块参数非法。路由层捕获后统一返回 400 + 明确提示。"""


# ── 长度口径 ────────────────────────────────────────────

_encoder = None


def _token_encoder():
    """惰性加载 tiktoken 编码器（避免 import 时就付出代价）。"""
    global _encoder
    if _encoder is None:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def token_length(text: str) -> int:
    """按 cl100k_base 计算 token 数（尺寸近似口径）。"""
    return len(_token_encoder().encode(text or ""))


def build_length_function(mode: str):
    """返回长度函数：char → len；token → tiktoken。"""
    if mode == "token":
        return token_length
    return len


# ── 配置归一化与校验 ────────────────────────────────────

def _as_bool(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "是"}


def _as_int(value, default: int, field: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ChunkConfigError(f"{field} 必须是整数，当前值: {value!r}")


def _unescape_separator(item: str) -> str:
    """支持在文本框里直接写 \\n / \\t 转义。"""
    return item.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def _normalize_separators(value, default) -> list[str]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                raise ChunkConfigError("separators 不是合法的 JSON 数组")
        else:
            # 简写：用 | 分隔多个分隔符，例如 "\n\n|\n|。"（\\\\n 会被转义成换行）
            value = text.split("|")
    if not isinstance(value, (list, tuple)):
        raise ChunkConfigError("separators 必须是字符串数组")
    if len(value) == 0:
        raise ChunkConfigError("至少需要配置一个分隔符")
    items = []
    for raw in value:
        if not isinstance(raw, str):
            raise ChunkConfigError("separators 的每一项都必须是字符串")
        items.append(_unescape_separator(raw))
    # 兜底：末尾必须带字符级分隔符 ""，否则高于它的分隔符都失败时 chunk_size
    # 无法成为硬上限（切分器只会按给定的分隔符断句）。这里显式补上并回显给用户，
    # 保证"分块最大长度"是可预期的；merge 策略下允许突破该上限（用户显式选择）。
    if "" not in items:
        items.append("")
    return items


def _normalize_level(source: dict, defaults: dict, label: str = "") -> dict:
    """把单层级（通用分块，或父块/子块）的输入归一化成完整参数。

    label 只用于给报错加"父块/子块"前缀，让用户知道是哪一层参数不对。
    """
    try:
        separators = _normalize_separators(
            source.get("separators", source.get("delimiters")), defaults["separators"]
        )
    except ChunkConfigError as exc:
        if label:
            raise ChunkConfigError(f"{label}{exc}")
        raise
    return {
        "separators": separators,
        "chunk_size": _as_int(source.get("chunk_size"), defaults["chunk_size"], "chunk_size"),
        "chunk_overlap": _as_int(source.get("chunk_overlap"), defaults["chunk_overlap"], "chunk_overlap"),
        "length_mode": str(source.get("length_mode") or defaults["length_mode"]).strip().lower(),
        "keep_separator": _as_bool(source.get("keep_separator"), defaults["keep_separator"]),
        "filter_whitespace": _as_bool(source.get("filter_whitespace"), defaults["filter_whitespace"]),
        "short_chunk_policy": str(
            source.get("short_chunk_policy") or defaults["short_chunk_policy"]
        ).strip().lower(),
        "min_length": _as_int(source.get("min_length"), defaults["min_length"], "min_length"),
    }


_LEVEL_KEYS = ("separators", "length_mode", "keep_separator",
               "filter_whitespace", "short_chunk_policy", "min_length")


def normalize_config(raw: dict | None = None, *, chunk_size=None, chunk_overlap=None) -> dict:
    """把（可能来自表单 / JSON / 旧版扁平字段的）输入归一化成完整配置字典。

    只做类型归一与默认值填充，不做范围校验；范围校验见 validate_config。
    父子分块时返回 {"strategy", "parent", "child", ...顶层镜像子块参数}，
    顶层镜像让知识库扁平列（chunk_size / chunk_overlap）与旧前端展示继续可用。
    """
    source = dict(raw or {})
    # 兼容旧版扁平字段：chunk_size / chunk_overlap 优先以显式参数为准
    if chunk_size is not None:
        source["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        source["chunk_overlap"] = chunk_overlap

    strategy = str(source.get("strategy") or DEFAULT_CONFIG["strategy"]).strip().lower()
    if strategy == "parent_child":
        parent_source = dict(source.get("parent") or {})
        child_source = dict(source.get("child") or {})
        # 顶层扁平字段作为子块参数兜底（老前端只传 chunk_size/chunk_overlap 时仍可用）
        for key in _LEVEL_KEYS:
            if key not in child_source and source.get(key) is not None:
                child_source[key] = source[key]
        if "chunk_size" not in child_source and source.get("chunk_size") is not None:
            child_source["chunk_size"] = source["chunk_size"]
        if "chunk_overlap" not in child_source and source.get("chunk_overlap") is not None:
            child_source["chunk_overlap"] = source["chunk_overlap"]

        parent = _normalize_level(parent_source, DEFAULT_PARENT_CONFIG, "父块")
        child = _normalize_level(child_source, DEFAULT_CHILD_CONFIG, "子块")
        return {
            "strategy": "parent_child",
            "parent": parent,
            "child": child,
            **{key: child[key] for key in _LEVEL_KEYS},
            "chunk_size": child["chunk_size"],
            "chunk_overlap": child["chunk_overlap"],
        }

    config = _normalize_level(source, DEFAULT_CONFIG)
    return {"strategy": strategy, **config}


def _validate_level(level: dict, label: str) -> None:
    """校验单层级参数（label 用于给出"父块/子块/分块"前缀的明确提示）。"""
    if not level["separators"]:
        raise ChunkConfigError(f"{label}至少需要配置一个分隔符")
    if len(level["separators"]) > MAX_SEPARATORS:
        raise ChunkConfigError(
            f"{label}分隔符最多 {MAX_SEPARATORS} 个，当前 {len(level['separators'])} 个"
        )
    if level["chunk_size"] < 1:
        raise ChunkConfigError(f"{label}最大长度必须大于 0，当前值: {level['chunk_size']}")
    if level["chunk_size"] > MAX_CHUNK_SIZE:
        raise ChunkConfigError(
            f"{label}最大长度不能超过 {MAX_CHUNK_SIZE}，当前值: {level['chunk_size']}"
        )
    if level["chunk_overlap"] < 0:
        raise ChunkConfigError(f"{label}重叠长度不能为负数，当前值: {level['chunk_overlap']}")
    if level["chunk_overlap"] >= level["chunk_size"]:
        raise ChunkConfigError(
            f"{label}重叠长度（{level['chunk_overlap']}）必须小于{label}最大长度（{level['chunk_size']}）"
        )
    if level["length_mode"] not in LENGTH_MODES:
        raise ChunkConfigError(
            f"{label}长度计算方式仅支持 {' / '.join(LENGTH_MODES)}，当前值: {level['length_mode']!r}"
        )
    if level["short_chunk_policy"] not in SHORT_CHUNK_POLICIES:
        raise ChunkConfigError(
            f"{label}过短分块处理方式仅支持 {' / '.join(SHORT_CHUNK_POLICIES)}，"
            f"当前值: {level['short_chunk_policy']!r}"
        )
    if level["min_length"] < 0:
        raise ChunkConfigError(f"{label}最短分块长度不能为负数，当前值: {level['min_length']}")
    if level["min_length"] > level["chunk_size"]:
        raise ChunkConfigError(
            f"{label}最短分块长度（{level['min_length']}）不能大于{label}最大长度（{level['chunk_size']}）"
        )


def validate_config(raw: dict | None = None, *, chunk_size=None, chunk_overlap=None) -> dict:
    """归一化 + 校验，返回可直接使用的配置；非法时抛 ChunkConfigError（带中文原因）。"""
    config = normalize_config(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if config["strategy"] not in STRATEGIES:
        raise ChunkConfigError(
            f"分块策略仅支持 {' / '.join(STRATEGIES)}，当前值: {config['strategy']!r}"
        )

    if config["strategy"] == "parent_child":
        _validate_level(config["parent"], "父块")
        _validate_level(config["child"], "子块")
        if config["parent"]["chunk_size"] <= config["child"]["chunk_size"]:
            raise ChunkConfigError(
                f"父块最大长度（{config['parent']['chunk_size']}）必须大于"
                f"子块最大长度（{config['child']['chunk_size']}）"
            )
        return config

    _validate_level(config, "分块")
    return config


def canonical_json(config: dict) -> str:
    """稳定序列化：键排序 + 无多余空格，便于入库与比对。"""
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(config: dict) -> str:
    """配置指纹：用于判断"文档实际索引配置"是否与当前配置一致（第 2 期使用）。"""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


# ── 从库里取配置 ────────────────────────────────────────

def config_from_kb(kb_row: dict | None) -> dict:
    """读取知识库的分块配置；旧库没有 chunk_config 时回落到扁平字段。"""
    raw: dict = {}
    if kb_row:
        stored = kb_row.get("chunk_config") or ""
        if stored:
            try:
                raw = json.loads(stored)
            except json.JSONDecodeError:
                logger.warning("知识库 %s 的 chunk_config 不是合法 JSON，已回落到默认配置", kb_row.get("kb_id"))
                raw = {}
        if kb_row.get("chunk_strategy"):
            raw.setdefault("strategy", kb_row["chunk_strategy"])
        if "chunk_size" not in raw and kb_row.get("chunk_size") is not None:
            raw["chunk_size"] = kb_row["chunk_size"]
        if "chunk_overlap" not in raw and kb_row.get("chunk_overlap") is not None:
            raw["chunk_overlap"] = kb_row["chunk_overlap"]
    return validate_config(raw)


def config_from_upload_task(task_row: dict | None, kb_row: dict | None = None) -> dict:
    """确定 Worker 实际使用的配置：任务内冻结快照 > 知识库当前配置 > 任务扁平字段。"""
    stored = (task_row or {}).get("chunk_config") or ""
    if stored:
        try:
            return validate_config(json.loads(stored))
        except (json.JSONDecodeError, ChunkConfigError) as exc:
            logger.warning(
                "任务 %s 的 chunk_config 不可用（%s），回落到知识库当前配置",
                (task_row or {}).get("task_id"), exc,
            )
    if kb_row is not None:
        return config_from_kb(kb_row)
    return validate_config({}, chunk_size=(task_row or {}).get("chunk_size"),
                           chunk_overlap=(task_row or {}).get("chunk_overlap"))


# ── 分块 ────────────────────────────────────────────────

def _apply_short_chunk_policy(pieces: list[str], config: dict, length_function) -> list[str]:
    """过短分块处理：merge 并入相邻块 / drop 丢弃 / keep 保留。

    merge 语义：优先并入**前一块**；若过短块是首块，则与**后一块**合并；
    若整篇都短且无相邻块，则原样保留。合并后长度允许超过 chunk_size
    （这是用户显式选择 merge 的预期结果，预览中会展示真实长度）。
    """
    if config["min_length"] <= 0 or config["short_chunk_policy"] == "keep":
        return pieces

    policy = config["short_chunk_policy"]
    merged: list[str] = []
    pending = ""

    for piece in pieces:
        if pending:
            piece = pending + piece
            pending = ""
        if length_function(piece) < config["min_length"]:
            if policy == "drop":
                continue
            # merge
            if merged:
                merged[-1] = merged[-1] + piece
            else:
                pending = piece
            continue
        merged.append(piece)

    if pending:
        merged.append(pending)
    return merged


def _split_pieces(text: str, level_config: dict) -> list[str]:
    """按某一层级配置切分文本：分隔符优先级回退 → 空白过滤 → 过短分块策略。"""
    length_function = build_length_function(level_config["length_mode"])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=level_config["chunk_size"],
        chunk_overlap=level_config["chunk_overlap"],
        length_function=length_function,
        separators=level_config["separators"],
        keep_separator=level_config["keep_separator"],
        # 过滤开关直接交给切分器：True 时先在切分内部 strip（与项目历史行为一致），
        # False 时保留原文空白与分隔符（"不过滤空白分块"的真实语义）。
        strip_whitespace=level_config["filter_whitespace"],
    )
    pieces = [piece for piece in splitter.split_text(text) if piece is not None]
    if level_config["filter_whitespace"]:
        pieces = [piece for piece in pieces if piece and piece.strip()]  # 丢弃纯空白块
    else:
        pieces = [piece for piece in pieces if piece]  # 仅去掉空串，保留空白
    return _apply_short_chunk_policy(pieces, level_config, length_function)


def _make_item(index: int, content: str, length_mode: str) -> dict:
    """构造带长度信息的块字典（length 按长度口径，chars 保留给旧预览接口）。"""
    length_function = build_length_function(length_mode)
    return {
        "chunk_index": index,
        "content": content,
        "length": length_function(content),
        "chars": len(content),
    }


def build_chunks(text: str, config: dict | None = None, **overrides) -> dict:
    """按配置切分文本，返回父子层级结构（通用分块时 parents 为空）。

    返回:
        {
          "strategy": "general" | "parent_child",
          "parents":  [{"chunk_index", "content", "length", "chars"}, ...],
          "children": [{"chunk_index", "content", "length", "chars",
                        "parent_index": int（通用分块为 -1）, "sense_index": int}, ...],
        }

    `children` 是**可检索叶子块**：入库时只有 children 会被向量化，
    父块只落 SQLite 用于回查组装完整上下文。
    """
    cfg = validate_config(config, **overrides)
    if not text or not text.strip():
        return {"strategy": cfg["strategy"], "parents": [], "children": []}

    if cfg["strategy"] == "parent_child":
        parents: list[dict] = []
        children: list[dict] = []
        child_index = 0
        for parent_index, parent_content in enumerate(_split_pieces(text, cfg["parent"])):
            parents.append(_make_item(parent_index, parent_content, cfg["parent"]["length_mode"]))
            for sense_index, child_content in enumerate(_split_pieces(parent_content, cfg["child"])):
                item = _make_item(child_index, child_content, cfg["child"]["length_mode"])
                item.update({"parent_index": parent_index, "sense_index": sense_index})
                children.append(item)
                child_index += 1
        return {"strategy": "parent_child", "parents": parents, "children": children}

    children = []
    for index, content in enumerate(_split_pieces(text, cfg)):
        item = _make_item(index, content, cfg["length_mode"])
        item.update({"parent_index": -1, "sense_index": index})
        children.append(item)
    return {"strategy": "general", "parents": [], "children": children}


def group_parents(built: dict) -> list[dict]:
    """把 build_chunks 的结果整理成父子层级视图（预览展示用，纯数据变换）。"""
    if not built.get("parents"):
        return []
    bucket: dict[int, list[dict]] = {}
    for child in built["children"]:
        bucket.setdefault(child.get("parent_index", -1), []).append(child)
    return [{**parent, "children": bucket.get(parent["chunk_index"], [])}
            for parent in built["parents"]]


def chunk_text(text: str, config: dict | None = None, **overrides) -> list[dict]:
    """把纯文本切分为**可检索叶子块**列表（通用分块=全部块；父子分块=子块）。

    返回元素：chunk_index / content / length / chars，
    父子分块时另含 parent_index（无父块为 -1）与 sense_index（父块内序号）。
    config 与 overrides 会一起归一化校验；overrides 用于扁平参数（chunk_size 等）。
    """
    return build_chunks(text, config, **overrides)["children"]


def chunk_document(file_path: str, config: dict | None = None, **overrides) -> list[dict]:
    """解析文档并切分，返回可检索叶子块列表（预览与入库共用的唯一入口）。"""
    return build_document_chunks(file_path, config, **overrides)["children"]


def build_document_chunks(file_path: str, config: dict | None = None, **overrides) -> dict:
    """解析文档并返回父子层级结构（Worker 入库使用）。"""
    from text_parser import load_text

    text = load_text(file_path)
    return build_chunks(text, config, **overrides)
