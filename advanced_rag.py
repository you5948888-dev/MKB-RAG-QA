"""
Advanced RAG 前置处理（Advanced RAG 迭代 · 第 3 期）。

三个互不相同的 LLM 能力，不要混为一谈：
  - **问题改写**：把依赖上下文的追问（"那住宿费呢？"）补全成可独立理解的问题；
  - **多查询扩展**：从**用户问题**生成多种表达，扩大召回；
  - **HyDE**：从**用户问题**生成一段假设答案，用假设答案去检索（答案本身不返回给用户）。
  （对比：**索引问题**是从**文档内容**生成问题，属于入库侧，见 Ch8。）

统一约定：
  - 每个能力独立开关，任一失败都只记录降级并回退，**绝不阻断主检索流程**；
  - 原问题**始终参与检索**，因此改写即使跑偏也不会丢召回；
  - 所有 LLM 输出都会被清洗与限长（控制调用成本与上下文膨胀）。
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import strategy as strategy_module
import langsmith_tracing

logger = logging.getLogger("rag.advanced")

REWRITE_SYSTEM = (
    "你是检索系统中的问题改写助手。请把用户当前问题改写成一句可以独立理解的问题。"
    "要求：保留用户的原始意图与关键实体，只补全上下文省略的信息（如指代、省略的主语）；"
    "不要回答问题，不要添加解释，不要改变问题的范围。只输出改写后的一句话。"
)

MULTI_QUERY_SYSTEM = (
    "你是检索系统中的查询扩展助手。请围绕用户问题生成若干语义相近但表达不同的检索问题。"
    "要求：每条一行，不要编号，不要解释，不要回答问题；"
    "覆盖不同角度（同义表达、专业术语、可能的关键词组合），彼此不要重复。"
)

HYDE_SYSTEM = (
    "你是检索系统中的假设答案生成助手。请根据用户问题写一段**假设性**的回答片段，"
    "用于向量检索（不会展示给用户）。要求：只输出正文，不要标题、不要分点编号、不要解释；"
    "允许不完全准确，但必须使用与问题同领域的术语和表述。"
)


@dataclass
class QueryPlan:
    """一次高级检索的问题计划：原问题 + 各能力产出的检索问题。"""

    original: str
    rewritten: str = ""
    multi_queries: list[str] = field(default_factory=list)
    hyde_answer: str = ""
    degradations: list[dict] = field(default_factory=list)

    def search_queries(self) -> list[tuple[str, str]]:
        """返回 [(path_id, query_text), ...]；path_id 会写进结果的 hit_paths。"""
        items: list[tuple[str, str]] = [("original", self.original)]
        if self.rewritten and self.rewritten != self.original:
            items.append(("rewritten", self.rewritten))
        for index, query in enumerate(self.multi_queries, start=1):
            if query and query != self.original:
                items.append((f"multi_query_{index}", query))
        if self.hyde_answer:
            items.append(("hyde", self.hyde_answer))
        return items

    def debug(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "multi_queries": list(self.multi_queries),
            "hyde_answer": self.hyde_answer,
            "degradations": list(self.degradations),
            "search_paths": [path for path, _ in self.search_queries()],
        }


# ── LLM 调用与清洗 ──────────────────────────────────────

def _call_llm(prompt: str, system: str, ability: str = "") -> str:
    """调用 LLM（通过模块属性访问，便于测试替换 llm_client）。

    ability 只用于 LangSmith 打标签（rewrite / multi_query / hyde），
    未开启追踪时对行为没有任何影响。
    """
    import embedding
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = langsmith_tracing.traced_llm(embedding.llm_client, mode="advanced", ability=ability)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])
    content = getattr(response, "content", "") or ""
    return str(content).strip()


def _clean_single_line(text: str, max_chars: int = 300) -> str:
    """清洗单行输出：去引号/编号/多余空白，并限长。"""
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    line = re.sub(r"^[\s\-*0-9]+[.、):：]?\s*", "", line)
    line = line.strip().strip("\"'“”‘’")
    return line[:max_chars]


def _clean_multi_lines(text: str, limit: int, question: str) -> list[str]:
    """清洗多行输出：去编号、去空行、去重、剔除与原问题重复的项。"""
    results: list[str] = []
    for raw in text.splitlines():
        line = _clean_single_line(raw)
        if not line or len(line) < 4:
            continue
        if line == question:
            continue
        if line not in results:
            results.append(line)
        if len(results) >= limit:
            break
    return results


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """按句末标点截断，避免把句子截半（HyDE 限长用）。"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for mark in ("。", "！", "？", "\n", ".", "!", "?"):
        index = window.rfind(mark)
        if index >= max_chars * 0.5:
            return window[:index + 1]
    return window


# ── 三个能力 ────────────────────────────────────────────

def rewrite_question(question: str, history: list[dict]) -> tuple[str, str | None]:
    """问题改写。返回 (改写问题, 失败原因)；失败时改写问题为空字符串。"""
    if not history:
        return "", None  # 无对话历史：直接用原问题（不算失败）
    history_text = "\n".join(
        f"{'用户' if msg.get('role') == 'user' else '助手'}: {msg.get('content', '')[:200]}"
        for msg in history[-6:]
    )
    prompt = f"对话历史：\n{history_text}\n\n用户当前问题：{question}\n\n改写后的问题："
    try:
        rewritten = _clean_single_line(_call_llm(prompt, REWRITE_SYSTEM, ability="rewrite"))
    except Exception as exc:  # 降级：使用原问题
        logger.warning("问题改写失败，回退原问题: %s", exc)
        return "", f"{type(exc).__name__}: {exc}"
    if not rewritten:
        return "", "改写结果为空"
    return rewritten, None


def expand_queries(question: str, count: int) -> tuple[list[str], str | None]:
    """多查询扩展。返回 (扩展问题列表, 失败原因)。"""
    if count <= 0:
        return [], None
    prompt = f"用户问题：{question}\n\n请生成 {count} 条不同的检索问题："
    try:
        queries = _clean_multi_lines(_call_llm(prompt, MULTI_QUERY_SYSTEM, ability="multi_query"),
                                    count, question)
    except Exception as exc:  # 降级：只用原问题
        logger.warning("多查询扩展失败，仅使用原问题: %s", exc)
        return [], f"{type(exc).__name__}: {exc}"
    if not queries:
        return [], "未生成有效的扩展问题"
    return queries, None


def generate_hypothetical_answer(question: str, max_chars: int = 500) -> tuple[str, str | None]:
    """生成 HyDE 假设答案（仅用于检索）。返回 (假设答案, 失败原因)。"""
    prompt = f"用户问题：{question}\n\n请写一段假设性回答片段（不超过 {max_chars} 字）："
    try:
        answer = _call_llm(prompt, HYDE_SYSTEM, ability="hyde")
    except Exception as exc:  # 降级：跳过 HyDE
        logger.warning("HyDE 生成失败，跳过该路径: %s", exc)
        return "", f"{type(exc).__name__}: {exc}"
    if not answer:
        return "", "假设答案为空"
    return _truncate_at_sentence(answer, max_chars), None


# ── 计划构建 ────────────────────────────────────────────

def build_plan(question: str, *, history: list[dict] | None = None,
               config: dict | None = None) -> QueryPlan:
    """按策略开关构建查询计划（并行调用各能力，失败只降级不抛出）。"""
    cfg = strategy_module.normalize_strategy(config)
    plan = QueryPlan(original=question)
    history = history or []

    jobs = []
    if cfg["enable_rewrite"]:
        jobs.append(("rewrite", lambda: rewrite_question(question, history)))
    if cfg["enable_multi_query"]:
        jobs.append(("multi_query", lambda: expand_queries(question, cfg["multi_query_count"])))
    if cfg["enable_hyde"]:
        jobs.append(("hyde", lambda: generate_hypothetical_answer(question, cfg["hyde_max_chars"])))
    if not jobs:
        return plan

    def _run(name, func):
        try:
            return name, func()
        except Exception as exc:  # 兜底：任何未捕获异常都降级
            logger.warning("%s 阶段异常，已降级: %s", name, exc)
            reason = f"{type(exc).__name__}: {exc}"
            return name, (("", reason) if name == "rewrite" else ([], reason))

    if len(jobs) == 1:
        outcomes = [_run(*jobs[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            outcomes = list(pool.map(lambda job: _run(*job), jobs))

    for name, outcome in outcomes:
        if name == "rewrite":
            rewritten, error = outcome
            plan.rewritten = rewritten
            if error:
                plan.degradations.append({"stage": "rewrite", "reason": error,
                                          "fallback": "使用原问题检索"})
        elif name == "multi_query":
            queries, error = outcome
            plan.multi_queries = queries
            if error:
                plan.degradations.append({"stage": "multi_query", "reason": error,
                                          "fallback": "仅使用原问题检索"})
        elif name == "hyde":
            answer, error = outcome
            plan.hyde_answer = answer
            if error:
                plan.degradations.append({"stage": "hyde", "reason": error,
                                          "fallback": "跳过 HyDE 路径"})
    return plan
