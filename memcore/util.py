"""零依赖工具：中文分词、token 估算、时间、JSON。"""
from __future__ import annotations

import json
import re
import time
from typing import Iterable

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_WORD = re.compile(r"[a-zA-Z]+|[0-9]+")

# 高频虚词 bigram，检索时降权用（不是严格停用词）
STOP_BIGRAMS = {
    "的时", "时候", "我们", "你们", "他们", "这个", "那个", "什么", "怎么",
    "就是", "然后", "因为", "所以", "但是", "如果", "可以", "一个", "现在",
    "今天", "自己", "觉得", "还是", "这样", "那样", "不是", "没有", "知道",
}


def tokenize(text: str) -> list[str]:
    """中文切字符 bigram + 英文/数字按词。用于 FTS 与词面相似度。"""
    if not text:
        return []
    out: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    out.extend(w.lower() for w in _WORD.findall(text))
    return [t for t in out if t]


# 高频单字：单字通道里必须压掉，否则"的了我你"会把所有条目拉到同分
STOP_UNI = set("的了着我你他她它是在不有和就都也这那一个上到说要去会好呢啊呀吧嗯很吗们"
               "又还再没什为以对给让被把从而且但所如才可能过下天来时")


def tokenize_uni(text: str) -> list[str]:
    """单字通道：解决中文 bigram 对单字关键词（"猫""香菜"）召不到的硬伤。"""
    if not text:
        return []
    out = [c for run in _CJK_RUN.findall(text) for c in run if c not in STOP_UNI]
    out.extend(w.lower() for w in _WORD.findall(text))
    return out


def tokenize_query(text: str) -> list[str]:
    """查询侧：去掉停用 bigram，保留有信息量的。全被滤掉则退回原始切分。"""
    toks = tokenize(text)
    keep = [t for t in toks if t not in STOP_BIGRAMS]
    return keep or toks


def est_tokens(text: str) -> int:
    """粗略 token 数：CJK 约 1 字 1 token，英文约 1 词 1.3 token。"""
    if not text:
        return 0
    cjk = len(_CJK_RUN.findall(text) and re.findall(f"[{_CJK}]", text) or [])
    words = len(_WORD.findall(text))
    other = len(re.sub(rf"[{_CJK}]", "", _WORD.sub(" ", text)).split())
    return int(cjk + words * 1.3 + other * 0.5) + 1


def now() -> int:
    return int(time.time())


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def loads(raw: str | None, default=None):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def join_limited(items: Iterable[str], budget_tokens: int) -> list[str]:
    """按 token 预算贪心装填，返回装得下的条目。"""
    used, kept = 0, []
    for it in items:
        n = est_tokens(it)
        if used + n > budget_tokens:
            break
        used += n
        kept.append(it)
    return kept
