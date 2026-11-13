"""出口：她说完之后，把不像人说的话拿掉，再切成几条（2026-11-05）。

三件事，一个模块，都是纯函数（第三件 2026-11-08 加）：

  1. ``filter_out`` —— 丢掉「她正常说话绝不会用」的规则句（首先/其次/综上/作为AI/
     语气/口吻/回他的…）。**只删整句，不动词**（2026-11-08 加了一层 A 级结构改写，见 ``_rewrite``）。
  2. ``split``      —— 按她那一轮的状态把一整段切成 1–4 条（人发消息的节奏）。

三条不能破的规矩：

  * **宽词一律不收**。`应该` / `需要` / `可以` / `可能` 这种日常词单独出现绝不删句 ——
    材料里踩过一次：「你应该早点睡」被整句删掉，那是她会说的话。所以这里全部用
    **组合触发**：同一句里同时出现两个"汇报腔"标记才算。
  * **滤空退回原文**。整条被她自己的规则滤没了，就照原文发出去，只打日志。
    静默发一条空话，比留着一点人机味糟得多。
  * **不立法**。这里不评价她说得好不好，只管"这句她本人不会这么说"。

跟 ``style.py`` 的分工：style 管**这一次说话的状态**（进注入），express 管
**她说出来的东西**（出发送与 L0）。一个在门口，一个在出口。
"""
from __future__ import annotations

import re

# 一条回复最多拆几条。人体感：再多就不像"分条"，像刷屏。
MAX_PARTS = 4
# 无标点时的硬切长度（材料 §5.9：无标点 48 字硬切）
HARD_CUT = 48
# 短于这个字数的段不单独成条，并回前一条（材料：≥4 字才切）
MIN_PART = 4

# 句子切分：保留标点，换行也算句末。
_SENT_RE = re.compile(r"[^。！？!?…~\n]*[。！？!?…~\n]|[^。！？!?…~\n]+$")
_BLANK_RE = re.compile(r"\n{2,}")


def _norm(s: str) -> str:
    """判断用：去掉空白与标点，只留字。"""
    return re.sub(r"[\s，,。．.！!？?、；;：:…~—－\-（）()\[\]【】「」『』\"'“”‘’]", "", s or "")


# —— A 级结构改写（2026-11-08）——
# 「不是A，是B」是最典型的 AI 痕迹：她先把话折成两截，前半截否一个对方根本没说的
# 意思，后半截才给正文。人聊天不这么起手（样例：「不是从零起，是找上一句说到哪儿了」）。
# 只处理**句首**这一种形状，改成只留后件；中间形式（X不是A，而是B）先不动 ——
# 拆开以后经常不成句，那是 B 级，要灰度。
_NOT_X_BUT_Y = re.compile(r"^不是(.{1,28}?)[，,]\s*(?:而是|是)\s*(.+)$")

# 句末意义附加：她讲完事又补一句「这说明…」。那一句通常不带新信息，只带顾问腔。
_MEANING_TAIL = ("这说明", "这意味着", "这本身就", "某种程度上", "这背后")

# 后件能独立成句的开头（代词/指示/时间/存在动词）。不在表里的（多数是动词开头，
# 如「找…」「是…」）不改 —— 删掉前件会留下没主语的半句。
_STANDALONE_HEAD = set("那这我你他她它们其有就在还反正现实当等所每到从对把被给让")


def _rewrite(sentence: str) -> str | None:
    """A 级结构改写：只改形状，不换词。

    Args:
        sentence: 一句原文（含标点）。

    Returns:
        ``None`` = 不动；空字符串 = 整句删掉；其他 = 替换成这一句。
    """
    s = (sentence or "").strip()
    if not s:
        return None

    # 1) 「不是A，是B」→ 「B」
    #    只在 B 自己能站住时才改：以这些字开头说明主语/对象已经在了，
    #    否则删掉 A 会留下「找上一句说到哪儿了」这种没主语的半句（实测踩过）。
    m = _NOT_X_BUT_Y.match(s)
    if m:
        rest = m.group(2).strip()
        if len(_norm(rest)) >= 3 and rest[0] in _STANDALONE_HEAD:
            return rest
        return None

    # 2) 句末意义附加：整句删（它不承载事实，只承载"我在分析"）
    if any(s.startswith(w) for w in _MEANING_TAIL):
        return ""

    return None


def _stiff(sentence: str) -> bool:
    """这一句是不是"她不会这么说"的规则句。

    判据全部是**组合**：单个宽词永远不够。

    Args:
        sentence: 一句原文（含标点）。

    Returns:
        True 表示可以整句丢掉。
    """
    s = _norm(sentence)
    if not s:
        return False

    # 1) 清单式：首先…其次／再者／然后／最后
    if "首先" in s and any(w in s for w in ("其次", "再者", "第二", "然后", "最后")):
        return True
    # 「首先，」这种单摆一个开头也在内 —— 她不会用书面序号说话
    if re.search(r"首先[，,：:]", sentence or ""):
        return True
    # 「其次，我分析一下。」单独一句也要收：清单式不一定两句挨在一起
    if re.match(r"^(其次|再者|第三|第四|最后一点|另外一点)", s):
        return True

    # 2) 总结腔
    if any(w in s for w in ("综上所述", "综上", "总而言之", "总的来说", "总结一下", "总结起来")):
        return True
    if "以下是" in s and any(w in s for w in ("分析", "总结", "建议", "回答", "要点")):
        return True

    # 3) 自报身份
    if "作为" in s and any(w in s for w in ("助手", "AI", "ai", "语言模型", "人工智能", "程序", "机器人")):
        return True

    # 4) 元话语：说"怎么回他""针对你的问题"
    if any(w in s for w in ("回他的", "回他时", "针对你的问题", "关于你的问题", "你的问题可以分为")):
        return True

    # 5) 语气／口吻类的自我说明（"我会用更轻松的语气"）
    if any(w in s for w in ("语气", "口吻")) and any(
        w in s for w in ("我会", "我要", "尽量", "更加", "更温柔", "更轻松", "自然一些", "随意一些")
    ):
        return True

    # 6) 客套收尾
    if "希望你" in s and any(w in s for w in ("满意", "理解", "喜欢", "有帮助")):
        return True

    # 7) "需要注意的是" 这种提示腔
    if "需要注意的是" in s or "值得一提的是" in s:
        return True

    return False


def filter_out(text: str) -> tuple[str, list[str]]:
    """丢掉规则句，返回 (清洗后文本, 被丢掉的句子)。

    Args:
        text: 她的原话。

    Returns:
        ``(kept, dropped)``。``kept`` 为空表示整条都被滤光了 —— 调用方必须
        退回原文（见模块说明），不要静默发空。
    """
    raw = text or ""
    if not raw.strip():
        return raw, []

    kept: list[str] = []
    dropped: list[str] = []
    for sentence in _SENT_RE.findall(raw):
        if not sentence.strip():
            continue
        fixed = _rewrite(sentence)
        if fixed == "":
            dropped.append(sentence.strip())      # 整句不要（意义附加）
            continue
        if fixed is not None:
            dropped.append(sentence.strip())      # 记原句，日志里能核对改了哪句
            sentence = fixed
        if _stiff(sentence):
            dropped.append(sentence.strip())
        else:
            kept.append(sentence)

    if not dropped:
        return raw, []

    out = "".join(kept).strip()
    if not out:
        # 兜底：整条被滤空 → 原文照发（只报，不改）
        return raw, dropped
    return out, dropped


def _cut(block: str) -> list[str]:
    """把一段切成若干句；没有标点的超长段硬切。"""
    out: list[str] = []
    for sentence in _SENT_RE.findall(block):
        if not sentence.strip():
            continue
        while len(sentence) > HARD_CUT:
            out.append(sentence[:HARD_CUT])
            sentence = sentence[HARD_CUT:]
        if sentence.strip():
            out.append(sentence)
    return out


def split(text: str, *, parts: int = MAX_PARTS) -> list[str]:
    """把一整段切成 1–4 条（人发消息的节奏）。永远至少返回原文一条。

    规则（按外部材料的实测阈值）：
      * 空行优先断（她本来就用空行分意思）
      * 标点断句，但 < 4 字的碎段并回前一条
      * 没有标点的超长段按 48 字硬切
      * 超过 ``parts`` 条：后面的并成一条（用单换行连接，前端不会再多拆气泡）

    Args:
        text: 她的原话（已过滤）。
        parts: 最多几条。

    Returns:
        非空的字符串列表。
    """
    t = (text or "").strip()
    if not t:
        return []

    blocks = [b.strip() for b in _BLANK_RE.split(t) if b.strip()] or [t]
    pieces: list[str] = []
    for b in blocks:
        pieces.extend(_cut(b))

    merged: list[str] = []
    for seg in pieces:
        seg = seg.strip()
        if not seg:
            continue
        if merged and len(_norm(seg)) < MIN_PART:
            merged[-1] = merged[-1] + seg       # 碎段并回前一条
        else:
            merged.append(seg)

    if not merged:
        return [t]
    if len(merged) > max(1, parts):
        head = merged[: max(1, parts) - 1]
        tail = "\n".join(merged[max(1, parts) - 1 :])
        merged = head + [tail]
    return merged


def segments(text: str, *, parts: int = MAX_PARTS) -> list[str]:
    """filter_out + split 的合并入口（出口一次做完）。

    整条被滤空时退回原文再切 —— 绝不返回空列表。
    """
    kept, _dropped = filter_out(text)
    out = split(kept, parts=parts)
    return out or [(text or "").strip()]
