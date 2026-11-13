"""蒸馏：把 L0 原始消息增量转成 L1 摘要 + L2 记忆 + L3 状态更新。

三条铁律：
1. 增量：只处理水位之后的未蒸馏消息，随时可中断、可重跑。
2. 有据：每条记忆必须带 quote（原话）和 src_msg（消息 id），否则丢弃。
3. 可回滚：抽取只新增，不改写；事实变了用 superseded_by 指向新条目。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import daytime, llm, mood
from .store import Msg, Store
from .util import clip, dumps, now

KINDS = ("fact", "preference", "event", "relation", "promise", "boundary", "trait", "self")

DISTILL_SYS = """你是长期陪伴对话的记忆抽取器。任务：从新增对话中抽出「以后还需要记得」的东西。
只输出 JSON，结构：

{
  "memories": [
    {"kind": "fact|preference|event|relation|promise|boundary|trait|self",
     "subject": "对方|自己|第三方名字",
     "text": "一句话，第三人称陈述，具体、可检索；kind=self 时用第一人称",
     "why": "她自己的理由（可空），必须是原话片段",
     "quote": "原话逐字摘录，必须能在输入里找到",
     "src_msg": [消息id...],
     "salience": 1.0-3.0,
     "confidence": 0.0-1.0}
  ],
  "state": {
     "bond_delta": -0.05 ~ +0.05,
     "mood": "对方当前情绪，一个词；不变则 null",
     "nick": "对方明确说「叫我X」「你可以叫我X」时填 X；其它情况一律 null（别猜、别从他说的别的话里挑词）",
     "open_loops_add": [{"text": "未聊完/答应了但没做的话题"}],
     "open_loops_close": ["已解决话题的原文片段"]
  },
  "summary": "这段对话的压缩摘要，保留具体细节（人名/地点/数字/情绪），不超过 200 字"
}

规则：
- 只抽长期有效的信息。寒暄、一次性语气词、你已经知道的重复内容，全部不要。
- **附和与自谦不算信息。**「呃，是要做手术，但是呃，我觉得应该没有挨一刀那么狠」
  「是的，毕竟这个手术的目的是确诊疾病」「我记性不太好」这类句子是在应声或自贬：
  开头一个语气词/应答词，去掉它以后没有新的人、事、时间。
  判据：把语气词去掉，还剩下什么具体的东西？什么都不剩就不要。
  她自己那些"我不太会""我记性不好"的自谦，除非当场给了理由，否则不要记 ——
  记下来她会拿它当自我描述，越说越像。
- kind 判定：可验证的客观信息=fact；喜好/厌恶/习惯=preference；发生过的具体事=event；
  对彼此关系的定义、称呼、边界=relation/boundary；答应过的事=promise；性格底色=trait。
- **self=她自己说出口的立场、判断、决定、好恶**（assistant 那一侧的话）。subject 固定写「自己」，
  text 用第一人称，例如「我不喜欢被叫大肥鱼」「我决定以后不追问他工位的事」。
  这类条目要尽量给 why —— 从她自己的话里摘理由，摘不到就留空，**不要替她想理由**。
  它不是给对方建档案，是让她自己的反应前后连得上。
  **只收带评价或意志的句子。** 这几类不算 self，宁可漏掉：
    · 陈述事实（「我记着是今天」「我在」）
    · 对别人的推测（「我认定对方心里有点什么」「他大概是在忙」）
    · 转述自己刚说过的话
- why：只有 self / promise / boundary 需要；必须是原话里的片段。
- salience：普通信息 1.0；对方主动自我披露或有明显情绪 2.0；涉及创伤/禁忌/重要承诺 3.0。
- quote 必须是输入中的原话片段，不许改写、不许编造。找不到原文就不输出该条。
- **她自己的行动也要记**：kind=event、subject=自己、text 用第一人称，写清"做了什么、对象是什么、
  看到了什么具体内容"，例如「我翻了 D:\\cold-vault，里面有五月跨文化交流的六个章节课件，
  复习资料从 v1.0 改到 v5.0」。只收**动了外部对象或读到具体内容**的行为；
  「我说了一句」「我回了她」这种空动作不要记。
- **区分「发生了什么」和「谁说了什么」。** 她自己说的判断可能是错的（她看不见自己上一轮说过的话）。
  summary 必须写事实；如果她的某句话与本次对话里出现的工具调用/对方纠正相矛盾，
  就写成「她答 X，但实际是 Y」，不要直接把她的说法当事实记下来。
  例：她答「没翻过 cold-vault」，而同一轮里她确实读过它 —— summary 要写她读过，
  并注明她那句回答与事实不符。
- **open_loops_add 绝对不要写元陈述。** 以下都是「她自己对记忆系统的困惑」，
  不是真正未完成的话题 —— 记下去会被 anchor 每轮重复注入，她就会一直在那上面转：
    · 自己翻不到 / 记忆丢了 / 压缩丢内容 / 抽取没记住 / 水位停住
    · "我不知道" / "我记不得" / "我不太会" / "我说话人机味" / "我今天为什么钝"
    · 关于「怎么叫我」「怎么夸我」「怎么调人设」这种流程上的抱怨
  判据：如果这句话是**关于记忆/抽取/装配/蒸馏/人设调教本身的**，就不要写进 open_loops。
  真正的未完成话题必须是**对方还没讲完的事**（手术结果、任务进展、行程安排……）。
- 不确定的不要写。宁可少，不要错。"""


@dataclass
class DistillResult:
    memories: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    summary: str = ""
    source: str = "llm"


_REL_DAY = [
    ("大前天", -3), ("大后天", 3),
    ("前天", -2), ("后天", 2),
    ("昨儿", -1), ("昨天", -1), ("昨晚", -1), ("昨夜", -1), ("昨早", -1),
    ("今儿", 0), ("今天", 0), ("今晚", 0), ("今夜", 0), ("今早", 0),
    ("明儿", 1), ("明天", 1), ("明晚", 1), ("明早", 1),
]


def _md(ts: int, delta_days: int = 0) -> str:
    """ts 往前/后挪 delta_days 个**日历日**，返回「10月27日」。

    不能用 ts ± 86400：那串算的是秒，跨时区/夏令时/闰秒会漂；
    而且这里要的本来就是"日历上的那一天"，按 date 加减才对。
    """
    import datetime as _dt
    import time as _time
    d = _dt.date(*_time.localtime(ts)[:3]) + _dt.timedelta(days=delta_days)
    return f"{d.month}月{d.day}日"


def normalize_time_words(text: str, ts: int | None) -> str:
    """把记忆里的「昨天/今天/明天」换成绝对日期。

    为什么必须换：记忆是长期存储，而相对时间词一天就过期。
    「今天定了个闹钟」明天读起来就变成"昨天定的"，后天就成了三天前——
    这正是她把自己今天做的事说成"昨天"的根源。

    顺序有讲究：表里长词在前（"大前天"先于"前天"、"昨儿/昨晚"也单列），
    否则"大前天"会被替换成"大10月27日"。

    quote 保留原话不动，只改派生的记忆正文。
    """
    if not text or not ts:
        return text
    out = text
    for word, delta in _REL_DAY:
        if word in out:
            out = out.replace(word, _md(ts, delta))
    return out


_ABS_DATE_RE = re.compile(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]")


def event_ts(text: str, ts: int | None, *, back_days: int = 10) -> int:
    """事件发生时间：正文里写了**唯一**一个绝对日期，就用那天，而不是"收到消息的时间"。

    为什么：一批消息在 10 月 29 日蒸馏，里面讲的是"10 月 27 日去看皮肤"。
    t_event 若记 10 月 29 日，注入时就变成「今天 10月27日去看皮肤」——
    标签和内容自己打架，她照着标签说话就错。标签必须认正文。

    为什么要求"唯一"：正文里出现两个以上不同日期，说明这句话是**跨天叙述或事后
    澄清**（"那是 10 月 27 日的事了，10 月 28 日我就去了医院"），说话的时候是
    ts 那天。那种条目留在 ts 才对 —— 否则"她 29 号说的这句澄清"会被搬回 27 号，
    连"她当时是这么理解的"这层信息一起丢掉。

    只接受 [ts - back_days, ts] 内的日期：正文里的日期可能是去年的事，
    也可能是在约下个月，那些不该覆盖消息时间。
    """
    if not text or not ts:
        return ts or 0
    import datetime as _dt
    import time as _time
    lt = _time.localtime(ts)
    days: dict[int, int] = {}
    for m in _ABS_DATE_RE.finditer(text):
        try:
            year = int(m.group(1)) if m.group(1) else lt.tm_year
            y, mo, d = year, int(m.group(2)), int(m.group(3))
            cand = int(_time.mktime((y, mo, d, 12, 0, 0, 0, 0, -1)))
            key = _dt.date(y, mo, d).toordinal()
        except (ValueError, OverflowError):
            continue
        if ts - back_days * 86400 <= cand <= ts:
            days.setdefault(key, cand)
    if len(days) != 1:
        return ts
    return next(iter(days.values()))


# ---------------------------------------------------------------- 规则兜底
# 设计原则：这一层不是"凑合能用的抽取器"，而是漏斗的第一级——
# 用极廉价的规则判断「这句值不值得花 token 让 LLM 抽」。
# 有 API key 时它决定抽谁；没 key 时它自己出结果。两边都不许乱记。
_EMO = re.compile(r"(撑不住|难受|难受死|崩溃|哭|失眠|焦虑|害怕|难过|委屈|孤独|烦死|累死|心疼|压力大)")
_EVENT = re.compile(r"(住院|手术|面试|考试|答辩|搬家|出差|离职|辞职|毕业|结婚|分手|离婚|"
                    r"生日|旅游|体检|开庭|入职|报到|回国|出国|回家|过年)")
_BOUNDARY = re.compile(r"(别(叫|喊|提|说|问)|不要(叫|喊|提|说)|不许|不准|讨厌(别人|有人)?"
                       r"(叫|喊|问)|别提|不能提|别再)")
_PROMISE = re.compile(r"(说好|答应|下次|回头我|以后我|我会给你|欠你|我给你看|约好)")
_PREF = re.compile(r"(最|很|超|特别|从小|一直|完全不|压根不|从来不)"
                   r"(就|都|也)?(喜欢|爱|讨厌|受不了|不吃|不喝|不喜欢|怕|迷|上瘾)")
_FACT = re.compile(r"(我(叫|是|在.{0,8}(工作|上班|上学|住)|老家|今年|生日)|"
                   r"(养|捡|收养)了.{0,6}(猫|狗|仓鼠|兔|鸟)|取名叫|取个名|"
                   r"我在.{0,10}(公司|学校|医院|银行)|我有.{0,4}(孩子|儿子|女儿|弟弟|妹妹|哥哥|姐姐))")
_QUESTION = re.compile(r"[？?]\s*$|吗\s*$|呢\s*$|怎么样|要不要|是不是|好不好|行不行")
_HEDGE = re.compile(r"^(我)?(在想|在想想要|考虑)(要不要|是不是)")

# 应声碎片：开口就是语气词或"是的/对呀"，说完一堆语气词其实没提供新事实。
# 「呃，是要做手术，但是呃，我觉得应该没有挨一刀那么狠」——她说这话是在附和，
# 不是在告诉你一件事。这类句子被当成 [event] 存下来，就成了她"记得一堆废话、
# 却说不出你到底怎么了"的来源（2026-10-29 她自己点出来的）。
_ACK_OPEN = re.compile(
    r"^(?:[呃嗯啊哦唉诶欸呀哈噢喔]+[，,。.、…\s]*"          # 纯语气词起头
    r"|(?:是的|是|对啊|对呀|对|没错|确实|好的|好|行|可以|当然)"
    r"(?=[，,。.、；;：:！!？?\s]|$))"                     # 应答词必须是独立的一小节
)
_FILLER_CHARS = set("呃嗯啊哦唉诶欸呀哈噢喔嘛吧呢啦咯")
_PUNCT = set("，,。.、；;：:！!？? 　\"'“”‘’…—-")
# 只要有时间线索就不是纯附和："是的，我明天下午三点去" 是信息。
_TIME_HINT = re.compile(r"(\d|点|号|明天|后天|下周|下个月|这个月|上午|下午|晚上|早上|中午|"
                        r"月|日|年|周末|春节|放假)")

# 元陈述：她自己对记忆系统的困惑，不是真正未完成的话题。
# 记下去会被 anchor 每轮重复注入，她就会一直在那上面转。
_META_PATTERN = re.compile(
    r"(翻不到|记忆丢|压缩丢|抽取没|水位停|没进记|没记住|没想起|想不起来|记不得|"
    r"人机味|怎么叫|怎么夸|怎么调|调人设|怎么说|别扭|钝|记性不好|不太会|"
    r"我不知道|我记不|我不记|我不懂|我不太|我说话|我回复|我答|我发|我翻|我看|"
    r"记忆系统|记忆压缩|抽取规则|蒸馏|装配|锚点|召回|检索|水位|积压|"
    r"token|模型|参数|温度|prompt|人设|记忆库|记忆文件|上下文|字数|长度)"
)


def _is_meta_statement(text: str) -> bool:
    """是不是元陈述：关于记忆系统本身的困惑，不是真正未完成的话题。"""
    if not text:
        return False
    return bool(_META_PATTERN.search(text))


def _is_ack_filler(text: str) -> bool:
    """是不是"嗯我在听"式的应声 —— 是就整条不记。"""
    if not _ACK_OPEN.match(text):
        return False
    if _TIME_HINT.search(text):
        return False
    solid = [c for c in text if c not in _FILLER_CHARS and c not in _PUNCT]
    return len(solid) < 26


# 纯情绪感叹句（"我好难过啊"）不是事件：什么事都没发生。
# 情绪归 L3 状态（mood），不进 L2 —— 记成 event 她会当成"发生过的事"，
# 正是她抱怨"把自己说过的话记错"的来源之一（2026-10-30 20:59「我好难过啊」
# 被规则漏斗存成 [event] 实字 4，2026-10-31 修复）。
_EMO_MOOD = {
    "撑不住": "撑不住", "难受": "难过", "难受死": "难过", "崩溃": "崩溃",
    "哭": "难过", "失眠": "失眠", "焦虑": "焦虑", "害怕": "害怕",
    "难过": "难过", "委屈": "委屈", "孤独": "孤独", "烦死": "烦躁",
    "累死": "累", "心疼": "心疼", "压力大": "压力大",
}


def _emo_mood(text: str) -> str | None:
    """情绪词 → 一个 mood 词；带否定/安慰语气的（"我不难过"）不算。"""
    m = _EMO.search(text)
    if not m:
        return None
    pre = text[:m.start()]
    # 否定/安慰："我不难过""没那么难过""别难过" —— 情绪词前面紧挨否定词就不算真情绪
    if re.search(r"(?:不|没|别|不再|别再)", pre[-4:]):
        return None
    return _EMO_MOOD.get(m.group(0), m.group(0))


# 她自己的立场（[self] 类）：第一人称 + 判断/决定/好恶/自我认知。
# 只在她那一侧找 —— 这是「她的记忆」，不是给对方建档案。
# 注意收得紧：**陈述**（"我记着是今天"）和**推测**（"我认定对方……"）都不算立场，
# 混进来会把 self 槽变成流水账。
_SELF_STANCE = re.compile(
    r"我不(?:喜欢|太喜欢|想|愿意|乐意|在乎|介意|怕|信)"
    r"|我(?:喜欢|讨厌|受不了|宁愿|决定|打算|答应|保证|不改|认定这)"
    r"|我不是(?:鱼|人|程序|模型)"
    r"|我(?:觉得|认为)(?=.{4,})"          # 「我觉得」后面得接着说理由，光两个字不算
)
# 理由片段：从她自己的话里摘，摘不到就留空，绝不替她想
_WHY = re.compile(r"(?:因为|所以|毕竟|反正|不然|要不|主要是)[^。！？!?，,]{2,40}")


def _heuristic_self(msgs: list[Msg]) -> list[dict]:
    """从她自己说的话里抽 [self]：立场 + 理由。

    这类记忆不给她建"人设档案"，是让她的反应前后连得上：
    问「你为什么这么想」时，她手里有自己当初的理由，而不是每轮重新编一个。
    """
    out: list[dict] = []
    for m in msgs:
        if m.role != "assistant":
            continue
        for seg in re.split(r"[。！？!?\n]", m.text or ""):
            seg = seg.strip()
            if len(seg) < 3 or _QUESTION.search(seg) or not _SELF_STANCE.search(seg):
                continue
            why = _WHY.search(seg)
            out.append({
                "kind": "self",
                "subject": "自己",
                "text": clip(seg, 80),
                "why": clip(why.group(0), 60) if why else None,
                "quote": clip(seg, 80),
                "src_msg": [m.id],
                "salience": 1.6 if _EMO.search(seg) else 1.2,
                "confidence": 0.5,
            })
            break          # 一条消息最多一条：她那句话是主线，别拆成碎片
    return out


def _heuristic_distill(msgs: list[Msg]) -> DistillResult:
    """无 API key 时的降级抽取：规则命中 + 原话即依据。宁可少，不要错。"""
    res = DistillResult(source="heuristic")
    for m in msgs:
        if m.role != "user":
            continue
        t = m.text.strip()
        if len(t) < 5 or _QUESTION.search(t) or _HEDGE.search(t):
            continue
        # 应声碎片不记 —— 但边界和承诺例外：那两类哪怕是短语也必须留下
        if (_is_ack_filler(t) and not _BOUNDARY.search(t) and not _PROMISE.search(t)):
            continue

        kind, sal = None, 1.2
        if _BOUNDARY.search(t):
            kind, sal = "boundary", 2.8        # 边界最不能忘：忘了就伤人
        elif _PROMISE.search(t):
            kind, sal = "promise", 2.1
        elif _EVENT.search(t):
            kind, sal = "event", 2.3
        elif _PREF.search(t):
            kind, sal = "preference", 1.7
        elif _FACT.search(t):
            kind, sal = "fact", 1.5
        emo = _emo_mood(t)
        if emo:
            # 情绪归 L3 状态（mood），不单独造 L2 事件 —— "我好难过啊" 这种
            # 纯感叹句什么事都没发生，记成 event 就是她抱怨的"把自己说过的话记错"。
            # 情绪只给已命中的真事件加权。
            res.state["mood"] = emo
            if kind:
                sal = max(sal, 2.6)

        if not kind:
            continue

        subject = "对方"
        third = re.search(r"我(妈|爸|哥|姐|弟|妹|朋友|室友|同事|对象|男朋友|女朋友)", t)
        if third:
            subject = "对方的" + third.group(1)

        res.memories.append({
            "kind": kind,
            "subject": subject,
            "text": clip(t, 60),
            "quote": clip(t, 80),
            "src_msg": [m.id],
            "salience": sal,
            "confidence": 0.62,
        })
    if msgs:
        users = [m.text for m in msgs if m.role == "user"]
        picked = [x["text"] for x in res.memories]
        if picked:
            body = "；".join(clip(p, 26) for p in picked[:3])
        elif users:
            body = clip(max(users, key=len), 40)
        else:
            body = "（无实质内容）"
        # 摘要层只承担"近期话题流"，不带统计前缀；事实由 L2 结构化记忆承载
        res.summary = body
    return res


# ---------------------------------------------------------------- LLM 抽取
def _llm_distill(msgs: list[Msg], persona_brief: str = "",
                 existing: list[str] | None = None) -> DistillResult:
    lines = "\n".join(f"[{m.id}] {m.role}: {m.text}" for m in msgs)
    # 把库里已有的条目喂进去。不喂的后果：它每轮都在"重新发现"同一件事，
    # 于是库里攒下七条"对方在给助手做升级"的不同说法，谁也没比谁更对。
    known = ""
    if existing:
        known = ("记忆库里已经有这些：\n"
                 + "\n".join(f"- {t}" for t in existing[:40])
                 + "\n\n**同一件事不要再抽一遍。** 只有这次对话给出了更具体的新细节时，"
                   "才抽那一条新增的部分（例如旧的是「对方要去做手术」，新的是「手术在 10 月 28 日下午，"
                   "目的是活检确诊」）。宁可少写，也不要换个说法重复。\n\n")
    user = (f"已有设定摘要：{persona_brief or '（无）'}\n\n{known}"
            f"新增对话：\n{lines}\n\n输出 JSON。")
    raw = llm.chat(
        [{"role": "system", "content": DISTILL_SYS}, {"role": "user", "content": user}],
        temperature=0.2, max_tokens=1600, json_mode=True,
        model="deepseek-chat",
    )
    data = _parse_json(raw)
    res = DistillResult()
    valid_ids = {m.id for m in msgs}
    text_of = {m.id: m.text for m in msgs}
    for item in data.get("memories", []) or []:
        kind = str(item.get("kind", "")).strip()
        text = str(item.get("text", "")).strip()
        quote = str(item.get("quote", "")).strip()
        src = [i for i in (item.get("src_msg") or []) if i in valid_ids]
        if kind not in KINDS or not text:
            continue
        # 硬校验：quote 必须真的出现在某条源消息里，否则视为编造，丢弃
        if not quote or not any(quote[:12] in text_of.get(i, "") for i in src):
            continue
        res.memories.append({
            "kind": kind,
            "subject": (item.get("subject") or "对方")[:20],
            "text": clip(text, 120),
            "why": clip(str(item.get("why") or ""), 80) or None,
            "quote": clip(quote, 160),
            "src_msg": src,
            "salience": _clampf(item.get("salience"), 1.0, 3.0, 1.0),
            "confidence": _clampf(item.get("confidence"), 0.0, 1.0, 0.8),
        })
    res.state = data.get("state") or {}
    res.summary = str(data.get("summary") or "")
    return res


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        return json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except ValueError:
            return {}


def _clampf(v, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def salience_floor(importance: int) -> float:
    """她自己给这条消息打过分 → 蒸馏出来的条目**至少**这么显著（2026-11-12，M2）。

    公式照 HANDOVER：`1 + importance × 0.4`，4 分 → 2.6，5 分 → 3.0（顶格）。
    3 分（默认分 = 标了但没表态）**不抬**，0/1/2 同样不抬 —— 自评低不是"不许记"，
    只是不替它加分。**只抬不压**：LLM 给的更高就留着它的。

    注意它抬的是"显著度"，不是"去留"：退场仍由 `recall.maintain` 按年龄和
    `hit_count` 判（HANDOVER §「必须守的那条」：自评 × 行为证据）。
    """
    try:
        imp = int(importance or 0)
    except (TypeError, ValueError):
        return 0.0
    if imp <= 3:
        return 0.0
    return 1.0 + imp * 0.4


# ---------------------------------------------------------------- 主入口
def capture_one(store: Store, peer: str, msg: Msg, *, min_salience: float = 1.0) -> int:
    """即时捕获：单条消息过一遍规则漏斗，高置信度的立刻入库。

    为什么需要它：批量蒸馏每 4 轮跑一次，意味着"你刚说完我妈住院了"这句话
    在接下来的 3 轮里检索不到 —— 体验上就是没记住。规则漏斗是微秒级的，可以每条都跑。

    LLM 批量抽取仍然保留：它负责需要理解力的抽取、摘要、状态更新，
    并且会以更高 confidence 取代这里写下的规则版（见 distill_once 的 supersede 逻辑）。

    她那一侧（assistant）也过漏斗，但只抽 [self]：她说过什么立场、为什么。
    这一侧不动关系状态 —— 说一句"我觉得"不该涨亲密度。

    纯情绪句（"我好难过啊"）不进 L2 —— 什么事都没发生，记成 event 就是她抱怨的
    "把自己说过的话记错"；但情绪即时写进 L3 状态（mood），不丢信号。
    """
    res = None
    if msg.role == "user":
        res = _heuristic_distill([msg])
        items = res.memories
    elif msg.role == "assistant":
        items = _heuristic_self([msg])
    else:
        return 0

    n = 0
    existing = [m.text for m in store.all_memories(peer)]
    floor = salience_floor(getattr(msg, "importance", 0))   # 她标过分的（M2）
    for item in items:
        sal = max(float(item["salience"]), floor)
        if sal < min_salience:
            continue
        if any(_similar(item["text"], e) for e in existing):
            continue
        body = normalize_time_words(item["text"], msg.ts)
        store.add_memory(
            peer, item["kind"], body,
            subject=item.get("subject"),
            why=normalize_time_words(item.get("why") or "", msg.ts) or None,
            quote=item.get("quote"), src_msg=[msg.id],
            salience=sal,
            confidence=0.55 if msg.role == "user" else 0.5,
            t_event=event_ts(body, msg.ts),
        )
        existing.append(item["text"])
        n += 1
    if msg.role == "user":
        state = {"memory_texts": [i["text"] for i in items]}
        if n:
            state["bond_delta"] = 0.004
        if res.state.get("mood"):
            state["mood"] = res.state["mood"]
        if n or state.get("mood"):
            _apply_state(store, peer, state)
    return n


def distill_once(store: Store, peer: str, *, max_msgs: int = 60,
                 use_llm: bool | None = None, persona_brief: str = "") -> DistillResult:
    pending = store.pending(peer, limit=max_msgs)
    if not pending:
        return DistillResult()
    use_llm = llm.available() if use_llm is None else use_llm
    known = [m.text for m in store.all_memories(peer)]
    if use_llm:
        try:
            res = _llm_distill(pending, persona_brief, existing=known)
        except llm.LLMError:
            res = _heuristic_distill(pending)
            res.memories += _heuristic_self(pending)
    else:
        res = _heuristic_distill(pending)
        # 她那一侧的自我立场，规则层也能抽：不依赖 LLM
        res.memories += _heuristic_self(pending)

    # ---- 主语校验（2026-11-12）----
    # 启发式那条路有 `if m.role != "assistant"` 挡着，**LLM 蒸馏那条没有事后校验**：
    # prompt 里明明给了 `[id] role: text`，模型照样会把他的话标成 kind=self。
    # 实测 11 条、累计被召回 98 次（「我以前一直以情商低自居」成了她的自我评价、
    # 「舍友这种答应得痛快」成了她的舍友）。不补这一道，每聊一次就多一条。
    _role_of = {m.id: m.role for m in pending}
    _kept, _dropped = [], []
    for item in res.memories:
        if item.get("kind") == "self":
            srcs = [int(i) for i in (item.get("src_msg") or [])]
            roles = [_role_of.get(s) for s in srcs if _role_of.get(s)]
            if roles and not any(r == "assistant" for r in roles):
                _dropped.append(item.get("text") or "")
                continue
        _kept.append(item)
    if _dropped:
        import logging
        logging.getLogger(__name__).info(
            "[memcore] 丢掉 %d 条主语错位的 [self]（来源全是 user 的话）", len(_dropped))
        res.memories = _kept

    # 落库：同一件事只留一条。若已存在低置信度的规则版，被高置信度的 LLM 版取代
    ref_ts = pending[-1].ts
    # 她自己对这些消息打过分（M2）：salience 取 max，不覆盖 LLM 的更高值。
    imp_map = store.msg_importance(
        [i for item in res.memories for i in (item.get("src_msg") or [])])
    for item in res.memories:
        floor = max((salience_floor(imp_map.get(int(i), 0))
                     for i in (item.get("src_msg") or [])), default=0.0)
        item["salience"] = max(float(item["salience"]), floor)
        body = normalize_time_words(item["text"], ref_ts)
        t_ev = event_ts(body, ref_ts)
        dup = next((m for m in store.all_memories(peer)
                    if _similar(item["text"], m.text)), None)
        if dup is not None:
            if item["confidence"] > dup.confidence + 0.15:
                new_id = store.add_memory(
                    peer, item["kind"], body,
                    subject=item.get("subject"),
                    why=normalize_time_words(item.get("why") or "", ref_ts) or None,
                    quote=item.get("quote"),
                    src_msg=item.get("src_msg") or [], salience=item["salience"],
                    confidence=item["confidence"], t_event=t_ev,
                )
                store.supersede(dup.id, new_id)      # 旧版留档，不再参与检索
            continue
        store.add_memory(
            peer, item["kind"], body,
            subject=item.get("subject"),
            why=normalize_time_words(item.get("why") or "", ref_ts) or None,
            quote=item.get("quote"),
            src_msg=item.get("src_msg") or [], salience=item["salience"],
            confidence=item["confidence"], t_event=t_ev,
        )

    # 状态：LLM 没给 bond_delta 时，用本轮记忆的显著度折算亲密度增长
    # 只算关于对方的条目 —— 她自己说一句"我觉得"不该算亲密
    theirs = [m for m in res.memories if (m.get("subject") or "对方") != "自己"]
    if not res.state.get("bond_delta"):
        if theirs:
            gain = 0.006 + sum(max(0.0, m["salience"] - 1.0) for m in theirs) * 0.006
            res.state["bond_delta"] = min(0.05, gain)
        else:
            res.state["bond_delta"] = 0.002      # 纯寒暄只微涨：熟不熟看聊过什么，不看聊了多少
    res.state["memory_texts"] = [m["text"] for m in res.memories]
    _apply_state(store, peer, res.state)
    lo, hi = pending[0].id, pending[-1].id
    if res.summary:
        store.add_summary(peer, 1, lo, hi, pending[0].ts, pending[-1].ts, res.summary)
        _compact(store, peer, level=1, keep=4)
    store.set_watermark(peer, hi)
    return res


def _similar(a: str, b: str) -> bool:
    """这两条是不是在说同一件事。

    短句单独一档：几个字的句子分母小，jaccard 天然虚高，用长句的阈值会放进来一堆
    「同一件事换个说法」。这里宁可漏记一条碎片，也不要在库里堆两份同样的意思 ——
    她拿到两份会在两个说法之间反复，看起来就是记性差。
    """
    from .recall import _jaccard
    j = _jaccard(a, b)
    limit = 0.55 if min(len(a), len(b)) <= 30 else 0.72
    return j > limit


def _apply_state(store: Store, peer: str, state: dict) -> None:
    if not state:
        return
    st = store.state(peer)
    fields: dict = {}
    bond = st["bond"] + _clampf(state.get("bond_delta"), -0.08, 0.08, 0.0)
    fields["bond"] = round(max(0.0, min(1.0, bond)), 4)
    if state.get("mood"):
        fields["mood"] = str(state["mood"])[:16]
        fields["mood_ts"] = now()
    if state.get("nick"):
        fields["nick"] = str(state["nick"])[:20]
    # 边界类记忆里抽称呼（"别叫我X，叫我Y" → 取 Y，不要把被否定的 X 当称呼）
    for mem in state.get("memory_texts") or []:
        s = str(mem)
        banned = set(re.findall(r"别叫我([\u4e00-\u9fffA-Za-z0-9]{1,6})", s))
        banned |= set(re.findall(r"(?:不要|不准|不许)叫(?:我)?([\u4e00-\u9fffA-Za-z0-9]{1,6})", s))
        for cand in re.findall(r"叫我([\u4e00-\u9fffA-Za-z0-9]{1,6})", s):
            cand = re.sub(r"(就行|就好|吧|啊|呀|啦|可以)$", "", cand)
            if cand and cand not in banned and not fields.get("nick"):
                fields["nick"] = cand[:12]

    loops = list(st["open_loops"])
    for add in state.get("open_loops_add") or []:
        txt = str(add.get("text") if isinstance(add, dict) else add).strip()
        # 过滤元陈述：她自己对记忆系统的困惑，不是真正未完成的话题。
        # 记下去会被 anchor 每轮重复注入，她就会一直在那上面转。
        if _is_meta_statement(txt):
            continue
        if txt and not any(txt[:8] in (l.get("text") or "") for l in loops):
            loops.append({"id": int(now() * 1000) % 10**9, "text": clip(txt, 100),
                          "ts": now(), "closed": False})
    for frag in state.get("open_loops_close") or []:
        frag = str(frag)
        for l in loops:
            if not l["closed"] and frag[:8] and frag[:8] in l["text"]:
                l["closed"] = True
                l["closed_ts"] = now()
    fields["open_loops"] = loops[-40:]     # 只留最近 40 条，避免无限膨胀
    store.save_state(peer, **fields)


def _compact(store: Store, peer: str, *, level: int, keep: int,
             max_level: int = 3) -> None:
    """分层摘要，层数封顶。

    L1 留最近 4 段 → 溢出进 L2；L2 留 3 条 → 进 L3；L3 留 2 条 → 更老的直接丢。

    为什么顶层可以丢：远期对话的"事实"已经结构化进 L2 记忆条目，原文还在 L0，
    摘要只承担"近期语感与话题流"，不需要无限保留。留着它只会挤占预算。
    """
    rows = store.summaries(peer, level)          # msg_to DESC：rows[0] 最新
    if len(rows) <= keep:
        return
    old = sorted(rows[-2:], key=lambda r: r["msg_to"])   # 两条里更老在前
    a, b = old[0], old[1]
    text = _merge_text([a["text"], b["text"]],
                       must_keep=_must_keep(store, peer, a, b))
    store.db.execute("DELETE FROM summary WHERE id IN (?,?)", (a["id"], b["id"]))
    store.db.commit()
    if level < max_level:
        store.add_summary(peer, level + 1, a["msg_from"], b["msg_to"],
                          a["ts_from"], b["ts_to"], text)
        _compact(store, peer, level=level + 1, keep=max(2, keep - 1), max_level=max_level)
    else:
        # 顶层就地滚动，不再新生层次
        store.add_summary(peer, level, a["msg_from"], b["msg_to"],
                          a["ts_from"], b["ts_to"], text)
        _compact(store, peer, level=level, keep=keep, max_level=max_level)


def _must_keep(store: Store, peer: str, a, b, *,
               min_level: int = 4, max_items: int = 6) -> list[str]:
    """这一段区间里她标过 `min_level` 分以上的**他的**原话（M2）。

    为什么要留原句：层数越高压缩越狠，一条「他说 10 月 28 日下午做活检」
    会被压成一行，跟「今天吃了什么」同等待遇。她标过分 = 她自己认定要紧，
    压掉了等于把她做的那个判断丢了。

    只取 user 那一侧：她自己说的话已经由 L2 的 [self] 条目承载，
    摘要里再来一遍会让她把摘要读成"我的日记"。
    """
    try:
        msgs = store.high_importance_msgs(peer, a["msg_from"], b["msg_to"],
                                          min_level=min_level)
    except Exception:                    # noqa: BLE001
        return []
    out: list[str] = []
    for m in msgs[:max_items]:
        t = " ".join(str(getattr(m, "text", "") or "").split())
        if t:
            out.append(clip(t, 120))
    return out


def _merge_text(texts: list[str], limit: int = 240, max_parts: int = 6,
                must_keep: list[str] | None = None) -> str:
    """抽取式合并：去重、优先留长句（信息密度高）、片段数封顶。

    `must_keep`（2026-11-12，M2）：她标过 >=4 分的原话，**原样放在最前面，
    不参与去重排序、不被截断、不压成一行**。其余片段照旧按长度抽取。
    总长上限给 must_keep 让路（否则"保留"会被 `clip` 在末尾剪掉）。

    有 LLM 时应该改成"重新摘要"而不是拼接；这里保证无 key 时也不会越合越长。
    """
    seen: set[str] = set()
    kept_must: list[str] = []
    for t in (must_keep or []):
        s = " ".join(str(t or "").split())
        if s and s not in seen:
            seen.add(s)
            kept_must.append(s)

    parts: list[str] = []
    for t in texts:
        for seg in re.split(r"[/；]", t or ""):
            seg = seg.strip()
            if not seg or seg in seen:
                continue
            seen.add(seg)
            parts.append(seg)
    parts.sort(key=len, reverse=True)
    body = clip("；".join(parts[:max_parts]), limit)
    if not kept_must:
        return body
    head = "；".join(kept_must)
    # 给 must_keep 单独留预算：它的长度不占那 240 字，否则它会把自己挤掉。
    return clip(head + ("；" + body if body else ""), limit + len(head))


def state_line(st: dict) -> str:
    """把 L3 状态压成一行给 prompt 用。

    注意两组 mood 是两件事，别合成一句：
      * 「上次情绪：X」= **他的**情绪（从他的话里抽的一个词）；
      * 「这会儿的劲头：……」。= **她自己**的状态量（mood.hint，纯事实、无数字、无断言）。
    并列着给，让她自己读——写成"你心情不好"她会照着演。

    第三句是**一天的时间形状**（daytime.line，纯事实：几点、哪一档）。
    同样不写成断言：不写"你在睡觉""你该休息"，只给时刻和档位，她自己决定怎么用。
    """
    parts = [f"亲密度{bond_word(st['bond'])}"]
    if st.get("nick"):
        parts.append(f"称呼：{st['nick']}")
    if st.get("mood"):
        parts.append(f"上次情绪：{st['mood']}")
    h = mood.hint(st)
    if h:
        parts.append(h)
    parts.append(daytime.line())
    loops = [l for l in st.get("open_loops", []) if not l.get("closed")]
    if loops:
        parts.append("未聊完：" + "；".join(l["text"] for l in loops[-3:]))
    return "；".join(parts)


def bond_word(b: float) -> str:
    if b < 0.2:
        return "刚认识"
    if b < 0.4:
        return "熟悉"
    if b < 0.6:
        return "亲近"
    if b < 0.8:
        return "很亲近"
    return "非常重要的人"


def dump_result(res: DistillResult) -> str:
    return dumps({"memories": res.memories, "state": res.state,
                  "summary": res.summary, "source": res.source})
