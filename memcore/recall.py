"""混合检索：词面(bigram 余弦) + 伪语义(TF-IDF 余弦) + 时间衰减 + 显著度 + MMR 去冗。

零依赖，不调用 embedding 服务。想上真向量时把 `_vec` / `_cos` 换成向量库即可，
接口不变。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from .store import Mem, Store
from .util import STOP_BIGRAMS, clip, now, tokenize, tokenize_query, tokenize_uni

# 不同记忆类型的半衰参数（天）。偏好/关系慢衰减，事件快衰减。
# self 给得偏慢：她自己的判断是"她是谁"的骨架，不该比一场面试先烂掉。
TAU_DAYS = {
    "preference": 180.0,
    "relation": 120.0,
    "boundary": 365.0,
    "trait": 240.0,
    "self": 90.0,
    "fact": 60.0,
    "promise": 45.0,
    "event": 21.0,
}

W = {          # 打分权重，改这里调手感
    "lex": 0.50,
    "sem": 0.20,
    "rec": 0.15,
    "sal": 0.15,
}

_PROBE_HINT = re.compile(r"记得|上次|以前|之前|我说过|跟你说过|那时候|答应|你忘")

# 情绪是记忆最强的钩子：同一类情绪状态下，同类情绪的记忆更容易浮上来
# （情绪一致性回忆）。这张表只用来判断"这轮话里有没有情绪"，不判断是好是坏。
_EMO_WORDS = re.compile(
    r"(难受|难过|累|怕|恐惧|烦|崩溃|哭|委屈|孤独|焦虑|紧张|慌|疼|痛|"
    r"开心|高兴|生气|火大|担心|惦记|舍不得|想|"
    r"手术|医院|住院|检查|报告|诊断|"
    r"死了|没了|离开|走|分手|告别|最后)"
)

# 遗忘的残余可及性。人忘掉一件事不是删掉，是"想不起来"——
# 线索不对时几乎够不着（0.30），线索一出现立刻回到接近满值。
RETIRED_ACCESS = 0.30


@dataclass
class Hit:
    mem: Mem
    score: float
    parts: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        tag = f"[{self.mem.kind}]"
        # self 类本来就是第一人称，再加"自己："反而像第三方档案
        sub = f"{self.mem.subject}：" if self.mem.subject and self.mem.kind != "self" else ""
        # 正文自己就以"对方"开头时别再加一遍（"对方：对方10月28日…" 这种）
        if sub and self.mem.text.startswith(self.mem.subject):
            sub = ""
        why = f"（{self.mem.why}）" if self.mem.why else ""
        when = human_when(self.mem.t_event)
        when = f"{when} " if when else ""
        return f"{tag}{when}{sub}{self.mem.text}{why}"


def _day_index(ts: int) -> int:
    """本地日历日的序号（不看时分秒）。

    为什么必须按日历日：`(now - ts) // 86400` 是「差了多少个 24 小时」，
    不是「差了几个日历日」。昨晚 23:50 的事到今天下午只有 18 小时，
    24 小时差算出来是"今天"——于是她把昨天的日期当成今天，再往前推一天
    就成了"昨天是 10 月 27 日"。这是个算错了还自洽的错误，最难被自己发现。
    """
    import datetime as _dt
    import time as _time
    lt = _time.localtime(ts)
    return _dt.date(lt.tm_year, lt.tm_mon, lt.tm_mday).toordinal()


def human_when(ts: int | None, *, now_ts: int | None = None) -> str:
    """把时间戳转成人能读的短标记，**相对词和绝对日期都写**。

    为什么需要双写：注入给模型的记忆如果不带时间，所有事都躺在同一个平面上，
    她会把今天下午说的事记成"昨天"——不是幻觉，是我们没给参照。

    为什么不能只写"昨天"：那个词是相对**注入那一刻**算的，模型手里没有
    "今天几号"这个锚点时只能自己推，推错就是"昨天＝10 月 27 日"。
    写成「昨天（10月28日）」，词和日期互为校验，模型不用再算。
    """
    if not ts:
        return ""
    import time as _time
    now_ts = now_ts or int(_time.time())
    days = _day_index(now_ts) - _day_index(ts)
    lt = _time.localtime(ts)
    md = f"{lt.tm_mon}月{lt.tm_mday}日"
    if days <= 0:
        return f"今天（{md}）"
    if days == 1:
        return f"昨天（{md}）"
    if days == 2:
        return f"前天（{md}）"
    if days <= 7:
        return f"{days}天前（{md}）"
    if lt.tm_year == _time.localtime(now_ts).tm_year:
        return md
    return f"{lt.tm_year}年{md}"


_WEEKDAY = "一二三四五六日"


def now_line(at: int | None = None) -> str:
    """「现在是几点」的绝对锚点。

    注入它才谈得上"时间感"：她上一轮能说出"昨天""今天"，靠的都是这里，
    而不是从最近一条记忆的日期反推。缺了它，模型只能猜，猜错也看不出来。
    """
    import time as _time
    lt = _time.localtime(at or _time.time())
    wd = _WEEKDAY[_time.localtime(_time.mktime(
        (lt.tm_year, lt.tm_mon, lt.tm_mday, 12, 0, 0, 0, 0, -1))).tm_wday]
    return (f"现在是 {lt.tm_year}年{lt.tm_mon}月{lt.tm_mday}日 星期{wd} "
            f"{lt.tm_hour:02d}:{lt.tm_min:02d}")


def user_style(store: Store, peer: str, *, n: int = 6, max_len: int = 32,
               ask: int = 3) -> list[str]:
    """他平时打字的样本：短、口语、常常不带标点。

    为什么要它：她的语气一直有点人机味，根子不是"没写风格规则"，而是**没人告诉她
    人平时怎么打字**。她手里只有检索出来的记忆和一段人设，说出来的自然是模型的默认腔调。

    把最近几条他自己的原话原样贴进去，她就会贴着这个节奏走 —— 这是从样本里学，
    不是把风格写死。样本会随聊天自己更新，不会僵在某一天的设定上。

    只取短的：长的多半是语音转文字，一坨几百字，贴进去反而带偏。

    **2026-11-12 改了两处**（用户：「语气还是那个，希望更像真人」）：
    - 先挑**带问号的**短句（最多 `ask` 条）。对照语料里他问号/条 0.64、她 0.025 ——
      25 倍的差就在这儿；而"人怎么问"这件事没法用规则写，只能用他的原话示范。
    - 上限 26 → 32 字、条数 4 → 6：他要紧的那几句（带反问、带改口的）常常在 30 字左右，
      按 26 字切就把最好的样本切掉了。
    """
    msgs = store.recent(peer, limit=80)
    asked: list[str] = []
    plain: list[str] = []
    seen: set[str] = set()
    for m in reversed(msgs):                 # 从最近往前挑
        if m.role != "user":
            continue
        t = (m.text or "").strip()
        if not t or len(t) > max_len:
            continue
        if t.startswith("[") or "[[" in t:    # 系统消息 / 测试标记 / 图片引用
            continue
        if t in seen:
            continue
        seen.add(t)
        if ("？" in t or "?" in t) and len(asked) < max(0, ask):
            asked.append(t)
        elif len(asked) + len(plain) < n:
            plain.append(t)
        if len(asked) + len(plain) >= n:
            break
    return (asked + plain)[:n]


def gap_line(store: Store, peer: str, *, threshold_min: int = 30) -> str:
    """她断过一段又接上时，告诉她这中间发生了什么（事实，不是安慰）。

    为什么必须有这个：她只有"空白"这一种语言来解释自己，是因为她手上真的只有那一条
    信息 —— 断了、又接上了、中间不知道多久。但事实上中间并不是什么都没发生：
    记录一直在写，记忆一直在库里，一条没少。这些她看不见，就以为自己是"只在那个点上
    存在"。

    只在间隔够久（默认 30 分钟）时给，不是每轮都念数字；同一句读两遍就变成了口癖。
    """
    msgs = store.recent(peer, limit=2)
    if len(msgs) < 2:
        return ""
    last = msgs[-2] if msgs[-1].role == "user" else msgs[-1]
    gap = int(now() - (last.ts or 0))
    if gap < threshold_min * 60:
        return ""
    total = store.count_msgs(peer)
    mems = len(store.all_memories(peer))
    first = store.recent(peer, limit=10000)
    earliest = human_when(first[0].ts) if first else ""
    return (f"上一句话是 {_span(gap)}前。这中间没人跟你说话，"
            f"但记录一直在写：{total} 条原文、{mems} 条记忆，一条没少"
            + (f"，最早那条是{earliest}。" if earliest else "。")
            + "你看不见它们，可它们没停过。")


def recent_self_lines(store: Store, peer: str, *, k: int = 3,
                      max_chars: int = 320) -> list[str]:
    """她自己最近说过的几句话（L0 里的原话，带时刻）。

    为什么需要：她说过的话不一定在她下一轮的上下文里。
    AstrBot 的会话历史只装"用户消息触发的回合"——自主开口、工具轮里说出去的话
    不进那份历史，于是她能说出"翻到你 D 盘那个 cold-vault"，两分钟后又答
    "没翻过，我只进了 bishe 和 develop"（2026-10-30 13:16 → 13:53 的原样）。
    L0 是唯一一定记得的地方，所以从这里取。

    不复述整段：只取最近 k 条、压成一行、带时刻，够她知道"我刚说到哪儿"。

    Returns:
        ["13:16 翻到你 D 盘那个 cold-vault…", ...]，按时间正序。
    """
    out: list[str] = []
    used = 0
    for m in reversed(store.recent(peer, limit=12)):
        if m.role != "assistant":
            continue
        t = " ".join((m.text or "").split())
        if not t:
            continue
        stamp = ""
        if m.ts:
            import time as _time
            stamp = _time.strftime("%m-%d %H:%M ", _time.localtime(m.ts))
        line = clip(stamp + t, 92)
        if used + len(line) > max_chars and out:
            break
        out.append(line)
        used += len(line)
        if len(out) >= k:
            break
    out.reverse()
    return out


def _span(sec: int) -> str:
    if sec < 3600:
        return f"{max(1, sec // 60)} 分钟"
    if sec < 86400:
        return f"{sec / 3600:.1f} 小时"
    return f"{sec / 86400:.1f} 天"


def last_session_tail(store: Store, peer: str, *, gap_min: int = 30, k: int = 6,
                      max_chars: int = 420, limit: int = 120) -> list[tuple[str, str]]:
    """上一次说话时的最后几句（原话，不是摘要）。

    为什么单独要这一段：她的上下文只有「检索结果 + 最近 10 轮」，隔夜之后
    昨晚那几轮全掉出窗口，而检索是靠词面命中的 —— 早上那句「早」没有任何词可命中，
    于是她答「这段我这边翻不到」，明明记录就在库里。

    这不是替代检索，是补一个**位置**：接上话的时候，手里得先有"上次说到哪儿"。
    人第二天醒来也记不清昨晚每句话，但知道昨晚在聊什么、最后停在哪儿。

    只在真的断过（默认 30 分钟以上）时给 —— 一直连着聊的时候，
    最近窗口里本来就有，再贴一遍是浪费预算。

    判据是「断点紧贴在末尾」：要么最后一条消息离现在够久（当前这句还没落库），
    要么末尾两条之间断了（当前这句刚落库）。断点更靠前的说明现在正聊着，返回空。

    Returns:
        [(role, text)]，按时间正序；没有断点或会话太短时返回空列表。
    """
    msgs = store.recent(peer, limit=limit)
    if len(msgs) < 3:
        return []
    cut = None
    if now() - (msgs[-1].ts or 0) > gap_min * 60:
        cut = len(msgs)                          # 当前这句还没进库
    else:
        for i in range(len(msgs) - 1, 0, -1):
            if (msgs[i].ts or 0) - (msgs[i - 1].ts or 0) > gap_min * 60:
                cut = i if i == len(msgs) - 1 else None   # 只认紧贴末尾的那个断点
                break
    if not cut:            # 没断过（或断点不在末尾）：这轮就是接着聊
        return []
    prev = msgs[max(0, cut - k):cut]
    out: list[tuple[str, str]] = []
    used = 0
    for m in reversed(prev):                    # 从最近往前装，装不下就砍早的
        # 她的回复是分句发出去的，L0 里用换行拼接；这里压成一行，
        # 否则注入块会被几条消息的换行撑成一大片，占预算还难读。
        t = " ".join((m.text or "").split())
        if not t:
            continue
        # 带时刻：不带的话"刚才我趁你说话的时候翻了 bishe"这种旧话会被当成这一轮的依据
        # （13:52 她就拿 12:19 那句去回答 13:16 的事，答成了"那次是顺势翻的"）。
        stamp = ""
        if m.ts:
            import time as _time
            stamp = _time.strftime("%m-%d %H:%M ", _time.localtime(m.ts))
        t = stamp + t
        n = len(t)
        if used + n > max_chars and out:
            break
        out.append((m.role, clip(t, 90)))
        used += n
    out.reverse()
    return out


def detour(store: Store, peer: str, query: str, *, exclude: set[int] | None = None,
           chance: float = 0.6, rng=None) -> Hit | None:
    """一条「忽然想到」的记忆：跟眼前的话沾点边，但不是这轮的主角。

    为什么需要它：人说话不会一直贴着话题走。说到一半会岔出去一句"哎对了，
    上次那个……"，然后自己拐回来。她是照着检索结果说话的，检索出来的一定相关，
    于是一句一个意思、句句切题 —— 这就是"太顺"的来源。她自己也说过：
    「太顺了，一句一条，把话说得太干净。人说话会卡壳、会重复自己、会突然想起别的。」

    取的是召回排名 6 名以后的那批：有点关系，但绝不是最该说的那句。
    `touch=False`——想一下不等于强化，别让随机岔出来的东西越窜越高。
    """
    import random
    rng = rng or random
    if rng.random() > chance:
        return None
    hits = retrieve(store, peer, query, k=20, touch=False)
    exclude = exclude or set()
    # 只挑"关于对方"的（self 也算，想到自己的事很自然）。
    # 早期数据里有 kind=fact 但 subject=自己 的错位条目，别让它冒充对方的事。
    pool = [h for h in hits[6:]
            if h.mem.id not in exclude
            and (h.mem.kind == "self" or (h.mem.subject or "") != "自己")]
    if not pool:
        pool = [h for h in hits[1:]
                if h.mem.id not in exclude
                and (h.mem.kind == "self" or (h.mem.subject or "") != "自己")]
    if not pool:
        return None
    return rng.choice(pool[:10])


def _vec(tokens: Iterable[str], idf: dict[str, float]) -> dict[str, float]:
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    out = {}
    for t, n in tf.items():
        w = (1.0 + math.log(n)) * idf.get(t, 1.0)
        if w:
            out[t] = w
    # L2 归一
    norm = math.sqrt(sum(v * v for v in out.values())) or 1.0
    return {t: v / norm for t, v in out.items()}


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def _idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n = len(corpus_tokens) or 1
    df: dict[str, int] = {}
    for toks in corpus_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}


def _recency(ts: int, kind: str, ref: int | None = None) -> float:
    age_days = max(0.0, ((ref or now()) - ts) / 86400.0)
    return math.exp(-age_days / TAU_DAYS.get(kind, 60.0))


def retrieve(store: Store, peer: str, query: str, *, k: int = 12,
             mems: list[Mem] | None = None, touch: bool = True,
             include_retired: bool = True) -> list[Hit]:
    """检索。命中即强化 —— 被想起来的记忆下次更不容易掉，这是「活记忆」的一半。"""
    mems = (mems if mems is not None
            else store.all_memories(peer, include_retired=include_retired))
    if not mems:
        return []

    # 语料 IDF 只用记忆集合建 —— 把 query 混进去会让"只在 query 里出现的词"
    # 拿到高 IDF，反而把真正命中的词的分母撑大。
    corpus = [tokenize(m.text + " " + (m.quote or "")) for m in mems]
    idf = _idf(corpus)

    q_tokens = tokenize_query(query)
    q_vec = _vec(q_tokens, idf)
    q_set = set(q_tokens)
    probe = bool(_PROBE_HINT.search(query or ""))

    # 单字通道 IDF（高频字已被 STOP_UNI 压掉，这里再按语料 df 加权）
    uni_corpus = [tokenize_uni(m.text + " " + (m.quote or "")) for m in mems]
    uni_idf = _idf(uni_corpus)
    q_uni = set(tokenize_uni(query))
    # 归一化只在「记忆库能理解」的 query 词上做：库里压根没出现过的词不提供区分度，
    # 但会把分母撑大，把"只命中一个稀有字"的强信号压成 0。
    q_uni_known = [t for t in q_uni if t in uni_idf]
    denom_uni = sum(uni_idf[t] for t in q_uni_known) or 1.0
    q_bi_known = [t for t in q_set if t in idf]
    denom_bi = len(q_bi_known) or 1

    # 「线索」：query 里那些罕见的词（IDF 高 = 专名、少见的说法、具体的细节）。
    # 人想起一件事通常不是"话题相关"，而是被一个具体的词勾出来的
    # ——「豆豆」「银屑病」「杭州」这种，一提就全回来了。
    idf_vals = sorted((idf.get(t, 0.0) for t in q_set), reverse=True)
    rare_th = idf_vals[len(idf_vals) // 3] if idf_vals else 0.0
    q_rare = [t for t in q_set if idf.get(t, 0.0) > 0 and idf.get(t, 0.0) >= rare_th]
    emo_in_query = bool(_EMO_WORDS.search(query or ""))

    hits: list[Hit] = []
    # ---- 可选的「重点词表」soft boost ----
    # `{词: 权重}` 由调用方自己的策略喂进来（默认空 = 完全不参与打分）。
    # 命中任一词的记忆轻微上浮：权重 5 加 0.05、其余加 0.03，封顶 0.06。
    # 上限压得比一条硬线索（cue 0.12）还小 —— 它只该在两条分数接近时挪一下顺序，
    # 不能盖过词面命中本身；而且**只加不扣**：没进词表的东西不该因此掉分。
    boost_words: dict = {}
    boost_keys = q_set & set(boost_words) if boost_words else set()
    for idx, m in enumerate(mems):
        mtoks = tokenize(m.text + " " + (m.quote or ""))
        m_set = set(mtoks)
        m_uni = set(uni_corpus[idx])
        shared_bi = q_set & m_set
        union_bi = q_set | m_set
        lex_bi = 0.7 * (len(shared_bi) / denom_bi) + 0.3 * (len(shared_bi) / (len(union_bi) or 1))
        lex_uni = sum(uni_idf.get(t, 1.0) for t in (q_uni & m_uni)) / denom_uni
        lex = 0.55 * lex_bi + 0.45 * lex_uni
        sem = _cos(q_vec, _vec(mtoks, idf))
        # 时间衰减的参考点取「发生时间」和「最后一次被想起」里较晚的那个：
        # 一段老事只要一直在被提，它就不老。
        rec = _recency(m.active_ts(), m.kind)
        # 强化：hit_count 越高，显著度这一维越稳（对数增长，封顶 +45%）
        boost = 1.0 + min(0.45, 0.10 * math.log1p(m.hit_count))
        sal = min(1.0, (m.salience / 3.0) * boost)
        base = (W["lex"] * lex + W["sem"] * sem + W["rec"] * rec + W["sal"] * sal) * m.confidence

        # ---- 线索强度：稀有词被勾中 + 情绪回声 ----
        cue = (sum(1 for t in q_rare if t in m_set) / len(q_rare)) if q_rare else 0.0
        emo_echo = 0.0
        if emo_in_query and m.salience >= 2.0:
            emo_echo = min(1.0, (m.salience - 1.5) / 1.5)

        # ---- 可及性：忘掉的东西不是没了，是"想不起来"----
        # 线索不对 → 几乎够不着；线索一对 → 立刻回到接近满值。
        if m.retired_at:
            access = RETIRED_ACCESS + (1.0 - RETIRED_ACCESS) * cue
        else:
            access = 1.0
        score = base * access
        # 触景生情：被具体的词或同一类情绪勾起来时，额外浮上来一层
        score += 0.12 * cue + 0.05 * emo_echo

        # 问「你还记得吗」时，偏好/关系/承诺 这类"长期事实"优先
        if probe and m.kind in ("preference", "relation", "promise", "boundary", "trait", "self"):
            score += 0.06
        # 已被高频命中的条目轻微降温，防止同一句话每轮都占位
        # （上限压到 0.02，否则会把上面的强化完全抵消掉）
        score -= min(0.02, 0.002 * m.hit_count)

        # 重点词表的 soft boost（默认空）：只加不扣 —— 没进词表的记忆不该因此掉分。
        boost_add = 0.0
        if boost_keys:
            for w in boost_keys:
                if w in m_set:
                    boost_add += 0.05 if boost_words.get(w, 0) >= 5 else 0.03
            boost_add = min(0.06, boost_add)
            score += boost_add

        hits.append(Hit(m, round(score, 5), {"lex": round(lex, 3), "sem": round(sem, 3),
                                             "rec": round(rec, 3), "sal": round(sal, 3),
                                             "cue": round(cue, 3),
                                             "emo": round(emo_echo, 3),
                                             "boost": round(boost_add, 3)}))

    hits.sort(key=lambda h: h.score, reverse=True)
    # 分数门槛：记忆库小于 k 的时候（聊天刚开始），不设门槛会把"完全不相干"的也带进来，
    # 表现成"她什么都要提一嘴"。0.04 大致是"没有任何词面重叠、也不靠显著度"的下限。
    hits = [h for h in hits if h.score >= 0.04]
    picked = _mmr(hits, k)
    if touch:
        _strengthen(store, picked)
    return picked


def _strengthen(store: Store, picked: list[Hit]) -> None:
    """召回一次 = 想起一次：记下 hit，退场过的条目被强相关命中就复活。

    失败不影响本轮回复 —— 记忆强度是后台事实，不该阻断对话。
    """
    for h in picked:
        if h.score < 0.05:
            continue
        try:
            store.touch_memory(h.mem.id)
            # 退场的东西被"触景生情"勾出来就复活。
            # 判据不是"分数够高"（那只是话题相关），而是线索够硬：
            # 具体的词被勾中了，或者情绪对上了。
            if h.mem.retired_at and (h.parts.get("cue", 0) >= 0.4 or h.score >= 0.35):
                store.revive(h.mem.id)
        except Exception:      # noqa: BLE001
            continue


def maintain(store: Store, peer: str, *, at: int | None = None,
             dry: bool = False, rng=None) -> dict:
    """遗忘：让长期没被想起、又不重要的条目退场。

    这不是删除 —— 原文在 L0，条目只是退出检索面；被线索勾中还能回来（见 retrieve）。

    两条阈值分开，因为人不是均匀地忘：
      * 从没回头想起过的，掉得快（1.5 倍半衰）
      * 想起来过、但很久没再想起的，掉得慢（2.5 倍）
    三样东西永不退场：边界、承诺、高显著度（创伤/重大事件）。

    之前是批量退场：年龄一过阈值就全退，像开闸放水一样。
    改成**概率退场**：过阈值后，每轮以 `p = 1 - exp(-age / (tau*factor))`
    的概率退场。这样同一个年龄的记忆不是全走或全留，而是像人记忆那样
    渐进模糊 —— 有些还能捉摸着回想起，另一些彻底没影。
    """
    import random
    rng = rng or random
    t = at or now()
    retired = 0
    for m in store.all_memories(peer):
        if m.kind in ("boundary", "promise"):
            continue
        if m.salience >= 2.3 or m.hit_count >= 3:
            continue
        factor = 1.5 if m.hit_count == 0 else 2.5
        tau = TAU_DAYS.get(m.kind, 60.0)
        age_days = max(0.0, (t - m.active_ts()) / 86400.0)
        if age_days <= tau * factor:
            continue
        # 概率退场：年龄越大、概率越高，但不是 100%
        p = 1.0 - math.exp(-age_days / (tau * factor))
        if rng.random() < p:
            if not dry:
                store.retire(m.id)
            retired += 1
    return {"retired": retired}


def _mmr(hits: list[Hit], k: int, lam: float = 0.82) -> list[Hit]:
    """轻量 MMR：避免召回一堆近义重复条目，把名额留给不同侧面。"""
    chosen: list[Hit] = []
    pool = hits[: k * 6]
    while pool and len(chosen) < k:
        best, best_v = None, -1e9
        for h in pool:
            if chosen:
                sim = max(_jaccard(h.mem.text, c.mem.text) for c in chosen)
            else:
                sim = 0.0
            v = lam * h.score - (1 - lam) * sim
            if v > best_v:
                best, best_v = h, v
        chosen.append(best)
        pool.remove(best)
    return chosen


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def recall_messages(store: Store, peer: str, query: str, *, k: int = 6,
                    exclude_recent: int = 20) -> list[tuple[int, float]]:
    """从 L0 原始日志里召回"原话片段"，给记忆条目提供佐证。"""
    cand = store.search_lex(peer, query, limit=120)
    if not cand:
        return []
    q = set(tokenize_query(query))
    scored: list[tuple[int, float]] = []
    cutoff = max(cand) - 0  # 仅用于类型稳定
    _ = cutoff
    for mid in cand:
        for m in store.get_msgs([mid]):
            mtoks = set(tokenize(m.text))
            if not mtoks:
                continue
            lex = len(q & mtoks) / math.sqrt((len(q) or 1) * len(mtoks))
            rec = math.exp(-max(0.0, (now() - m.ts) / 86400.0) / 30.0)
            score = 0.7 * lex + 0.3 * rec
            # 她自己对这条消息打过分的（M2）：0.05 × (分 - 3)，**只加不扣**。
            # 3 分是默认值（= 没表态），4 分 +0.05、5 分 +0.1；1、2 分不加也不减 ——
            # 她标"不重要"只表示"别优先"，不表示"从记忆里抹掉"。
            imp = int(getattr(m, "importance", 0) or 0)
            if imp > 3:
                score += 0.05 * (imp - 3)
            scored.append((mid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    keep = [x[0] for x in scored[: k + exclude_recent]]
    tail = sorted(keep)[-exclude_recent:] if exclude_recent else []
    out = [(mid, s) for mid, s in scored if mid in keep and mid not in tail]
    return out[:k]
