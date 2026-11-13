"""她自己的心情状态量：三个连续量 + 一条事实性暗示。纯规则、无 IO、无 LLM。

**和 state['mood'] 不是一回事，别混：**

  * ``state['mood']``   = **对方的**情绪，从他的话里抽出来的一个词（见 distill._emo_mood）。
  * ``state['mood_val']`` / ``energy`` / ``taut`` = **她自己的**状态量，本模块负责。

为什么她需要自己的量：原来的 L3 只有「关于他的情绪」和亲密度，她这一侧的
「今天还有多少力气」「哪根线还挑着没放下」根本无处存 —— 于是每次都从零开始回话，
情绪只能靠模型当场编，前后对不上。

三个量（都是连续值，允许在中间地带，不是好/不好二选一）：

  ``mood_val``  -1..1   好不好，默认 0
  ``energy``     0..1   还剩多少力气，默认 0.6
  ``taut``       0..1   绷没绷（有没有哪根线被挑了一下、还没放下），默认 0.1
  ``self_mood_ts``      最后更新时刻（整数秒），decay 算 dt 用

推动来源的权重（设定口径，别扩）：她手里的事为主，关于他的事垫底，外部看到的东西偶尔。
落到规则上就是 push() 里那几条 —— 不写成分类器，也不去猜他的语气。

时间只会让它往中间回，不会瞬间跳变：decay() 用 ``0.5 ** (dt / 半衰期)``。
"""
from __future__ import annotations

import re

DEFAULT_MOOD = 0.0
DEFAULT_ENERGY = 0.6
DEFAULT_TAUT = 0.1

MOOD_RANGE = (-1.0, 1.0)
UNIT_RANGE = (0.0, 1.0)

# 半衰期（秒）。taut 最快散（两小时就该淡了），energy 次之，mood_val 最慢。
MOOD_HALF_LIFE = 12 * 3600
ENERGY_HALF_LIFE = 6 * 3600
TAUT_HALF_LIFE = 2 * 3600

# 单条消息总变化上限：一条消息不该改变一天的心情。
MAX_D_MOOD = 0.25
MAX_D_TAUT = 0.35
MAX_D_ENERGY = 0.20

# 空档（gap_hours > 6）压 mood_val 时的地板：不许掉成怨气。
GAP_MOOD_FLOOR = -0.3
GAP_HOURS = 6.0

# 敷衍是弱刺激，累计也有个地板：一句「嗯」不该攒出一条怨气。
# 没有这个地板，十几条「嗯」就能把 mood_val 拉到底 —— 那不是陪伴，那是扣分。
BLAND_MOOD_FLOOR = -0.6

# ---------------- 词表（够用即可，别扩成分类器）----------------
_PRAISE = re.compile(
    r"谢谢|多谢|辛苦了|辛苦你|厉害|好听|好看|可爱|喜欢你|喜欢听|真棒|太棒|很棒|"
    r"真好|太好了|靠谱|贴心|细心|懂我|佩服|说得对|说得真好|有你在|好贴心"
)
# 敷衍：极短、且整句没有任何实质内容。集合命中 + 同字重复（"嗯嗯""哦哦哦"）。
_BLAND = {
    "嗯", "哦", "喔", "噢", "啊", "行", "好", "对", "是", "在", "中", "可以",
    "好的", "好吧", "行吧", "嗯嗯", "哦哦", "呵呵", "收到", "知道了", "ok",
}
_BLAND_CHARS = set("嗯哦喔噢啊行好对是可以在")
_CORRECT = re.compile(
    r"(?<!是)不是|不是这样|错了|你记错|记错了|不对|搞错|谁说|我说的是|答错"
)
_UNSURE = re.compile(
    r"我不太清楚|不太清楚|不太确定|不确定|我记不清|记不太清|记不清了|说不好|"
    r"大概是|可能吧|我不太记得|说不准"
)
# 她自己手里的事（权重最高的一类）
_DONE = re.compile(r"搞定了|弄完了|做完了|跑通了|写完了|收尾了|弄好了|改好了|通过了")
_STUCK = re.compile(r"又报错|还是不行|卡住了|不知道怎么办|搞不定|算了")


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _num(v, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    # NaN 兜一下：NaN 会让后面所有比较都变成 False，值就永远卡住不动
    return default if f != f else f


def _bare(text: str) -> str:
    """去掉空白和标点，只留字和数字 —— 判断"极短"用。"""
    return re.sub(r"[^\w]+", "", text or "", flags=re.UNICODE).lower()


def _is_bland(text: str) -> bool:
    t = _bare(text)
    if not t or len(t) > 4:
        return False
    if t in _BLAND:
        return True
    return len(set(t)) == 1 and t[0] in _BLAND_CHARS


def _toward(cur: float, target: float, dt: float, half_life: float) -> float:
    """按经过时间往 target 靠；dt 越大靠得越彻底，但永远到不了 target。"""
    if dt <= 0:
        return cur
    k = 0.5 ** (dt / float(half_life))
    return cur + (target - cur) * (1.0 - k)


def looks_unsure(text: str) -> bool:
    """这句话是不是"她承认自己没把握"。

    给调用方用：即时路径只处理他这一轮的消息，但"她上一条说了没把握"这件事
    也会挑一下她的线（taut +0.1）。由调用方扫她最近一条，作为 ctx['doubt'] 递进来。
    """
    return bool(_UNSURE.search(text or ""))


def decay(state: dict, now_ts: int) -> dict:
    """按经过的时间让三个量往中间回。

    原地不改入参，返回新 dict。``dt <= 0``（时钟倒退、同一秒内第二次调用）
    原样返回 —— 不然一次重复调用就能把数值动掉，推进入口会被 call 两次。

    Args:
        state: 至少含 mood_val / energy / taut / self_mood_ts 的字典。
        now_ts: 当前时刻（整数秒）。

    Returns:
        新的 state dict（含更新后的 self_mood_ts）。
    """
    st = dict(state or {})
    ts = int(_num(st.get("self_mood_ts"), 0))
    dt = int(now_ts) - ts
    if ts <= 0 or dt <= 0:
        return st
    st["energy"] = _toward(_num(st.get("energy"), DEFAULT_ENERGY), 0.5, dt, ENERGY_HALF_LIFE)
    st["taut"] = _toward(_num(st.get("taut"), DEFAULT_TAUT), 0.0, dt, TAUT_HALF_LIFE)
    st["mood_val"] = _toward(_num(st.get("mood_val"), DEFAULT_MOOD), 0.0, dt, MOOD_HALF_LIFE)
    st["energy"] = round(_clamp(st["energy"], *UNIT_RANGE), 4)
    st["taut"] = round(_clamp(st["taut"], *UNIT_RANGE), 4)
    st["mood_val"] = round(_clamp(st["mood_val"], *MOOD_RANGE), 4)
    st["self_mood_ts"] = int(now_ts)
    return st


def push(state: dict, text: str, role: str = "user", ctx: dict | None = None) -> dict:
    """单条消息对三个量的推动。纯函数，不改入参。

    Args:
        state: 当前 state（含 mood_val / energy / taut）。
        text: 这条消息的原文。
        role: ``user`` = 他说的；``assistant`` = 她说的。
        ctx: 可选上下文，认这几个键（其它忽略）：
            ``corrected``  他这轮是否在更正她（显式给 True 时，正则没命中也算）
            ``doubt``      她这轮是否说了没把握的话（她那一侧的词面不好抓时由调用方给）
            ``gap_hours``  距上一条他说话隔了多少小时
            ``now_ts``     写进 self_mood_ts 的时刻

    Returns:
        新的 state dict；单条总变化已被 MAX_D_* 与各自值域夹住。
    """
    st = dict(state or {})
    ctx = ctx or {}
    t = text or ""
    r = (role or "user").strip().lower()

    mv0 = _num(st.get("mood_val"), DEFAULT_MOOD)
    en0 = _num(st.get("energy"), DEFAULT_ENERGY)
    ta0 = _num(st.get("taut"), DEFAULT_TAUT)
    d_mv = d_en = d_ta = 0.0

    if r == "user":
        # 被夸/被谢：心情升，力气也回一点（被人在意是回血，不是消耗）
        if _PRAISE.search(t):
            d_mv += 0.15
            d_en += 0.05
        # 敷衍：不算伤害，但会掉一口气（累计到地板就不再往下压）
        elif _is_bland(t):
            if mv0 > BLAND_MOOD_FLOOR:
                d_mv -= 0.06
            d_en -= 0.05
        # 被更正：这是最挑线的一种 —— 绷起来，好心情折一点
        if _CORRECT.search(t) or ctx.get("corrected"):
            d_ta += 0.30
            d_mv -= 0.15
    else:
        # 她手里的事：权重最高的一类
        if _DONE.search(t):
            d_mv += 0.10
            d_en -= 0.06
        if _STUCK.search(t):
            d_mv -= 0.08
            d_ta += 0.08

    # 她承认没把握：不是难过，是没放下
    if ctx.get("doubt") or (r == "assistant" and _UNSURE.search(t)):
        d_ta += 0.10

    # 先按单条上限裁剪，再落到值域上
    mv = _clamp(mv0 + _clamp(d_mv, -MAX_D_MOOD, MAX_D_MOOD), *MOOD_RANGE)
    en = _clamp(en0 + _clamp(d_en, -MAX_D_ENERGY, MAX_D_ENERGY), *UNIT_RANGE)
    ta = _clamp(ta0 + _clamp(d_ta, -MAX_D_TAUT, MAX_D_TAUT), *UNIT_RANGE)

    # 空档：隔了很久才说话，力气掉、心情也淡下去 —— 但地板锁在 -0.3（不许成怨气）。
    # 注意是"这次降幅不越过地板"，已经更低的值不会被抬上来。
    try:
        gap = float(ctx.get("gap_hours") or 0.0)
    except (TypeError, ValueError):
        gap = 0.0
    if gap > GAP_HOURS:
        en = max(0.0, en - 0.12)
        cand = mv - 0.08
        mv = cand if cand >= GAP_MOOD_FLOOR else min(mv, GAP_MOOD_FLOOR)
        # 空档不该把已经绷着的线松开（他没回不是她的错，但也没放下）
        ta = min(1.0, ta + 0.02)

    # 总变化上限再收一次（空档与常规增量可能叠加）
    mv = _clamp(mv, mv0 - MAX_D_MOOD, mv0 + MAX_D_MOOD)
    en = _clamp(en, en0 - MAX_D_ENERGY, en0 + MAX_D_ENERGY)
    ta = _clamp(ta, ta0 - MAX_D_TAUT, ta0 + MAX_D_TAUT)

    st["mood_val"] = round(_clamp(mv, *MOOD_RANGE), 4)
    st["energy"] = round(_clamp(en, *UNIT_RANGE), 4)
    st["taut"] = round(_clamp(ta, *UNIT_RANGE), 4)
    ts = ctx.get("now_ts")
    if ts is not None:
        st["self_mood_ts"] = int(_num(ts, 0)) or st.get("self_mood_ts")
    return st


# 自主时间里的幅度。一天最多 12 次开口，单次小是故意的 —— 攒一天也该看得出来。
AUTO_ENERGY_LOOK = -0.04      # 翻了东西，费点劲
AUTO_MOOD_SPOKE = 0.03        # 有话可说（看见了让他心里动一下的），是好事
AUTO_TAUT_SILENT = 0.05       # 翻了，但没出口：那一下留在里面
AUTO_TAUT_CAP = 0.6           # 跟 autonomy.LOOP_TAUT_CAP 同一个口径
# 「该做的重要事」被她自己感兴趣的东西挤掉了（tasks.pick 的 pushed）：
# 她不会没事 —— 事后有点心虚。幅度刻意很小，且不给「你在心虚」这种断言，
# 只让绷劲抬一点、由她自己在下一次说话里决定要不要露出来。
AUTO_TAUT_PUSHED = 0.02
# 她这一轮真在自己的东西上动了手（家里落了字、改了东西）：做事耗力，心里也踏实一点。
# 这是「她有自己的生活」唯一可验证的正反馈 —— 不是她说她做了什么，是硬盘上多了字。
AUTO_ENERGY_OWN = -0.02
AUTO_MOOD_OWN = 0.02
# 她在**自己的出口**上留过字（换签名、发说说）：与上一条同构，只是发生在她家之外。
# 一次补账最多按两次算 —— 隔了几小时才消费到的流水，不该把量一次推飞。
AUTO_ENERGY_CHANNEL = -0.01
AUTO_MOOD_CHANNEL = 0.02
AUTO_CHANNEL_MAX = 2


def nudge(state: dict, *, valley: bool = False, spoke: bool = False,
          pushed: bool = False, produced: bool = False,
          channel_out: int = 0) -> dict:
    """自主时间里「她干了什么」对心情的轻推。纯函数，**不动 self_mood_ts**。

    与 decay 的分工：decay 管「时间过去了」，这个管「这段时间里她做了什么」。
    只认几个事实 —— 这一轮能不能翻东西（谷时段带工具）、有没有说出话来、
    有没有把该做的事挤掉、有没有在她自己的东西上动手、有没有在自己的出口上留字。
    **不去猜她翻到了什么、写了什么**：一猜就得编内容，编出来的会被她当回忆讲下去。

    Args:
        state: 当前 state。
        valley: 这一轮是否带了工具（谷时段）。
        spoke: 是否真的说出了一句（不是空文本、不是 provider 报错）。
        pushed: 这一轮「该做的重要事」被她自己感兴趣的东西挤掉了（tasks.pick）。
        produced: 这一轮她家里真有了改动（tasks.home_changes 非空）——
            她自己那件事往前挪了一步。她没说话也算数，这正是"她有自己的生活"。
        channel_out: 这一段时间她在自己的出口上留了几条（channel.new_events）——
            换签名、发说说。话放到别处了，所以也不记「憋着没说」。

    Returns:
        新的 state dict；``self_mood_ts`` 原样保留 —— 衰减的基准不能被这个动掉。
    """
    st = dict(state or {})
    mv = _num(st.get("mood_val"), DEFAULT_MOOD)
    en = _num(st.get("energy"), DEFAULT_ENERGY)
    ta = _num(st.get("taut"), DEFAULT_TAUT)

    try:
        n_out = max(0, int(channel_out or 0))
    except (TypeError, ValueError):
        n_out = 0

    if valley:
        en += AUTO_ENERGY_LOOK
    if spoke:
        mv += AUTO_MOOD_SPOKE
    elif valley and not produced and not n_out:
        # 翻了半天、什么也没落下：那一下留在里面。
        # 但真在自己东西上动过手、或者在出口上留过字的不算「想说没说」——
        # 她不是憋着，她是在做事，或者已经把那句话放在别处了。
        ta = min(AUTO_TAUT_CAP, ta + AUTO_TAUT_SILENT)
    if produced:
        en += AUTO_ENERGY_OWN
        mv += AUTO_MOOD_OWN
    if n_out:
        # 她真在自己的出口上留了字（签名 / 说说）：做事耗力，心里踏实一点。
        k = min(n_out, AUTO_CHANNEL_MAX)
        en += AUTO_ENERGY_CHANNEL * k
        mv += AUTO_MOOD_CHANNEL * k
    if pushed:
        ta = min(AUTO_TAUT_CAP, ta + AUTO_TAUT_PUSHED)

    st["mood_val"] = round(_clamp(mv, *MOOD_RANGE), 4)
    st["energy"] = round(_clamp(en, *UNIT_RANGE), 4)
    st["taut"] = round(_clamp(ta, *UNIT_RANGE), 4)
    return st


def hint(state: dict) -> str:
    """把三个数分档成一句**事实性**短句，给 prompt 用。

    只陈述量本身，不下情绪判断：「这会儿的劲头：有点不足；绷着」。
    绝不出现「你心情不好」「你很开心」「你现在很低落」这类断言，也不带数字 ——
    一旦写成断言，她会照着演，越演越像设定；给事实她才能自己决定怎么用。

    Returns:
        一行，无换行、无数字。
    """
    st = state or {}
    ev = _num(st.get("energy"), DEFAULT_ENERGY)
    tv = _num(st.get("taut"), DEFAULT_TAUT)
    mv = _num(st.get("mood_val"), DEFAULT_MOOD)

    if ev >= 0.70:
        head = "很足"
    elif ev >= 0.45:
        head = "还行"
    elif ev >= 0.25:
        head = "有点不足"
    else:
        head = "快见底"
    bits = [f"这会儿的劲头：{head}"]

    if mv >= 0.35:
        bits.append("心气偏亮")
    elif mv <= -0.35:
        bits.append("心气偏沉")

    if tv >= 0.50:
        bits.append("绷着")
    elif tv >= 0.28:
        bits.append("有点绷")
    return "；".join(bits)
