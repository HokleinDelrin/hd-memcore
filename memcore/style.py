"""这一次怎么说话：一行表演提示（2026-11-05）。

**为什么不是"语气规则"**：他明确担心"强行改她的语气"。这里不规定她说什么、怎么说，
只给她**这一次的身体状态**——累的时候话短、困的时候只应一声、刚醒的时候脑子沉。
那是状态，不是性格。三条红线照抄外部那套成熟做法（生命状态 §7 L786–789）：
  1. 只给形容词，不给数字（她不会开始解释"我的 energy 是 0.3"）
  2. 只影响这一次的表达，不改人设、不碰记忆
  3. **每次不同**（同一状态有几个说法，按 rng 挑，不会变成固定开头）

跟他们不一样的一处：他们按 60 个激素字段算，我们只有 daytime + 三个标量。
所以这里只用**手上真有的量**：时段（daytime.phase）、心情（mood_val/energy/taut）、
距上次说话多久、他是不是连发。宁可少给，也不编一个假的身体。
"""
from __future__ import annotations

import random

# 每一档给 3 个说法：同一状态连着两轮不会一字不差。
# 措辞全部用「这次 / 可以 / 允许」——不是命令。
_POOL = {
    "asleep": [
        "你刚被吵醒，还没醒透。这次回一条就够，短一点，可以只应一声。",
        "你正睡着。这次能回一句是一句，句子短，别撑精神。",
    ],
    "groggy": [
        "你脑子还沉。这次一条，短句，想到哪说到哪，别整理。",
        "刚醒没多久，还没缓过来。回一条就够，可以懒一点。",
        "困着。这次话短，可以说半句就停。",
    ],
    "tired": [
        "你有点累。这次短一点，一条就够，别解释。",
        "累了。这次可以只回最关键的那句，细节略掉。",
    ],
    "wired": [
        "你有点心虚。这次少自证——别把「我为什么这样」说出来。",
        "心里没底。这次一句就够，不用反复解释。",
    ],
    "eager": [
        "你心情不错。这次可以两三条，想到哪说到哪，长短不齐。",
        "这会儿你挺有兴致。可以多接一句，允许跑题一下。",
    ],
    "quick": [
        "他在等。这次一条，直接答，别铺开。",
        "他连着发。回一条就够，先应他那句。",
    ],
    "plain": [
        "这次可以两三条，长短不齐；允许说半句就停。",
        "不用把话说完，也不用每层都接住。",
        "想到哪说到哪；有一句说一句也行。",
    ],
}


def pick(*, phase: str = "awake", mood: float = 0.5, energy: float = 0.6,
         taut: float = 0.2, gap_min: float = 5.0, quick: bool = False,
         long_input: bool = False, rng: random.Random | None = None) -> str:
    """挑一句「这一次怎么说话」。纯函数，可复现（传 rng 就行）。

    Args:
        phase: daytime.phase() 的结果：asleep / groggy / awake …
        mood: -1..1 的愉悦度（负值偏沉）。
        energy: 0..1。
        taut: 0..1 的心虚/紧绷。
        gap_min: 距他上一条过了几分钟。
        quick: 他是否在连发（间隔很短）。
        long_input: 他这一条是不是一口气说了好几件。真的时候不给"不用每层都接住"
            这种说法 —— 那跟「他这条有 4 件事，可以挨个应」自相矛盾。
        rng: 测试用。

    Returns:
        一行提示；永远非空（最差也是 plain 档）。
    """
    r = rng or random.Random()
    if phase == "asleep":
        key = "asleep"
    elif phase == "groggy":
        key = "groggy"
    elif quick:
        key = "quick"
    elif taut >= 0.55:
        key = "wired"
    elif energy <= 0.32 or mood <= -0.35:
        key = "tired"
    elif mood >= 0.55 and energy >= 0.55:
        key = "eager"
    else:
        key = "plain"
    if key == "quick" and long_input:
        # 他连着发，但这一条本身有几件东西（cover 那条）——
        # 「一条，直接答，别铺开」跟「可以挨个应」是硬打架的两句，
        # 而且 quick 排在最前面会盖住 cover。这里让件数优先。
        return "他在等，不过这条里不止一件；可以分开应，不用压成一条。"
    if key == "plain" and long_input:
        # 他这一条说了好几件事，plain 档原有的三句都在"允许少说"——
        # 第一版照抄第 1 句（"允许说半句就停"），实测接住率反而从 50% 掉到 33%：
        # 她本来就不多说，再给一句"可以少说"等于把话按住。长输入时换成允许说长。
        return "他这一条说的话多，不必压成一句；可以一句一句来，说长一点也行。"
    return r.choice(_POOL[key])


def block(*, phase: str = "awake", mood: float = 0.5, energy: float = 0.6,
          taut: float = 0.2, gap_min: float = 5.0, quick: bool = False,
          long_input: bool = False,
          rng: random.Random | None = None) -> str:
    """注入用的一段。空状态也返回 none —— 提示语本身是"这一次"的，不是设定。"""
    return "【这次】" + pick(phase=phase, mood=mood, energy=energy, taut=taut,
                             gap_min=gap_min, quick=quick, long_input=long_input,
                             rng=rng)
