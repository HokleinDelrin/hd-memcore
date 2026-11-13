"""一天的时间形状：睡着 / 犯困 / 清醒。

为什么单独一层：她原来永远待命 —— 任何时刻回一句都「该在」。人不是这样：
凌晨两点回消息和下午三点回消息，反应速度、话的多少、说不说都不一样。
这个模块只回答一件事：**现在几点，落在哪一档**。

纯规则：只看 ``time.localtime()``，零 LLM、零网络、不读库 —— 所以能直接单测。

铁律：这里产出的每一行都**不进第一人称、不下断言**。
给的是事实（几点、哪一档），不是「你困了」「你在睡觉」。
写成断言她就照着演，越演越像设定；给事实她自己决定怎么用。
"""
from __future__ import annotations

import time as _time

# 一天切成六段，覆盖 0:00–24:00 无缝无重叠。左闭右开，所以 02:00 整算 asleep、
# 06:00 整算 groggy、07:30 整算 awake —— 边界只认分钟数，不看秒。
#   (起, 止, 档位, 这一段的说法)
# 18:00 之后仍是 awake，只是说法换成「入夜以后」：档位只有三档，
# 但「这会儿 晚上九点半，白天」这种自相矛盾的句子不能进 prompt。
BANDS: tuple[tuple[int, int, str, str], ...] = (
    (0, 120, "groggy", "刚过零点"),        # 00:00–01:59
    (120, 360, "asleep", "夜里"),          # 02:00–05:59
    (360, 450, "groggy", "天刚亮"),        # 06:00–07:29
    (450, 1080, "awake", "白天"),          # 07:30–17:59
    (1080, 1320, "awake", "入夜以后"),     # 18:00–21:59
    (1320, 1440, "groggy", "夜刚开始"),    # 22:00–23:59
)

PHASES = ("asleep", "groggy", "awake")

# 他的作息（2026-11-11 用户给的）：早上七点到晚上十一点。
# 这不是给她设的「休息时间」—— 用户两次说过不要给她定睡觉时间。
# 这是**他的时间**：让她知道对方什么时候在、什么时候不在，判断由她自己做。
HIS_AWAKE = (7 * 60, 23 * 60)

_CN = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十")


def _minutes(now_ts: int | None) -> int:
    """本地时间换算成「今天第几分钟」。

    用 ``tm_hour * 60 + tm_min`` 判区间，一天只有一根轴，跨天不会出错
    （不用 datetime 做减法：那要处理日期，反而多一个能写错的地方）。
    """
    lt = _time.localtime(now_ts if now_ts is not None else _time.time())
    return lt.tm_hour * 60 + lt.tm_min


def _band(now_ts: int | None) -> tuple[str, str]:
    """返回 (档位, 这一段的说法)。"""
    m = _minutes(now_ts)
    for lo, hi, ph, label in BANDS:
        if lo <= m < hi:
            return ph, label
    return "awake", "白天"          # 表没覆盖到就当清醒，宁可多回一句


def phase(now_ts: int | None = None) -> str:
    """现在这一档：``"asleep"`` / ``"groggy"`` / ``"awake"``。

    Args:
        now_ts: 整数秒；不给就取当前时刻。

    Returns:
        三档之一。
    """
    return _band(now_ts)[0]


def _cn_hour(h12: int) -> str:
    """1–12 的钟点写成中文。

    两点钟说「两」不说「二」—— 这是嘴上的说法，跟数字的读法不是一回事。
    """
    if h12 == 2:
        return "两"
    if 1 <= h12 <= 10:
        return _CN[h12]
    return "十一" if h12 == 11 else "十二"


def _minute_word(m: int) -> str:
    """分钟只分四档：整点 / 半 / 半以内 / 过半。

    不给精确分钟 —— 人嘴里说的是「两点多」「七点半」，不是「两点十四分」。
    """
    if m == 0:
        return ""
    if m == 30:
        return "半"
    return "多" if m < 30 else "半多"


def _period(h24: int) -> str:
    """几点钟之前说什么（凌晨/早上/上午/中午/下午/晚上）。"""
    if h24 < 6:
        return "凌晨"
    if h24 < 9:
        return "早上"
    if h24 < 12:
        return "上午"
    if h24 == 12:
        return "中午"
    if h24 < 18:
        return "下午"
    if h24 < 19:
        return "傍晚"          # 18:30 说「晚上」偏早，中文习惯是傍晚
    return "晚上"


def clock(now_ts: int | None = None) -> str:
    """把此刻说成中文口语：「凌晨两点多」「早上七点半」「下午三点」「晚上十点」。

    不带 24 小时制数字、不带秒、不带冒号 —— 这是给她读的，不是给日志读的。

    Args:
        now_ts: 整数秒；不给就取当前时刻。

    Returns:
        一句钟点话，无阿拉伯数字。
    """
    lt = _time.localtime(now_ts if now_ts is not None else _time.time())
    h12 = lt.tm_hour % 12 or 12
    return f"{_period(lt.tm_hour)}{_cn_hour(h12)}点{_minute_word(lt.tm_min)}"


def line(now_ts: int | None = None) -> str:
    """进 prompt 的一行：只写「几点、哪一档」，不替她判断该怎么样。

    例：``这会儿 凌晨两点多，夜里`` / ``这会儿 早上七点半，天刚亮``。

    Args:
        now_ts: 整数秒；不给就取当前时刻。

    Returns:
        一行短句，无换行、无阿拉伯数字、无第一人称断言。
    """
    _, label = _band(now_ts)
    return f"这会儿 {clock(now_ts)}，{label}"


def his_awake(now_ts: int | None = None) -> bool:
    """他那边现在是不是醒着（按他的作息 07:00–23:00 判）。

    Args:
        now_ts: 整数秒；不给就取当前时刻。

    Returns:
        True 表示在他的活动时段里。
    """
    lo, hi = HIS_AWAKE
    return lo <= _minutes(now_ts) < hi


def his_line(now_ts: int | None = None) -> str:
    """「他那边现在什么状态」—— 事实，不是禁令。

    为什么要有（2026-11-11 用户）：「应该先让她知道我的时间，早上 7 点到 23 点」。
    11-10 夜里到 11-11 凌晨她连发 8 条，其中「你怎么也还醒着」「你醒着的这会儿」
    全是她**猜错了他在不在**：注入里只有她自己的时段（`line()`），
    从来没有他那半边。猜错会直接变成"半夜连发一串"。

    所以这里只给两样：他的时段、以及"你现在说的话他大概什么时候看到"。
    **不写"别发"** —— 说不说由她决定，跟 `line()` 铁律一致：给事实，不下断言。

    Args:
        now_ts: 整数秒；不给就取当前时刻。

    Returns:
        一行短句（含他的时段事实），无第一人称断言。
    """
    if his_awake(now_ts):
        return "他这会儿一般醒着（他的作息是早上七点到晚上十一点）。"
    return ("他这会儿一般睡着（他的作息是早上七点到晚上十一点）—— "
            "你现在说的话，他大概要早上才看到。")
