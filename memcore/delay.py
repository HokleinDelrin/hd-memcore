"""延迟回复：不是每句话都当场答（2026-11-03 第 7 件）。

为什么单独一层：
- 秒回是"服务"的节奏，不是人的节奏。人在忙、在别人家里、刚洗完手，
  都会让一句"在吗"多等几十秒。她要是每条都零延迟，那就还是消息驱动的人偶。
- 但延迟**不能**做成独立定时器：一旦另起任务到点发送，就会和「他连发的新消息」
  与「她自己主动开口」抢同一条时间线 —— 谁先到点谁发，顺序就乱了。
  所以这里只算"该等多久"，真正等待在 AstrBot 的 pipeline 里原地睡（见
  `main._delay_reply`）：天然按他发消息的顺序排队，中途他再说一条，
  这一轮就自己作废，让最新那条来答。

本模块只做两件纯事，方便单测：
  - `pick(...)`：**该等多长**，以及为什么（原因进日志，别让延迟变成黑箱）
  - `note(...)`：把这一次记进 `delay_state.json`（只读用途：体检时看得到节奏）

上限的由来：前端走 Cloudflare 隧道，免费版单个 HTTP 请求约 100 秒会被掐。
所以 `cap` 默认 75 秒 —— 再长宁可让这一轮放弃（他要是等不了会再发一条）。
"""
from __future__ import annotations

import json
import pathlib
import random
import time
from dataclasses import dataclass

STATE_NAME = "delay_state.json"
KEEP = 40                      # 状态里只留最近多少次
DEFAULT_LO = 8.0
DEFAULT_HI = 45.0
DEFAULT_CAP = 75.0
LONG_TEXT = 60                 # 他这条超过这么多字，她得先读完
QUICK_REPLY_GAP = 90.0         # 他上一条与这条隔得比这还近 → 他在等，别装忙


@dataclass(frozen=True)
class Plan:
    """一次延迟的决定。`why` 是要给人看的，不是给模型看的。"""

    seconds: float
    why: str
    parts: tuple = ()

    @property
    def skip(self) -> bool:
        return self.seconds < 1.0


def pick(
    text: str = "",
    *,
    phase: str = "awake",
    gap_s: float | None = None,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
    cap: float = DEFAULT_CAP,
    rng: random.Random | None = None,
) -> Plan:
    """该等多久。

    Args:
        text: 他这条的原话（长度是唯一用得上的信息：长句要多读一会儿）。
        phase: `memcore.daytime.phase()` 的结果（asleep 不会走到这里）。
        gap_s: 距他上一条消息多少秒；None 表示这是这一阵的第一条。
        lo/hi: 基础区间（秒）。lo>=hi 时按 lo 处理，别让配置写反了导致负数。
        cap: 硬上限，见模块说明。
        rng: 传 random.Random(种子) 就能复现，测试用。

    Returns:
        Plan：秒数 + 一句原因。
    """
    rng = rng or random.Random()
    lo = max(0.0, float(lo))
    hi = max(lo, float(hi))
    cap = max(0.0, float(cap))

    base = rng.uniform(lo, hi) if hi > lo else lo
    parts: list[str] = [f"基础 {base:.0f}s"]

    # 他在写长句：她得先读完才答得上。按超出部分最多再加 50%。
    n = len(text or "")
    if n >= LONG_TEXT:
        bump = min(0.5, (n - LONG_TEXT) / 200.0)
        base *= 1.0 + bump
        parts.append(f"长句 +{bump * 100:.0f}%")

    # 犯困：反应本来就慢，这是唯一允许"变慢"的时段。清醒时不额外加。
    if phase == "groggy":
        base *= 1.6
        parts.append("犯困 ×1.6")

    # 他刚说完不到一分半又说了一句：那是他在等，不是在忙 —— 收起延迟。
    if gap_s is not None and 0 <= gap_s < QUICK_REPLY_GAP:
        base *= 0.5
        parts.append("他连着发 ×0.5")

    if cap is not None and float(cap) > 0 and base > cap:
        parts.append(f"封顶 {float(cap):.0f}s")
        base = float(cap)
    elif cap is not None and float(cap) <= 0:
        # cap<=0 是"明确不要延迟"（配置写 0 的意思是关掉，不是"没有上限"）
        parts.append("上限 0 = 不延迟")
        base = 0.0
    return Plan(round(max(0.0, base), 1), "，".join(parts), tuple(parts))


def state_path(data_dir) -> pathlib.Path:
    return pathlib.Path(data_dir) / STATE_NAME


def superseded(mid: int | None, last_id: int | None) -> bool:
    """睡醒之后，这一轮还算不算数。

    为什么单独成一个函数：这是"他连发时只回最后一条"的唯一判定，
    写反了不会报错，只会表现成"她有时候莫名其妙不回" —— 那种 bug 最难查。
    所以判定抽出来单测，插件里只管调用。

    Args:
        mid: 睡之前记下的 L0 行号（他这条）。
        last_id: 睡醒后库里最新的行号。

    Returns:
        True = 这期间又有新消息进来，本轮作废，让最新那条来答。
    """
    if mid is None or last_id is None:
        return False
    try:
        return int(last_id) != int(mid)
    except (TypeError, ValueError):
        return False


def load(path) -> dict:
    """读状态；没有就是没延迟过，不是错误。"""
    try:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def note(path, *, seconds: float, why: str, peer: str = "",
         dropped: bool = False) -> dict:
    """记一次（真正等了的，或睡醒后发现被新消息取代而放弃的）。

    只写这一份文件；读它的是体检脚本和人，不是她的 prompt ——
    "我上次让他等了 37 秒"这种事一旦进了注入，她就变成在表演耐心。
    """
    st = load(path)
    events = list(st.get("events") or [])
    events.append({
        "ts": int(time.time()),
        "peer": peer,
        "seconds": round(float(seconds), 1),
        "why": why,
        "dropped": bool(dropped),
    })
    st["events"] = events[-KEEP:]
    today = time.strftime("%Y-%m-%d")
    if st.get("day") != today:
        st["day"] = today
        st["today"] = 0
    st["today"] = int(st.get("today") or 0) + (1 if not dropped else 0)
    st["last_at"] = int(time.time())
    try:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return st


def summary(state: dict) -> str:
    """一行话：今天等了几次、平均多久、最近一次为什么。

    "今天"必须按 `day` 判断再报 —— 跨过零点还念昨天的数，第二天早上会看着像
    "她今天已经让他等过 12 次"，实际一次都没有（2026-11-04 早上就撞见过）。
    """
    evs = [e for e in (state or {}).get("events") or [] if isinstance(e, dict)]
    if not evs:
        return "还没让他等过（delay_state.json 不存在或为空）"
    real = [e for e in evs if not e.get("dropped")]
    avg = sum(float(e.get("seconds") or 0) for e in real) / len(real) if real else 0.0
    last = evs[-1]
    ago = int(time.time()) - int(last.get("ts") or 0)
    today = (state.get("today", 0)
             if state.get("day") == time.strftime("%Y-%m-%d") else 0)
    return (f"今天让等 {today} 次（累计 {len(real)} 次，平均 {avg:.0f}s）；"
            f"最近一次 {last.get('seconds', 0)}s{'（作废）' if last.get('dropped') else ''}"
            f"，{max(0, ago) // 60} 分钟前：{last.get('why', '')}")
