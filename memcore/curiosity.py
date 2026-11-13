"""她的「兴趣」：不是写出来的，是从行为里统计出来的。

定论（2026-11-03，用户拍板）：
  · 兴趣 = 「没人要求也会回去看的东西」，**不是「喜欢什么」**。
    写进人设卡就是编的；而且从第二天起，她说话会为了贴合那张清单变生硬。
  · 三判据：不是任务 / 会重复 / 会多出来。
    落地只留一条硬门槛（**跨天出现 ≥3 次**）+ 一条加权（说出口过），
    因为只有「跨天重复」是能从数据里直接数出来的事实，另两条各自有落点：
      不是任务 → 数据只从自主时间那趟来（对话回合里用同一个工具不记）；
      会多出来 → `spoke` 标记（她真的说了一句关于它的）。
  · 两周没碰就淡化；是变动列表，不是固定标签 —— 所以这里不删记录，只做窗口过滤。

数据从哪来：她自主时间翻了哪些东西（路径、搜索词）。
**只记「翻了没有」，绝不判断她看到了什么** —— 一判断就得我先替她编一条
（「她看到北大的文件夹有点感触」），那就成了我写她念。她真看见了什么，
信号在她自己说的那句话里，也就是 `spoke`。

谁写进来：`safe_fs.build_tools(..., on_access=回调)`，回调由 autonomy 传。
主 pipeline（对话回合）挂的同一批工具不传回调 —— 那些是任务，不算兴趣。
"""
from __future__ import annotations

import pathlib
import re
import time

WINDOW_DAYS = 14        # 统计窗口；同时也是「两周没碰就淡化」的实现
MIN_DAYS = 3            # 跨天出现几次才算兴趣（定论：3 次以上）
TOP_N = 5               # 注入里最多列几条
MAX_TOPIC = 40          # 主题键长度上限

# 路径里不带信息的段：去掉它们，剩下的前两段才是「她在看哪一类东西」。
# 例：C:/All/Work/VibeCoding/DVC/docs/note.md → Vibecoding/DVC
# 注意 "vibecoding" **不能**进这张表：All/Work 是他放东西的通用夹，
# 但 Vibecoding 下面那层（DVC、disk-sentry…）正是「一类东西」，要留着。
_GENERIC = frozenset({
    "c:", "d:", "e:", "users", "administrator", "desktop", "documents", "downloads",
    "all", "work", "projects", "project", "code", "codes", "repo", "repos",
    "src", "lib", "docs", "doc", "data", "test", "tests", "tmp", "temp", "new", "old",
    "build", "dist", "out", "output", "assets", "static", "public", "node_modules",
    ".git", "__pycache__", "venv", ".venv", "app", "apps", "main", "readme.md",
})
_WS = re.compile(r"\s+")
_TOO_GENERIC_PAT = re.compile(r"^[\W_]+$")


def _segments(target: str) -> list[str]:
    """把路径切成「有信息量的段」，去盘符、去通用目录名、去扩展名。"""
    raw = (target or "").strip().strip('"').strip("'")
    if not raw:
        return []
    parts = re.split(r"[\\/]+", raw)
    out: list[str] = []
    for seg in parts:
        s = seg.strip()
        if not s or s in (".", ".."):
            continue
        low = s.lower()
        if low in _GENERIC:
            continue
        if "." in low and low.rsplit(".", 1)[-1] in ("md", "py", "txt", "json", "csv", "yaml",
                                                     "yml", "toml", "ini", "log", "js", "ts",
                                                     "html", "css", "pdf", "docx", "xlsx", "png",
                                                     "jpg", "jpeg", "zip", "exe", "db"):
            s = s.rsplit(".", 1)[0]
            low = s.lower()
            if not s or low in _GENERIC:
                continue
        out.append(s)
    return out


def topic_of(target: str, source: str = "zero_read") -> str:
    """把「她看的那个东西」归一成主题键。空串表示这条不值一提（不记）。

    · 文件/目录（zero_read）：取有信息量的前两段，如 ``Vibecoding/DVC``。
      取「类」不取「篇」——她翻 DVC 下十个文件，是一件事，不是十件。
    · 搜索词（zero_grep）：词本身就是主题（``词:cold-vault``）；纯正则/纯符号丢掉，
      因为「搜过 ^\\d+$」不构成兴趣。
    """
    if source == "zero_grep":
        pat = _WS.sub(" ", (target or "").strip().strip('"')).strip()
        if not pat or len(pat) > MAX_TOPIC or _TOO_GENERIC_PAT.match(pat):
            return ""
        if re.search(r"[|()\[\]*+?\\^$]", pat):     # 正则/通配：太泛，不成主题
            return ""
        return "词:" + pat.lower()
    segs = _segments(target)
    if not segs:
        return ""
    key = "/".join(s.lower() for s in segs[:2])   # 小写归一：VibeCoding 和 vibecoding 是一类东西
    return key[:MAX_TOPIC]


def _now(ts: int | None = None) -> int:
    return int(ts if ts is not None else time.time())


def day_of(ts: int) -> str:
    """本地日历日。跨天判据靠它 —— 不能用 (now-ts)//86400，那是「满 24 小时」。"""
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def record(store, peer: str, source: str, target: str, ts: int | None = None) -> str | None:
    """记一条「她翻了这一类东西」。返回主题键（没记就是 None）。

    同一天同一类只留一条（UNIQUE 键顶上）—— 她一口气翻 DVC 下二十个文件，
    那是「今天去过 DVC 一次」，不是二十次。频次的意义在跨天。
    """
    topic = topic_of(target, source)
    if not topic:
        return None
    t = _now(ts)
    try:
        store.db.execute(
            "INSERT OR IGNORE INTO curiosity(peer,ts,day,source,topic,detail)"
            " VALUES(?,?,?,?,?,?)",
            (peer, t, day_of(t), source, topic, (target or "")[:200]),
        )
        store.db.commit()
    except Exception:                    # noqa: BLE001
        return None
    return topic


def mark_spoke(store, peer: str, topic: str, ts: int | None = None) -> None:
    """她说出口过 —— 「会多出来」那一维的落地：看完想说一句，不说难受。

    打在她最近碰过的那一条上，不新开行：这条记录的意义没变，只是多了个注脚。
    """
    if not topic:
        return
    try:
        store.db.execute(
            "UPDATE curiosity SET spoke=1 WHERE peer=? AND topic=?"
            " AND id=(SELECT id FROM curiosity WHERE peer=? AND topic=? ORDER BY ts DESC LIMIT 1)",
            (peer, topic, peer, topic),
        )
        store.db.commit()
    except Exception:                    # noqa: BLE001
        pass


def interests(store, peer: str, now_ts: int | None = None, *, window_days: int = WINDOW_DAYS,
              min_days: int = MIN_DAYS, limit: int = TOP_N) -> list[dict]:
    """她最近老回去看的那几类东西。

    Returns:
        ``[{topic, days, last_ts, spoke}]``，按跨天天数降序（同数看最近）。
        窗口内没记录的自然不在里面 —— 这就是「两周没碰就淡化」，不需要清理任务。
    """
    t = _now(now_ts)
    since = t - window_days * 86400
    try:
        rows = store.db.execute(
            "SELECT topic, COUNT(*) days, MAX(ts) last_ts, MAX(spoke) spoke"
            " FROM curiosity WHERE peer=? AND ts>=?"
            " GROUP BY topic HAVING days>=?"
            " ORDER BY days DESC, last_ts DESC LIMIT ?",
            (peer, since, min_days, limit),
        ).fetchall()
    except Exception:                    # noqa: BLE001
        return []
    return [{"topic": r["topic"], "days": int(r["days"]),
             "last_ts": int(r["last_ts"]), "spoke": int(r["spoke"])} for r in rows]


def line(store, peer: str, now_ts: int | None = None, *, limit: int = 3) -> str:
    """注入用的一行事实。没有就返回空串。

    措辞是**事实**（「回去翻过好几次的」），不是断言（「你喜欢 DVC」）：
    一说成「你的兴趣」，她就开始演那个兴趣。
    """
    it = interests(store, peer, now_ts, limit=limit)
    if not it:
        return ""
    names = "、".join(x["topic"] for x in it)
    return f"最近两周你回去翻过好几次的：{names}。"


def all_rows(store, peer: str, limit: int = 50) -> list[dict]:
    """给人看的原始记录（排查用，不进 prompt）。"""
    try:
        rows = store.db.execute(
            "SELECT ts,day,source,topic,detail,spoke FROM curiosity WHERE peer=?"
            " ORDER BY ts DESC LIMIT ?", (peer, limit)).fetchall()
    except Exception:                    # noqa: BLE001
        return []
    return [dict(r) for r in rows]


def _demo() -> None:                     # pragma: no cover - 手工跑
    import tempfile
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from memcore.store import Store
    from memcore import curiosity

    s = Store(str(pathlib.Path(tempfile.mkdtemp()) / "d.db"))
    p = "qq:private:1"
    for d in range(5):
        ts = int(time.time()) - d * 86400
        curiosity.record(s, p, "zero_read", r"C:\All\Work\VibeCoding\DVC\docs\a.md", ts)
        curiosity.record(s, p, "zero_grep", "cold-vault", ts)
    curiosity.mark_spoke(s, p, "Vibecoding/DVC")
    print(curiosity.interests(s, p))
    print(curiosity.line(s, p))


if __name__ == "__main__":               # pragma: no cover
    _demo()
