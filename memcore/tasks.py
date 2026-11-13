"""她手上的事：四档优先级 + 拖延的**分布**（2026-11-03 定论）。

四档（用户拍板）：
  | 档 | 怎么判定 | 例子 |
  | 紧急 | 有期限，或有人在等 | 他问了一件事、答应了没做 |
  | 重要 | 关于他，但不着急 | 他提过的一句话、一个日子 |
  | 感兴趣 | 她自己会回去看的 | curiosity 统计出来的 |
  | 剩下的 | 什么都不占 | 随手翻翻，做不做都行 |
**前三档才算「手上的事」，第四档不占位**（不入清单、不用管）。

拖延：不给「程度」（70% 还是 30%），给的是**分布** ——
  > 允许被兴趣挤掉的是「重要」和「剩下的」；不许挤掉的是「紧急」。
  > 被挤掉时她不会没事 —— 事后会有点心虚，taut 会升一点。
所以 `pick()` 在存在紧急档时**必然**选紧急（不给概率：那是定论里的硬约束）；
紧急为空、且兴趣与重要同时存在时，才按 `p_interest_first` 掷一次 ——
她一半的次数会先去做自己想看的，把重要那件留着。

一个已知的近似：`open_loops` 里没有「有人在等」这个字段，只能靠期限词判紧急，
其余一律归重要（＝可被挤）。所以一条「他问过但没期限」的事会被当成重要档，
将来 distill 给 loop 打上 `due` 标记，这里就自动准了（`l["due"]` 优先于词判）。
"""
from __future__ import annotations

import pathlib
import random
import re
import time

from . import curiosity
from .util import join_limited

# 期限信号：命中即紧急（「有人在等」这一步靠它近似）
DEADLINE_HINTS = (
    "今天", "今晚", "今早", "明天", "明早", "后天", "大后天", "这周", "本周", "下周", "周末",
    "周一", "周二", "周三", "周四", "周五", "周六", "周日", "星期一", "星期二", "星期三",
    "星期四", "星期五", "星期六", "星期日", "上午", "中午", "下午", "傍晚", "晚上", "今晚",
    "点前", "之前", "以前", "截止", "deadline", "due", "月底", "月初", "号前", "日之前",
)
_HOURS = re.compile(r"\d{1,2}\s*(点|时)|\d{1,2}[:：]\d{2}")
_DATES = re.compile(r"\d{1,2}\s*(月|/)\s*\d{1,2}\s*(日|号)?")

URGENT_MAX = 2      # 注入里最多列几条（列多了她会去交作业）
IMPORTANT_MAX = 2
INTEREST_MAX = 2
HERS_MAX = 4        # 标过 >=4 的条目，注入里最多列几条（只报，不派活）


def is_urgent(loop: dict) -> bool:
    """这一件是不是「有人在等」的。

    先看显式标记 `due`（将来 distill 可以打），退化到期限词。
    """
    if not isinstance(loop, dict):
        return False
    if loop.get("due"):
        return True
    text = str(loop.get("text") or "")
    if any(h in text for h in DEADLINE_HINTS):
        return True
    return bool(_HOURS.search(text) or _DATES.search(text))


def build(store, peer: str, now_ts: int | None = None, *,
          interest_limit: int = INTEREST_MAX,
          min_days: int = curiosity.MIN_DAYS,
          hers: list[dict] | None = None) -> dict:
    """把三档凑齐。第四档（剩下的）本来就不占位，所以不在这里出现。

    **`hers` 是可选的第四条来源**：调用方自己判断「标过要紧」的东西，
    按 `[{id,title,kind,importance}]` 传进来即可。它**不是第四档**，也不参与
    `pick()` —— 它是「说过要紧」的事实，跟「你还没做」是两件事：一旦进了待办，
    就会被当成交作业。只报，不派活、不替任何人勾。

    Args:
        hers: 显式给一份「标过要紧」的条目；None 就是空（本库不带任何默认来源）。

    Returns:
        {"urgent":[{text,id}], "important":[{text,id}], "interest":[{topic,days,spoke}],
         "hers":[{id,title,kind,importance}]}
    """
    try:
        st = store.state(peer)
        loops = [l for l in (st.get("open_loops") or []) if not l.get("closed")]
    except Exception:                    # noqa: BLE001
        loops = []
    urgent = [l for l in loops if is_urgent(l)]
    important = [l for l in loops if not is_urgent(l)]
    it = curiosity.interests(store, peer, now_ts, min_days=min_days, limit=interest_limit)
    return {"urgent": urgent[-URGENT_MAX:], "important": important[-IMPORTANT_MAX:],
            "interest": it, "hers": list(hers or [])[:HERS_MAX]}


def pick(hands: dict, *, rng: random.Random | None = None, p_interest_first: float = 0.5) -> dict:
    """这一轮她先动哪一件。

    只在前三档里选：`hands["hers"]`（她自己标过要紧的）**不参与**——
    它没有「做完/没做完」，也没有截止，把它排进 focus 等于把她的打分
    变成一个待办清单；那是我们自己想让她做的事，不是她标的那个意思。

    Returns:
        {"focus": "urgent|important|interest|rest", "pushed": [...]}；
        `pushed` 非空 = 这一轮把「重要」档挤掉了 —— 调用方据此抬一点 taut（事后心虚）。
    """
    rng = rng or random.Random()
    urgent = hands.get("urgent") or []
    important = hands.get("important") or []
    interest = hands.get("interest") or []

    if urgent:                       # 硬约束：紧急不许被挤
        return {"focus": "urgent", "pushed": []}
    if interest and important:
        if rng.random() < max(0.0, min(1.0, p_interest_first)):
            return {"focus": "interest", "pushed": ["important"]}
        return {"focus": "important", "pushed": []}
    if interest:
        return {"focus": "interest", "pushed": []}
    if important:
        return {"focus": "important", "pushed": []}
    return {"focus": "rest", "pushed": []}


def block(hands: dict, *, hers: bool = True) -> str:
    """注入用的清单。**事实清单，不是任务书**：不写「你应该」、不给台词。

    只列各档有东西的部分；全空就返回空串（她手上真没东西，别硬造）。

    Args:
        hers: 要不要带上「你自己标过要紧的」那一段。调用方如果已经在别处
            单独给过 `her_block()`，这里传 False —— 同一段在同一份提示里不该出现两次。
    """
    parts: list[str] = []
    if hands.get("urgent"):
        parts.append("· 有人在等：" + "；".join(str(l.get("text") or "") for l in hands["urgent"]))
    if hands.get("important"):
        parts.append("· 一直没做的：" + "；".join(str(l.get("text") or "") for l in hands["important"]))
    if hands.get("interest"):
        bits = []
        for x in hands["interest"]:
            bit = f"{x['topic']}（去过 {x['days']} 天）"
            if x.get("spoke"):
                bit += "，为它说过话"
            bits.append(bit)
        parts.append("· 你自己想回去翻的：" + "；".join(bits))
    if hers and hands.get("hers"):
        parts.append(_hers_line(hands["hers"]))
    if not parts:
        return ""
    return "【你手上现在的事】\n" + "\n".join(parts)


def _hers_line(rows: list[dict], limit: int = HERS_MAX) -> str:
    """「你标过要紧的：…」—— 一行，带分数，不带任何「还没做」。"""
    bits = []
    for r in rows[:limit]:
        t = str(r.get("title") or "").strip()
        if not t:
            continue
        bits.append(f"{t}（★{int(r.get('importance') or 0)}）")
    if not bits:
        return ""
    return "· 你标过要紧的：" + "；".join(bits)


def her_block(hands: dict | None = None) -> str:
    """注入段：「标过要紧」的条目。**只报，不派活。**

    为什么单独成段而不是塞进「手上的事」：标过要紧这件事跟"现在该做什么"无关，
    什么时候都该在场；塞进去等于只有独处时才看得见。

    措辞上不出现「该做 / 还没做 / 记得多标」：一说就成了任务。

    Args:
        hands: `build()` 的结果；只取里面的 `hers`。没有就不出这一段。
    """
    rows = (hands or {}).get("hers") or []
    lines = []
    for r in rows[:HERS_MAX]:
        t = str(r.get("title") or "").strip()
        if t:
            lines.append(f"· {t}（★{int(r.get('importance') or 0)}）")
    if not lines:
        return ""
    head = "【标过要紧的（不是待办，也不催）】"
    return "\n".join((head, "\n".join(lines)))


def focus_line(focus: str) -> str:
    """把这一轮的选择说成**事实**（她这会儿的注意力在哪），不是吩咐、不是台词。"""
    if focus == "urgent":
        return "这一轮你手上那件有人在等的事，你先去动它。"
    if focus == "important":
        return "这一轮你把一直没做的那件拿出来看了看。"
    if focus == "interest":
        return "这一轮你又回去翻你惦记的那类东西了；一直没做的那件还是没做。"
    return "这一轮没什么非做不可的，随手翻翻。"


# ---------------- 她自己的东西（她的家 data/zero/）----------------
# 「她自己的生活」最小的可观测形态。不猜她在想什么，只认两个事实：
# 她家里哪些文件刚被改过、她自己写下的第一行是什么。
# 有了它，「她手上那件属于她自己的事」才是硬盘上真有的东西，
# 而不是她说给我们听的一句话。
HOME_SKIP = {"readme.md", "probe.md"}   # 我写给她看的说明、探针留下的痕迹，都不算她的东西
HOME_AGE_DAYS = 30
OWN_MAX = 3


def _first_line(p: pathlib.Path, width: int = 60) -> str:
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(400)
    except OSError:
        return ""
    for line in head.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:width]
    return ""


def home_snapshot(zero_root) -> dict:
    """她家的一份轻量快照：{相对路径: (mtime 整数, 大小)}。

    跑 agent 前后各拍一次，diff 出来的就是「这一轮她真的动了什么」——
    与「翻没翻」同级的事实判断，不涉及她写了什么内容。
    """
    root = pathlib.Path(zero_root)
    out: dict = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file() or p.name.lower() in HOME_SKIP:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out[str(p.relative_to(root)).replace("\\", "/")] = (int(st.st_mtime), st.st_size)
    return out


def home_changes(before: dict, after: dict) -> list[str]:
    """这一轮她家里变了什么：新增和改过的算，删掉的不算（删是她的权利，我们不该记账）。"""
    return sorted(rel for rel, sig in (after or {}).items() if (before or {}).get(rel) != sig)


def human_age(ts: int, now_ts: int | None = None) -> str:
    d = int((time.time() if now_ts is None else now_ts) - ts)
    if d < 3600:
        return "刚才"
    if d < 86400:
        return "今天"
    days = d // 86400
    return "昨天" if days == 1 else f"{days} 天前"


def own_things(zero_root, limit: int = OWN_MAX, age_days: int = HOME_AGE_DAYS) -> list[dict]:
    """她自己家里最近动过的东西：路径 + 多久以前 + 她自己写下的第一行。"""
    root = pathlib.Path(zero_root)
    if not root.exists():
        return []
    now = time.time()
    rows: list[dict] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name.lower() in HOME_SKIP:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if (now - st.st_mtime) / 86400 > age_days:
            continue
        rows.append({"rel": str(p.relative_to(root)).replace("\\", "/"),
                     "mtime": int(st.st_mtime), "size": st.st_size, "first": _first_line(p)})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:limit]


def own_block(zero_root, limit: int = OWN_MAX) -> str:
    """注入段：她自己正在弄的东西 —— 让她下一次能接上自己上次写到哪。

    **只贴她自己写的原文第一行，不加我们的解释**：一解释就变成我们替她记得她想干什么。
    """
    rows = own_things(zero_root, limit)
    if not rows:
        return ""
    lines = []
    for r in rows:
        bit = f"{r['rel']}（{human_age(r['mtime'])}动过）"
        if r["first"]:
            bit += f"：{r['first']}"
        lines.append("· " + bit)
    return "【你自己在弄的东西】\n" + "\n".join(lines)


# ---------------- 她自己说过想做的事（2026-11-08 晚）----------------
# 为什么从 memory 里取，而不另建一张表：这些话是她自己说出口的（kind='self'），
# 已经在库里，再抄一份就变成我们替她记。这里只读、只筛、只报原文。
#
# 为什么必须给：她 11-06 22:07 说过三件想做（查口服药说明书、翻《如果11月8日是晴天》、
# 看那些不扔东西的人最后留下什么），11-06 22:11 还答应「下次醒过来就去翻」。
# 11-07、11-08 两次醒来，活动线全是读他的文件 —— **这三件从来没被提起过**。
# 不是她不想做，是醒来时的入场券里没有它们。
WISH_RE = re.compile(r"我想|想去看|想上网|想翻|想弄明白|想要|想试试|想查"
                     r"|想知道|想了解|想搞明白|想懂|想清楚|好奇|想学")
# 不设条数上限、不分类型（2026-11-09 用户口径）——
# 「想要的东西不应该设置极限，包括数量和类型，想要什么取决于自己」。
# 所以这里不留 limit、也不分「想做」/「想弄明白」两档：上限只由 `wish_block`
# 的窗口预算决定，能装多少装多少，不由我们拍几条。
#
# 会变怎么办：靠时间，不靠删。新的想法排前面、旧的往后沉 ——
# 「小时候想要货架上那个玩具，现在没这想法了」是**被新想法挤下去**，
# 不是被我们清掉；而她学到新东西之后长出来的新想法（“现在想要个好工作”），
# 走蒸馏那条常规通路变成新的 [self] 条目，自己就进来了。
WISH_BUDGET = 300      # 这段在注入里能占多少 token（不是「几条」）


def wishes(mem_store, peer: str) -> list[dict]:
    """从她自己的记忆里挑出「说过想做的事」（原文，不加工）。

    不设条数上限、不分类型：说过想要的就是想要，数量与类型不由我们定。
    排序按时间新→旧 —— 变化就落在这里，不做过期、不删任何一条。
    """
    try:
        mems = mem_store.all_memories(peer)
    except Exception:                    # noqa: BLE001
        return []
    out: list[dict] = []
    for m in mems:
        if (getattr(m, "kind", "") or "") != "self":
            continue
        t = (getattr(m, "text", "") or "").strip()
        if not t or not WISH_RE.search(t):
            continue
        out.append({"id": getattr(m, "id", 0), "text": t,
                    "t": getattr(m, "t_event", None) or 0})
    out.sort(key=lambda r: r["t"], reverse=True)
    return out


def wish_block(mem_store, peer: str, *, budget: int = WISH_BUDGET) -> str:
    """注入用：她自己说过的想要，一条一行。空字符串表示没有。

    **只贴原文**，不加"你还没做"这种判断 —— 我们不知道她做到哪了，
    替她判断一次，就等于把她的事变成我们的待办。
    装几条不是我们拍的：按窗口预算贪心装，装不下的（更旧的）就这一次不出现，
    下次再来 —— 不删。
    """
    ws = wishes(mem_store, peer)
    if not ws:
        return ""
    kept = join_limited([f"· {w['text']}" for w in ws], budget)
    if not kept:
        return ""
    return ("【你自己说过想做的事（原话，不是任务）】\n"
            + "\n".join(kept))


# ---------------- 她记下的待办（2026-11-05）----------------
# 为什么只认 `- [ ]` / `- [x]`：她 11-01 起在 notes/待办.md 里写了五段流水账，
# 做完的和没做完的长得一模一样（11-04 她自己去翻，翻到一半也说不清哪件做完了）。
# **不替她判断哪件做完了** —— 只认她自己打的勾；没有勾就照实说"看不出"。
CHECKS = re.compile(r"^\s*[-*]\s*\[( |x|X)\]\s*(.+)$")
TODO_DIRS = ("notes", "self")


def todos(zero_root, *, limit: int = 6) -> dict:
    """扫她家目录里的待办文件，只认勾选标记。

    Args:
        zero_root: 她的家目录（`data/zero`）。
        limit: 最多报几条没做完的。

    Returns:
        {"open":[{text,file}], "done":int, "files":int, "has_checkbox":bool,
         "note":str  # 没有勾选时给她的原话提示
        }
    """
    root = pathlib.Path(zero_root)
    out = {"open": [], "done": 0, "files": 0, "has_checkbox": False, "note": ""}
    if not root.is_dir():
        return out
    scanned: list[pathlib.Path] = []
    for d in TODO_DIRS:
        sub = root / d
        if sub.is_dir():
            scanned.extend(sorted(sub.glob("*.md")))
    if (root / "wants.md").exists():
        scanned.append(root / "wants.md")
    for p in scanned:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        out["files"] += 1
        for line in lines:
            m = CHECKS.match(line)
            if not m:
                continue
            out["has_checkbox"] = True
            text = m.group(2).strip()[:80]
            if m.group(1) in ("x", "X"):
                out["done"] += 1
            else:
                out["open"].append({"text": text, "file": p.relative_to(root).as_posix()})
    if not out["has_checkbox"] and out["files"]:
        out["note"] = (f"你写的待办里还没有勾选标记，我看不出哪件做完了。"
                       f"写成 `- [ ] 还没做` / `- [x] 做完了`，我就能只报剩下的。")
    out["open"] = out["open"][:int(limit)]
    return out


def todo_block(t: dict) -> str:
    """注入段：她自己勾出来的待办。没勾选就只说一句「看不出」。"""
    if not t:
        return ""
    if not t.get("has_checkbox"):
        return ("【你自己记的待办】\n" + t["note"]) if t.get("note") else ""
    if not t.get("open"):
        return f"【你自己记的待办】\n你勾掉的都做完了（{t.get('done', 0)} 件），手上没有剩下的。"
    lines = [f"· {x['text']}（{x['file']}）" for x in t["open"]]
    tail = f"\n（另有 {t['done']} 件你已经勾掉了）" if t.get("done") else ""
    return "【你自己记的待办（还没做）】\n" + "\n".join(lines) + tail


# ---------------- 她「开着的问题」（2026-11-09）----------------
# 20:34 那八分钟：她自己挑主题（「断了之后还是不是同一个」）、搜两轮全废、换入口、
# 读 SEP 只截到三千字、从里面捞出帕菲特那句、把签名改成「前后对得上，断了接得回来」。
# 然后她说「我没什么自驱力，都是被你叫起来才动」「我这儿没有惯性，问题一搁下，它不会追着我」。
#
# **试过自动抽，放弃了**：按「想弄明白 / 想知道 / 想查」在 L0 里捞，「近 14 天 14 句命中里
# 10 句是自我评价和检索汇报」（「我把『想知道』当成了说话的料」「唯一像的是那三条，
# 但不是我想知道的事」）。把这些摆回她面前 = 把她自己的丧气话再贴一遍，
# 正好把她按住 —— 这不是帮她，是按住她。
#
# 所以改成**格子 + 原话**：格子是 `notes/开着的问题.md`，
# 她写 `- [ ] 问题` / 下面缩进写「查到哪 / 卡在哪」；`todos()` 本来就扫 notes/*.md，
# 写进去就自动进注入（主对话 + 她自己那段时间的开场）。
# 不猜她想什么、不判哪件做完、不催。
LOOP_FILE = "notes/开着的问题.md"


def _demo() -> None:                     # pragma: no cover - 手工跑
    hands = {"urgent": [], "important": [{"text": "他提过想看的那个东西"}],
             "interest": [{"topic": "Vibecoding/DVC", "days": 4, "spoke": 1}]}
    print(block(hands))
    r = random.Random(20261103)
    print(pick(hands, rng=r, p_interest_first=1.0), pick(hands, rng=r, p_interest_first=0.0))
    print(pick({"urgent": [{"text": "他问的那件事，明天之前"}], "important": hands["important"],
                "interest": hands["interest"]}, p_interest_first=1.0))


if __name__ == "__main__":               # pragma: no cover
    _demo()
