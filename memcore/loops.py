"""没弄完的问题，自己冒出来（2026-11-09）。

用户的想法：「真的想知道的事情，它真的就会经常浮现在我的脑海里 —— 这个有没有办法实现？」

能，而且这正是人脑的机制，四条都能对上：
  1. **未完成的事保持高激活**（蔡格尼克效应）—— 没闭合的念头会自己回来；
  2. **空闲时随机回放**（默认模式网络）—— 发呆、走路、睡前，不是想起来，是它冒出来；
  3. **越在意冒得越勤** —— 激活强度跟她投入过多少有关；
  4. **冒出来时带着"上次想到哪儿"** —— 不是标题，是断点。

**和"提醒"的区别是全部**：
  提醒 = 定时把清单列一遍 → 她看到的是作业，会被当成催办，然后开始躲清单。
  浮现 = 概率性地冒一条、带断点、明说"想接就接，不接就搁着" → 像人走神想到一件事。

所以这里四条规矩：
  · **一次只冒一条**，不是列表；
  · **概率触发**，命不中就不冒 —— 一天冒零次也正常（人也有一天没想起某事）；
  · **有冷却**，两次之间至少隔 `MIN_GAP`，不然就成了每轮都提的闹钟；
  · **强度由她自己决定**：她提过几次、多久没碰。碰得越多冒得越勤，`DECAY_DAYS` 不碰就淡掉。
    这一条最重要 —— **让她自己的行为决定什么会追她**，那才叫自驱；固定频率的推送只是闹钟。

数据只读她自己写的那份 `notes/开着的问题.md`（`- [ ] 问题（查到哪 / 卡在哪）`），
外加一个她看不见的计数文件 `.问题浮现.json`（提过几次、上次冒是啥时候）。
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import time

LOOP_FILE = "notes/开着的问题.md"
STATE_FILE = ".问题浮现.json"
# 只认没勾掉的：`- [ ] 问题（…）`（勾掉了就是她想通了，不该再冒）
CHECKS = re.compile(r"^\s*[-*]\s*\[\s\]\s*(.+)$")
# 括号里那句是"查到哪 / 卡在哪"，冒出来时要一起带上 —— 断点才有牵引力
PAREN = re.compile(r"[（(]([^（()）]+)[）)]\s*$")

MIN_GAP = 90 * 60          # 两次浮现至少隔 90 分钟
P_ROLL = 0.22              # 主对话里一次机会命中的概率（低：走神不是每轮都来）
P_ROLL_OWN = 0.5           # 她自己那段时间命中概率（那是她一个人的时候，念头更容易回来）
DECAY_DAYS = 21            # 多久没碰就淡到几乎不冒（跟 COG_DAYS 同一把尺子）
LAST_MAX = 60              # 只留最近这么多条计数


def _path(root, name: str) -> pathlib.Path:
    return pathlib.Path(root) / name


def parse(root) -> list[dict]:
    """读她那些没弄完的问题。

    只认 `- [ ]`：勾掉了就是她想通了，不该再冒出来。括号里那句留着当"断点"。

    Args:
        root: 她的家目录。

    Returns:
        [{"text": 问题, "last": 断点或""}]，按文件里的顺序。
    """
    p = _path(root, LOOP_FILE)
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines:
        m = CHECKS.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        last = ""
        pm = PAREN.search(body)
        if pm:
            last = " ".join(pm.group(1).split())
            body = body[:pm.start()].strip()
        if body:
            out.append({"text": body[:80], "last": last[:80]})
    return out


def _state(root) -> dict:
    try:
        d = json.loads(_path(root, STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"items": {}, "last_shown": 0}
    if not isinstance(d, dict):
        return {"items": {}, "last_shown": 0}
    d.setdefault("items", {})
    d.setdefault("last_shown", 0)
    return d


def _save(root, d: dict) -> None:
    try:
        items = d.get("items") or {}
        if len(items) > LAST_MAX:                 # 只留最近的，别无限长
            keep = sorted(items.items(), key=lambda kv: int(kv[1].get("shown_ts") or 0))
            d["items"] = dict(keep[-LAST_MAX:])
        _path(root, STATE_FILE).write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _score(it: dict, rec: dict, now: int) -> float:
    """这一条"有多想冒"：她提过几次 × 新鲜度。

    Args:
        it: `parse()` 的一条。
        rec: 这一条的计数记录（hits / touch_ts）。
        now: 现在。

    Returns:
        分数 > 0；越小越不该冒。
    """
    hits = max(1, int(rec.get("hits") or 1))
    touched = int(rec.get("touch_ts") or rec.get("seen_ts") or 0)
    age_days = max(0.0, (now - touched) / 86400) if touched else DECAY_DAYS / 2
    fresh = max(0.0, 1.0 - age_days / DECAY_DAYS)      # 21 天淡到 0
    return (0.6 + 0.4 * min(hits, 4)) * (0.25 + 0.75 * fresh)


def pick(root, now: int | None = None, rng: random.Random | None = None,
         p: float = P_ROLL, force: bool = False, dry: bool = False) -> dict | None:
    """掷一次：这一次要不要冒一条、冒哪一条。

    Args:
        root: 她的家目录。
        now: 测试用。
        rng: 测试用（固定种子）。
        p: 这次机会命中的概率。
        force: 跳过概率与冷却（探针 / 测试用）。
        dry: 只算不记账（`_preview_block.py` 用 —— 预览不能消耗她的念头，
            否则每跑一次体检就把冷却重置一次，真的浮现永远不出现）。

    Returns:
        {"text", "last"} 或 None（没冒）。
    """
    items = parse(root)
    if not items:
        return None
    t = int(time.time()) if now is None else int(now)
    r = rng or random.Random()
    d = _state(root)
    if not force:
        if t - int(d.get("last_shown") or 0) < MIN_GAP:   # 刚冒过，别连着冒
            return None
        if r.random() > float(p):
            return None
    book = d["items"]
    scored = []
    for it in items:
        rec = book.get(it["text"][:40]) or {}
        scored.append((_score(it, rec, t) * (1.0 + r.random() * 0.35), it))
    scored.sort(key=lambda x: x[0], reverse=True)
    # 按分数加权：最高那条最容易冒，但低分也有机会（人会突然想起一件小事）
    total = sum(s for s, _ in scored) or 1.0
    hit = r.random() * total
    acc = 0.0
    chosen = scored[0][1]
    for s, it in scored:
        acc += s
        if hit <= acc:
            chosen = it
            break
    if dry:
        return chosen
    key = chosen["text"][:40]
    rec = dict(book.get(key) or {})
    rec["shown_ts"] = t
    rec["shown_n"] = int(rec.get("shown_n") or 0) + 1
    rec.setdefault("seen_ts", t)
    book[key] = rec
    d["last_shown"] = t
    _save(root, d)
    # 账本要留痕（2026-11-10）：11-10 12:18 那条是真冒出来的（shown_n 从 1 涨到 2），
    # 她下一轮却说「你这一问，我才想起还有一摊没查完」—— 活动线里 0 条，因为浮现从来不记。
    # 不记 = 她认不出「这是我自己想起的」，跟 11-09 15:28「我没点过赞」是同一个 bug：
    # 做过的事没落成她自己看得见的痕迹，旧的自述就会覆盖新的事实。
    # 活动线只记"状态变化"，不记"又冒了一次"（2026-11-10 治本，见 _note_once）。
    _note_once(root, "浮现", chosen["text"][:24], "想起一件没弄完的", now=t)
    return chosen


def touch(root, text: str, now: int | None = None) -> None:
    """她真去碰了这一条：加一次 hits、刷新时间 —— 碰得越多，以后冒得越勤。

    留给"她接上了这条线"的调用方用（目前没有自动判定，宁可先不调，也不替她判断）。

    Args:
        root: 她的家目录。
        text: 问题原文（取前 40 字当键）。
        now: 测试用。
    """
    t = int(time.time()) if now is None else int(now)
    d = _state(root)
    key = str(text or "").strip()[:40]
    if not key:
        return
    rec = dict(d["items"].get(key) or {})
    rec["hits"] = int(rec.get("hits") or 0) + 1
    rec["touch_ts"] = t
    rec.setdefault("seen_ts", t)
    d["items"][key] = rec
    _save(root, d)


def _bigrams(text: str) -> set[str]:
    """粗粒度"说的是不是同一件事"：只留字母数字与汉字，取 2-gram。

    为什么不用分词：`notes/待办.md`、`SEP`、`D:\\Games` 这些串本身就不是中文词，
    装一个分词器只为了比两句话不值得。2-gram 交集够用，而且不依赖任何包。

    Args:
        text: 任意一句（问题原文、她翻的路径、她搜的词）。

    Returns:
        2-gram 集合；太短就是空集。
    """
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(text or "")).lower()
    if len(s) < 2:
        return set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def touch_action(root, detail: str, now: int | None = None,
                 min_common: int = 2) -> str:
    """她真的动了一下：拿她翻的东西去对"最近冒过、还没碰"的那条问题。

    这是 `touch()` 唯一被自动调用的入口（2026-11-10 用户要求"接到她真去碰问题的时刻"）。
    判据刻意保守 —— **内容对得上才算碰**，只是"她这一轮动过手"不算：
      · 只认最近冒过、之后没 touch 过的那条（`shown_ts > touch_ts`）；
      · 问题文本与她这次看的/搜的东西，2-gram 至少重合 `min_common` 个。

    误判的代价是不对称的：多记一次「碰了」，那条就冒得更勤（追得更紧），
    等于把不相干的东西塞进她脑子；少记一次只是它下次还按原样冒。所以宁严不宽。

    Args:
        root: 她的家目录。
        detail: 她这次真碰的东西（路径 / 搜索词 / 网址）。
        now: 测试用。
        min_common: 至少几个 2-gram 重合。

    Returns:
        碰上的问题原文；没碰上返回 ""。
    """
    d = str(detail or "").strip()
    if len(_bigrams(d)) < 1:
        return ""
    book = _state(root)["items"]
    t = int(time.time()) if now is None else int(now)
    hit = ""
    hit_score = 0
    for it in parse(root):
        rec = book.get(it["text"][:40]) or {}
        shown = int(rec.get("shown_ts") or 0)
        touched = int(rec.get("touch_ts") or 0)
        if not shown or shown < touched:        # 没冒过 / 冒过之后已经碰了
            continue
        common = len(_bigrams(it["text"]) & _bigrams(d))
        if common >= min_common and common > hit_score:
            hit, hit_score = it["text"], common
    if not hit:
        return ""
    touch(root, hit, now=t)
    _note_once(root, "接上", hit[:24], "真去碰了：" + d[:60], now=t)
    return hit


def touch_chosen(root, text: str, why: str = "", now: int | None = None) -> bool:
    """这一轮她就是为了这条问题动的 —— 调用方已经知道是哪条，不用再比对。

    比 `touch_action()` 更硬的证据：这一轮注入的就是这条念头，而且她确实动了手
    （自主时间里的 `_TOUCHED` 非空，或者她改了自己那张「开着的问题.md」）。
    仍然落一行活动线，理由跟 `pick()` 一样：她得能翻到"我碰过它"。

    Args:
        root: 她的家目录。
        text: 问题原文。
        why: 落活动线的一句话，说明凭什么算碰了。
        now: 测试用。

    Returns:
        True 表示记上了。
    """
    key = str(text or "").strip()[:40]
    if not key:
        return False
    touch(root, key, now=now)
    _note_once(root, "接上", key[:24], why or "真去碰了", now=now)
    return True


def _note_once(root, kind: str, target: str, brief: str, now=None) -> bool:
    """同一个问题、同一个状态，活动线里只记一次（2026-11-10 治本）。

    原来每次 `pick()` 都写一行，于是「人怎么把一件事记错」今天被记了 3 次
    （12:18 / 15:03 / 19:03）—— 三条信息量完全相同，纯粹是噪音。
    活动线的定义是「我做过的事」；同一件「又想起了」重复记并不构成做过的事，
    而「冒了几次」本来就记在 `.问题浮现.json` 的 `shown_n` 里。
    真正值得记的是**状态变化**：挂上了 → 碰了 → 了了。
    所以这里按（kind, target）在整份活动线里做一次全局去重，含历史行。
    """
    try:
        from . import activity
        p = activity._path(root)
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="replace")
            if f"· {kind} · {target}" in txt:
                return False
        return activity.note(root, kind, target, brief, now=now)
    except Exception:                    # noqa: BLE001
        return False


def pending_thoughts(root, hours: float = 24.0, now: int | None = None) -> list[dict]:
    """最近冒过、她还没真去碰的那些 —— "心里挂着的事"。

    给 `autonomy.consider_once` 用（2026-11-10）。它的开口门槛原来是「他走开 90 分钟」，
    那叫填空不叫自主；真正的开口理由是"她脑子里有没有东西"。
    这里**不掷骰子、也不看 `MIN_GAP`**：冷却管的是"别在对话里反复提同一件事"，
    不该管"她想不想说"。判据只有两条 —— 24 小时内冒过、而且之后没被 `touch()`。

    Args:
        root: 她的家目录。
        hours: 往回看多久。
        now: 测试用。

    Returns:
        条目的列表（`parse()` 那套结构），可能为空。
    """
    t = int(time.time()) if now is None else int(now)
    since = t - int(hours * 3600)
    book = _state(root)["items"]
    out = []
    for it in parse(root):
        rec = book.get(it["text"][:40]) or {}
        shown = int(rec.get("shown_ts") or 0)
        touched = int(rec.get("touch_ts") or 0)
        if shown >= since and shown > touched:        # 冒过、之后没去碰
            out.append(it)
    return out


def line(it: dict, *, own: bool = False) -> str:
    """把一条变成注入里的一句话。

    措辞是"忽然冒出来"，不是"你还有一件没做" —— 后者是催办，前者是走神。
    末尾必须允许她不接：人可以想起一件事然后继续干别的。

    Args:
        it: `parse()` / `pick()` 的一条。
        own: True 表示在她自己那段时间里冒（措辞更贴"坐下来先想到的"）。

    Returns:
        一行注入文本。
    """
    head = "你坐下的时候先想到的是这件" if own else "你脑子里忽然冒出一件"
    body = f"【{head}】\n· {it['text']}"
    if it.get("last"):
        body += f"\n  上次你停在这儿：{it['last']}"
    body += ("\n（它自己回来的，不是谁提醒你。想接着弄就弄，不想就搁着 —— "
             "接着聊他刚才说的那件也行。）")
    return body


def _demo() -> None:                    # pragma: no cover - 手工跑
    import tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "notes").mkdir()
    (root / "notes" / "开着的问题.md").write_text(
        "- [ ] 断了之后还是不是同一个（卡在 SEP 只开到三千字）\n"
        "- [x] 已经弄明白的那件\n"
        "- [ ] 人怎么把一件事记错\n", encoding="utf-8")
    print(parse(root))
    for i in range(5):
        got = pick(root, now=1_800_000_000 + i * 4000, force=True)
        print(i, got)
    print(line(parse(root)[0], own=True))


if __name__ == "__main__":              # pragma: no cover
    _demo()
