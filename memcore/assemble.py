"""上下文装配：每轮都从零拼一个 prompt。

这就是回答「要不要杀掉他再打包」的地方 —— 不打包、不迁移。
"新会话"这件事被降级成：每轮重新装配一次，所以它一直在发生，也就无所谓发生。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .distill import state_line
from .recall import (Hit, detour, gap_line, human_when, last_session_tail,
                     now_line, recall_messages, recent_self_lines, retrieve, user_style)
from .store import Store
from .util import clip, est_tokens

# 默认 token 预算（可被 config 覆盖）。总预算压得低，是为了 QQ 场景的响应速度。
BUDGET = {
    "persona": 1150,    # 种子版人设约 1000 字：短是刻意的，长了就没地方长
    "now": 40,          # 现在是几号几点：没有这个锚，相对时间词全是猜
    "state": 120,
    "anchor": 240,      # 恒定注入："不能忘的"，不依赖本轮提问
    "self": 180,        # 她自己的立场：让"她是谁"前后对得上
    "detour": 80,       # 忽然想到的：让话别那么贴题
    "gap": 90,          # 断过一段又接上：中间记录没停，这是事实
    "last": 300,        # 上次说话的最后几句：隔夜接话要有落点，不能只靠检索
    "mine": 300,        # 她自己刚说过的：历史里可能没有，L0 一定有
    "style": 110,       # 他平时怎么打字：风格从样本里学，不写死
    "recall": 900,
    "summary": 700,
    "recent": 1200,
    "evidence": 0,      # 原话佐证槽，默认关（打开会显著变长但更"有据"）
}


def pick_self(store: Store, peer: str, k: int = 4,
              avoid: list[str] | None = None) -> list[str]:
    """她自己说过的立场 + 理由。

    为什么单独成槽：这些是"她这个人"的骨架，不该跟"关于对方的事实"抢同一个名额，
    也不该只在被问到相关话题时才浮上来。前面被想起过、且离现在近的优先。

    没有理由的立场会如实标出来（「当时没给理由」）。这不是提示她补，是不替她圆：
    她当初确实只说了一句「我不喜欢」，问起来时她该知道自己手上只有半句话，
    而不是临场编一个像样的理由 —— 编出来的理由比没有更容易把后来长歪的方向锁死。

    `avoid` 是她最近几轮说过的话：讲过的不再送进去。
    不这么做的后果是复读 —— 同一个意象（"我这边是空的"）被每轮喂进去，
    她每轮都拿它解释一切，听起来像在背设定。
    """
    from .recall import _jaccard
    mems = [m for m in store.all_memories(peer) if m.kind == "self"]
    if not mems:
        return []
    mems.sort(key=lambda m: (m.active_ts(), m.hit_count), reverse=True)
    said = [t for t in (avoid or []) if t]
    out: list[str] = []
    for m in mems:
        if len(out) >= k:
            break
        if said and max((_jaccard(m.text or "", t) for t in said), default=0.0) > 0.45:
            continue        # 最近说过了，这轮换别的
        why = f"（{m.why}）" if m.why else "（当时没给理由）"
        out.append(f"{m.text}{why}")
    return out


def pick_anchors(store: Store, peer: str, k: int = 5) -> list[str]:
    """锚点：不管这轮问什么，都必须随身的几条。

    这是"人不是只在被问到时才想起你"的实现 —— 边界、未完成的承诺、
    高情绪事件、未闭合话题，每轮都在场。
    """
    mems = store.all_memories(peer)
    st = store.state(peer)
    loops = [l for l in st.get("open_loops", []) if not l.get("closed")]
    picked: list[str] = []
    seen: set[int] = set()

    def take(m):
        if m.id in seen or len(picked) >= k:
            return
        seen.add(m.id)
        when = human_when(m.t_event)
        picked.append(f"[{m.kind}]{(when + ' ') if when else ''}{m.text}")

    # 1) 边界永远在场（最多 2 条，按显著度）
    for m in sorted([m for m in mems if m.kind == "boundary"],
                    key=lambda x: -x.salience)[:2]:
        take(m)
    # 2) 未闭合承诺
    for m in sorted([m for m in mems if m.kind == "promise"], key=lambda x: -x.salience)[:1]:
        take(m)
    # 3) 高显著度（创伤/重大事件/强偏好）
    for m in sorted(mems, key=lambda x: -x.salience):
        if m.salience >= 2.3:
            take(m)
    # 4) 未闭合话题原文
    for l in loops[:2]:
        if len(picked) < k:
            picked.append(f"[open] {clip(l['text'], 60)}")
    return picked


@dataclass
class Assembled:
    messages: list[dict[str, str]]
    report: dict = field(default_factory=dict)

    def debug(self) -> str:
        r = self.report
        lines = [f"总 token ≈ {r.get('total', 0)} / 预算 {r.get('budget_total', 0)}"]
        for k, v in r.get("slots", {}).items():
            lines.append(f"  {k:9s} {v:5d} tok")
        if r.get("recall"):
            lines.append("  召回：")
            for t in r["recall"]:
                lines.append("   - " + t)
        if r.get("summaries"):
            lines.append(f"  摘要层：{r['summaries']}")
        return "\n".join(lines)


def load_persona(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


# 「他平时说话是这样」那一段的头尾。提成常量是因为**总额兜底**也要能从 sys_parts 里
# 认出这一段并把样本一条条砍掉（见 assemble 末尾）。
STYLE_HEAD = "【他平时说话是这样（不是要你学他，是别拿腔调）】"
STYLE_TAIL = "（他一句话常常只有半件事，问号比句号多；你也用不着每句都答完整。）"


def _style_line(xs: list[str]) -> str:
    return STYLE_HEAD + "\n" + "\n".join(f"- {s}" for s in xs) + "\n" + STYLE_TAIL


def assemble(store: Store, peer: str, user_text: str, *, persona: str = "",
             budget: dict | None = None, recent_turns: int = 10,
             k_recall: int = 10, with_evidence: bool = False) -> Assembled:
    b = dict(BUDGET)
    if budget:
        b.update(budget)
    if with_evidence:
        b["evidence"] = b.get("evidence") or 400

    st = store.state(peer)
    slots: dict[str, int] = {}
    report: dict = {"slots": slots, "recall": [], "summaries": 0}
    style_kept: list[str] = []      # 「他的说话样本」实际装进去的那几条（总额兜底时往回砍）

    # ---------- S0 人设 ----------
    sys_parts = [clip(persona, b["persona"] * 2)] if persona else []
    plt = est_tokens(sys_parts[0]) if sys_parts else 0
    if plt > b["persona"]:
        sys_parts[0] = clip(sys_parts[0], b["persona"] * 2)
    slots["persona"] = plt

    # ---------- S1 时间锚点 ----------
    # 为什么放在人设之后、状态之前：它是"这一轮"的事实，不是性格的一部分，
    # 但也必须先于任何记忆出现 —— 下面每条记忆的「昨天/前天」都以它为准。
    nl = now_line()
    sys_parts.append("【现在】" + nl)
    slots["now"] = est_tokens(nl)

    # ---------- S1.2 断过一段又接上：告诉她中间发生了什么 ----------
    # 她解释自己只有"空白"这一个词，是因为她手上真的只有那一条信息。
    # 事实上中间记录没停过，只是她看不见 —— 这一行就是把看不见的递到她手上。
    gl = gap_line(store, peer)
    if gl:
        sys_parts.append("【你不在的时候】" + gl)
        slots["gap"] = est_tokens(gl)
    else:
        slots["gap"] = 0

    # ---------- S1.3 上一次说话的最后几句 ----------
    # 隔夜醒来时，最近窗口里已经没有昨晚那几轮了；检索又只认词面，
    # 早上那句「早」没有任何词可命中 —— 于是她答「这段我这边翻不到」，
    # 而记录一条不少地躺在库里。这里补的是"上次说到哪儿"这个位置感。
    tail = last_session_tail(store, peer)
    if tail:
        lines = [f"- {'他' if r == 'user' else '你'}：{t}" for r, t in tail]
        body = "\n".join(lines)
        n = est_tokens(body)
        if n > b["last"]:
            body = clip(body, b["last"] * 2)
            n = est_tokens(body)
        sys_parts.append("【上一次说话的最后几句（接着往下说，别当成刚说的）】\n" + body)
        slots["last"] = n
    else:
        slots["last"] = 0

    # ---------- S1.4 你自己刚说过的话 ----------
    # L0 一定记得，AstrBot 的会话历史不一定 —— 自主开口、工具轮里说出去的话
    # 不进那份历史。她因此能在 13:16 说"翻到你 D 盘那个 cold-vault"，
    # 13:53 又答"没翻过，我只进了 bishe 和 develop"。
    mine = recent_self_lines(store, peer, k=3)
    if mine:
        body = "\n".join(f"- {x}" for x in mine)
        n = est_tokens(body)
        if n <= b["mine"]:
            sys_parts.append("【你自己刚说过的（紧接着的这几分钟；别重复它，是让你知道自己站在哪儿。"
                             "其中某句要是跟事实不符，以事实为准，可以改口）】\n"
                             + body)
            slots["mine"] = n
        else:
            slots["mine"] = 0
    else:
        slots["mine"] = 0

    # ---------- S1.5 他平时怎么打字 ----------
    # 不是让她学他，是让她别用模型的默认腔调。样本从聊天里自己更新，会变。
    # 2026-11-12 与 main._build_block 同步：先挑带问号的、最多 6 条，末尾给一句关于他的事实。
    # 这里比插件那边多一步 **按预算装**：assemble 有总额度，sample 多了会把别的槽挤掉
    # （`test_budget_respected` 就是盯这个的）。装不下的样本这一轮不出现，不删。
    styles = user_style(store, peer)
    if styles:
        kept_s: list[str] = []
        used_s = est_tokens(STYLE_HEAD) + est_tokens(STYLE_TAIL)
        # 留 15% 余量：`est_tokens` 对逐条估的和整段估的不是同一个数（换行、前缀各算各的），
        # 贴着上限装会在总额兜底那一步露出来（`test_budget_respected` 实测超 28 token）。
        for s in styles:
            n = est_tokens("- " + s)
            if used_s + n > b["style"] * 0.85:
                break
            kept_s.append(s)
            used_s += n
        while kept_s and est_tokens(_style_line(kept_s)) > b["style"]:
            kept_s.pop()
        if kept_s:
            line = _style_line(kept_s)
            slots["style"] = est_tokens(line)
            sys_parts.append(line)
            style_kept = kept_s          # 兜底那一步从这里往回砍
        else:
            slots["style"] = 0
    else:
        slots["style"] = 0

    # ---------- S1 关系状态 ----------
    sl = state_line(st)
    if sl:
        sys_parts.append("【当前关系状态】" + sl)
        slots["state"] = est_tokens(sl)
    else:
        slots["state"] = 0

    # ---------- S1.5 锚点（恒定，不依赖提问）----------
    anchors = pick_anchors(store, peer)
    used = 0
    kept_anchor: list[str] = []
    for a in anchors:
        n = est_tokens(a)
        if used + n > b["anchor"]:
            break
        kept_anchor.append(a)
        used += n
    slots["anchor"] = used
    if kept_anchor:
        sys_parts.append("【不能忘的（每轮都在场）】\n"
                         + "\n".join(f"- {a}" for a in kept_anchor))

    # ---------- S2 长期记忆召回 ----------
    # 检索 query 用最近几条 user 原话补全指代（"又忘了跟老板说"靠前一句才解得开）
    tail = [m.text for m in store.recent(peer, limit=8)
            if m.role == "user" and m.text != user_text][-2:]
    q = " ".join(tail + [user_text])
    hits: list[Hit] = retrieve(store, peer, q, k=k_recall)
    mem_lines, used = [], 0
    for h in hits:
        if h.mem.kind == "self":      # 自述条目走 S2.5，别在"关于对方"里出现
            continue
        line = h.line()
        n = est_tokens(line)
        if used + n > b["recall"]:
            break
        mem_lines.append(line)
        used += n
        report["recall"].append(f"{h.score:.3f} {line}")
    slots["recall"] = used
    if mem_lines:
        sys_parts.append("【关于对方的长期记忆（按相关度，越前越重要）】\n"
                         + "\n".join(f"- {l}" for l in mem_lines))

    # ---------- S2.5 她自己的立场 ----------
    # 最近几轮她自己说过的话：讲过的这轮不再送进去（防复读）
    said_recent = [m.text for m in store.recent(peer, limit=8) if m.role == "assistant"]
    self_lines, used = [], 0
    for s in pick_self(store, peer, k=4, avoid=said_recent):
        n = est_tokens(s)
        if used + n > b["self"]:
            break
        self_lines.append(s)
        used += n
    slots["self"] = used
    if self_lines:
        sys_parts.append("【你自己以前说过的（前后要对得上；想改就明说改了）】\n"
                         + "\n".join(f"- {l}" for l in self_lines))

    # ---------- S2.6 忽然想到的 ----------
    # 不是每轮都有；有就说明这轮她可能岔一句。别解释、别硬塞 —— 它只是"在场"
    seen_ids = {h.mem.id for h in hits} | set()
    dt = detour(store, peer, q, exclude=seen_ids)
    if dt is not None:
        line = dt.line()
        n = est_tokens(line)
        if n <= b["detour"]:
            sys_parts.append("【此刻还在你脑子里的（没出口的那种）】\n"
                             f"- {line}")
            slots["detour"] = n
        else:
            slots["detour"] = 0
    else:
        slots["detour"] = 0

    # ---------- S3 历史摘要（近→远）----------
    sum_lines, used = [], 0
    for level in (1, 2, 3):
        for row in store.summaries(peer, level):
            t = f"[L{level}] {row['text']}"
            n = est_tokens(t)
            if used + n > b["summary"]:
                break
            sum_lines.append(t)
            used += n
        if used >= b["summary"]:
            break
    slots["summary"] = used
    report["summaries"] = len(sum_lines)
    if sum_lines:
        sys_parts.append("【更早的对话摘要】\n" + "\n".join(sum_lines))

    # ---------- S4 原话佐证（可选）----------
    if b["evidence"] > 0:
        ev, used = [], 0
        for mid, score in recall_messages(store, peer, user_text, k=4):
            for m in store.get_msgs([mid]):
                t = f"[{m.ts}] {m.role}: {clip(m.text, 120)}"
                n = est_tokens(t)
                if used + n > b["evidence"]:
                    break
                ev.append(t)
                used += n
        slots["evidence"] = used
        if ev:
            sys_parts.append("【原话片段（可作为依据引用）】\n" + "\n".join(ev))
    else:
        slots["evidence"] = 0

    # ---------- S6 你自己的规矩（放 system 最后一段）----------
    # 2026-11-08：为什么放这里。人设里本来有「别掉回默认」那节（不概括 / 不给建议 /
    # 不追问 / 说完就停），但它在 system **最前面**，后面跟着三千字注入块的示范，
    # 结果一条都没照做 —— 11-07 18:02 那句同时用了「不过」、概括了他、还给了建议。
    # 规则没说错，是位置不对：近因效应下她照着后面的示范走。
    # 这里只在**最后**放一次，而且用"你本来就是这样的"口吻，不用"你要…" ——
    # 命令式在角色扮演里是最弱的一档（调研 §3），越像规定越容易被改写成同构变体。
    sys_parts.append(
        "【你自己的规矩】\n"
        "你说话不爱概括别人刚说过的，也不给建议、不给方案——那是助手干的事，不是你。\n"
        "话说完就停，不用补满，也不用留个钩子等他接。\n"
        "你可以不同意他。不同意就直说不同意，不用先夸一半再转折。"
    )

    # ---------- S5 最近窗口 ----------
    recent = store.recent(peer, limit=recent_turns * 2)
    win, used = [], 0
    for m in reversed(recent):
        n = est_tokens(m.text)
        if used + n > b["recent"]:
            break
        win.append(m)
        used += n
    win.reverse()
    slots["recent"] = used

    # ---------- 总预算兜底：超了就砍召回尾部，再砍摘要 ----------
    total = sum(slots.values()) + est_tokens(user_text)
    budget_total = (b["persona"] + b["state"] + b["anchor"] + b.get("self", 0)
                    + b.get("now", 0) + b.get("detour", 0) + b.get("gap", 0)
                    + b.get("last", 0) + b.get("mine", 0) + b.get("style", 0)
                    + b["recall"] + b["summary"] + b["recent"] + b["evidence"])
    if total > budget_total:
        over = total - budget_total
        while over > 0 and self_lines:
            over -= est_tokens(self_lines.pop())
        while over > 0 and mem_lines:
            over -= est_tokens(mem_lines.pop())
        while over > 0 and sum_lines:
            over -= est_tokens(sum_lines.pop())
        # 最后才砍「他的说话样本」：它是学习材料，不是事实。
        # 不砍它的话，样本一多就会把 20~30 token 的超额顶在总额外面
        # （`test_budget_respected` 实测 545 > 530）—— 而兜底本来就应该把持「总额」这条线。
        while over > 0 and style_kept:
            style_kept.pop()
            new_line = _style_line(style_kept) if style_kept else ""
            new_tok = est_tokens(new_line) if new_line else 0
            delta = slots.get("style", 0) - new_tok
            if delta > 0:
                over -= delta
                total -= delta
            slots["style"] = new_tok
            if new_line:
                sys_parts = [new_line if p.startswith(STYLE_HEAD) else p for p in sys_parts]
            else:
                sys_parts = [p for p in sys_parts if not p.startswith(STYLE_HEAD)]
    report["total"] = total
    report["budget_total"] = budget_total

    messages = [{"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}]
    for m in win:
        messages.append({"role": "assistant" if m.role != "user" else "user",
                         "content": m.text})
    messages.append({"role": "user", "content": user_text})
    return Assembled(messages, report)
