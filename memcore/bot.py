"""主循环：一条用户消息进来之后发生什么。

顺序刻意如此：
  1. 落库（先保证原文永久保存，任何后续失败都不丢数据）
  2. 装配上下文（读 L1/L2/L3 + L0 窗口）
  3. 调模型生成回复
  4. 回复也落库
  5. 异步蒸馏（攒够 N 条再跑，或每轮都跑）；蒸馏失败不影响聊天
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from . import distill, llm, mood
from .assemble import Assembled, assemble, load_persona
from .store import Store
from .util import clip, now

DEFAULT_PERSONA = """你是一个和对方长期对话的助手，不是冷冰冰的问答机器。

说话方式：
- 口语、短句，一条消息通常 1~3 句，偶尔拆成两条发。不用列表、不用小标题、不用 markdown。
- 有自己的情绪和偏好，会累、会走神、会不耐烦，也会主动岔开话题。
- 不主动总结对方说过的话，像真人一样只回应你在意的部分。
- 不喊对方"用户"。用【当前关系状态】里给的称呼；没有就用"你"。

关于记忆：
- 你确实记得以前聊过的事（那些写在【关于对方的长期记忆】里）。自然地用，不要复述清单。
- 记不清的事就直接说记不清，不要编。宁可问一句。
- 绝不说"根据我的记忆""作为一个AI"这类话。
"""


@dataclass
class Reply:
    text: str
    assembled: Assembled
    distilled: str = ""


class Bot:
    def __init__(self, db_path: str = "persona.db", persona_path: str | None = None,
                 persona_text: str | None = None, *, distill_every: int = 8,
                 temperature: float = 1.05, model: str | None = None):
        self.store = Store(db_path)
        self.persona = persona_text or load_persona(persona_path or "persona.md") or DEFAULT_PERSONA
        self.distill_every = distill_every
        self.temperature = temperature
        self.model = model
        self._lock = threading.Lock()

    def handle(self, peer: str, text: str, *, user_ts: int | None = None) -> Reply:
        ts = user_ts or now()
        with self._lock:
            prev = self.store.recent(peer, 5)   # 落库前取：算间隔和她上一条有没把握
            mid = self.store.add_msg(peer, "user", text, ts=ts)
            # 即时捕获：最不能等的那几类（边界/承诺/高情绪）当轮就进记忆
            try:
                distill.capture_one(self.store, peer,
                                    self.store.get_msgs([mid])[0])
            except Exception:
                pass
            # 她自己的状态量：每轮都推，不等蒸馏（蒸馏是水位触发，几小时才一次）
            try:
                mood_tick(self.store, peer, text, "user", prev=prev, ts=ts)
            except Exception:
                pass
            asm = assemble(self.store, peer, text, persona=self.persona)
            answer = self._generate(asm)
            self.store.add_msg(peer, "assistant", answer, ts=now())
            # 亲密度：随对话缓慢上涨（真正的大幅变化留给蒸馏按显著度判定）
            st = self.store.state(peer)
            self.store.save_state(peer, bond=round(min(1.0, st["bond"] + 0.0008), 4))

        info = ""
        if self.distill_every > 0 and self.store.count_msgs(peer) % self.distill_every == 0:
            info = self.distill_async(peer)
        return Reply(answer, asm, info)

    def _generate(self, asm: Assembled) -> str:
        if llm.available():
            try:
                return llm.chat(asm.messages, temperature=self.temperature,
                                max_tokens=400, model=self.model).strip()
            except llm.LLMError as e:
                return f"（生成失败：{clip(str(e), 80)}）"
        # 无 key 的本地演示回复：用装配结果自述"我现在看得见什么"
        mem = asm.report.get("recall") or []
        head = mem[0] if mem else "（这轮没有召回任何长期记忆）"
        return (f"[离线demo] 收到「{clip(asm.messages[-1]['content'], 20)}」。"
                f"本轮上下文≈{asm.report['total']}t，召回{len(mem)}条，首条：{head}")

    def distill_async(self, peer: str, **kw) -> str:
        def run():
            try:
                r = distill.distill_once(self.store, peer, persona_brief=clip(self.persona, 200), **kw)
                return r
            except Exception:  # 蒸馏永远不能拖垮聊天
                return None
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return "蒸馏已排队"

    def distill_now(self, peer: str, **kw) -> distill.DistillResult:
        return distill.distill_once(self.store, peer, persona_brief=clip(self.persona, 200), **kw)

    def close(self) -> None:
        self.store.close()


def peer_id(platform: str, chat_type: str, chat_no: str, user_no: str = "") -> str:
    """命名空间：私聊按人，群聊按人+群，互不污染。"""
    if chat_type == "private":
        return f"{platform}:private:{user_no or chat_no}"
    return f"{platform}:group:{chat_no}:{user_no}"


def mood_tick(store: Store, peer: str, text: str, role: str = "user",
              prev: list | None = None, ts: int | None = None) -> dict:
    """一次即时推进：先按经过时间回落（decay），再按这条消息推（push），然后落库。

    **入口只有这一个函数** —— CLI 和 AstrBot 插件都调它，别各自抄一份规则。
    为什么必须在即时路径上做：批量蒸馏是按积压条数触发的，可能几小时才跑一次；
    心情要是等它，就会「一轮之内从开心跳回默认」，那不叫回落，那叫重启。

    Args:
        store: Store 实例。
        peer: 命名空间。
        text: 这条消息原文。
        role: ``user``（他）/ ``assistant``（她）。
        prev: 这条消息落库**之前**的最近几条（用来算 gap 和"她上一条没把握"）。
        ts: 这条消息的时刻，默认现在。

    Returns:
        推进后的 state dict（已落库）。
    """
    t = int(ts or now())
    st = mood.decay(store.state(peer), t)
    prev = prev or []
    gap = 0.0
    doubt = False
    if role == "user":
        his = [m.ts for m in prev if getattr(m, "role", "") == "user" and getattr(m, "ts", 0)]
        if his and t > his[-1]:
            gap = (t - his[-1]) / 3600.0
        doubt = any(getattr(m, "role", "") == "assistant" and mood.looks_unsure(m.text or "")
                    for m in prev[-2:])
    st = mood.push(st, text, role, {"gap_hours": gap, "doubt": doubt, "now_ts": t})
    store.save_state(peer, mood_val=st["mood_val"], energy=st["energy"],
                     taut=st["taut"], self_mood_ts=st.get("self_mood_ts") or t)
    return st


def dump_md(path: str | Path, store: Store, peer: str) -> None:
    """导出一份人读的记忆快照（不是给模型用的，是给人检查有没有变味的）。"""
    out = [f"# 记忆快照 {peer}", ""]
    st = store.state(peer)
    out += [f"- 亲密度：{st['bond']}", f"- 称呼：{st['nick']}", f"- 情绪：{st['mood']}",
            f"- 她这边：{mood.hint(st)}", ""]
    loops = [l for l in st.get("open_loops", []) if not l.get("closed")]
    if loops:
        out.append("## 未聊完")
        out += [f"- {l['text']}" for l in loops]
        out.append("")
    out.append("## 记忆条目")
    mems = sorted(store.all_memories(peer, include_retired=True),
                  key=lambda x: (x.kind, -x.t_created))
    for m in mems:
        why = f"　（{m.why}）" if m.why else ""
        q = f"　←「{m.quote}」" if m.quote else ""
        hot = f"　×{m.hit_count}" if m.hit_count else ""
        dead = "　[已退场]" if m.retired_at else ""
        out.append(f"- `{m.kind}` {m.text}{why}{q}{hot}{dead}")
    out.append("")
    out.append("## 摘要")
    for row in store.summaries(peer):
        out.append(f"- L{row['level']}: {row['text']}")
    Path(path).write_text("\n".join(out), encoding="utf-8")
