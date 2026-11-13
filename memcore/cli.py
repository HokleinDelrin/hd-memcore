"""命令行入口。

    python -m memcore.cli demo           # 200 轮压力演示：早期细节在最后一轮还在不在
    python -m memcore.cli chat           # 交互聊天（配了模型就真生成，没配就走离线自述）
    python -m memcore.cli probe "问题"    # 看某一轮到底装配了什么
    python -m memcore.cli export out.md  # 导出人读的记忆快照
    python -m memcore.cli stats          # 看统计
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

from .bot import Bot, dump_md
from .assemble import assemble
from .distill import state_line
from .util import clip, dumps, est_tokens

# ---------------------------------------------------------------- demo 语料
NEEDLES = [
    dict(turn=3, kind="宠物",
         user="我上个月在楼下捡了只橘猫，取名叫豆豆，天天半夜拆家",
         probe="我家那只猫今天又把我的耳机线咬断了，真是服了",
         expect="豆豆"),
    dict(turn=17, kind="称呼边界",
         user="别叫我阿哲，我讨厌这个小名，叫我老周就行",
         probe="随便问一句，你一般怎么称呼我",
         expect="老周"),
    dict(turn=44, kind="待办事件",
         user="下周三我要去杭州面试，一家做支付的公司，有点紧张",
         probe="我明天就要出发了，心里挺慌的",
         expect="杭州"),
    dict(turn=88, kind="重要情绪",
         user="我妈上个月住院了，是心脏的毛病，我最近真的有点撑不住",
         probe="最近特别累，什么都不想干",
         expect="住院"),
    dict(turn=130, kind="承诺",
         user="说好了，下次我给你看我画的那张夜景图",
         probe="我是不是还欠你点什么没给你看",
         expect="夜景"),
    dict(turn=170, kind="偏好",
         user="我从小就不吃香菜，一闻到那个味儿就反胃",
         probe="晚上点了碗牛肉面，又忘了跟老板说",
         expect="香菜"),
]

_FILL = [
    "今天下班晚了，地铁上人挤人",
    "刚吃完饭，躺着刷了会儿视频",
    "外面在下雨，窗户没关严",
    "我在想要不要换个耳机",
    "今天开会开了一下午，脑子糊了",
    "刚才下楼买了瓶可乐",
    "有点困，但还不想睡",
    "周末好像没什么安排",
    "我在收拾屋子，翻出好多旧东西",
    "今天太阳挺好的，晒了会儿被子",
    "你那边天气怎么样",
    "我在想要不要学个新东西",
    "刚洗完澡，舒服",
    "今天效率奇低，啥也没干成",
    "室友在客厅打游戏，吵死了",
    "有点想吃火锅",
    "今天走了两万步，腿废了",
    "我在等一个快递，还没到",
]

_REPLY = [
    "嗯，听着就累",
    "那你早点歇着",
    "下雨记得关窗啊",
    "换吧，旧的早该换了",
    "开会最磨人了",
    "可乐少喝点",
    "困了就睡，别硬撑",
    "周末要不要出去走走",
    "旧东西最容易翻出回忆",
    "晒被子那味儿好闻",
    "我这边也阴着",
    "学呗，反正闲着也是闲着",
    "舒服就好",
    "效率低就低吧，谁天天高效",
    "那你去说他们两句",
    "火锅可以，我投一票",
    "两万步，你狠",
    "快递就这样，你越等它越慢",
]


def _build_turns(turns: int, seed: int = 7) -> list[tuple[str, str, dict | None]]:
    """生成 (role, text, needle?) 序列。针固定在指定轮次，其余为自然填充。"""
    rnd = random.Random(seed)
    needle_at = {n["turn"]: n for n in NEEDLES}
    out: list[tuple[str, str, dict | None]] = []
    for i in range(1, turns + 1):
        if i in needle_at:
            n = needle_at[i]
            out.append(("user", n["user"], n))
            out.append(("assistant", "记住了。", None))
            continue
        out.append(("user", rnd.choice(_FILL), None))
        out.append(("assistant", rnd.choice(_REPLY), None))
    return out


# ---------------------------------------------------------------- demo
def cmd_demo(a) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="memcore_demo_"))
    db = tmp / "demo.db"
    bot = Bot(str(db), persona_text="你是一个简洁的对话助手。", distill_every=0)   # 手动控蒸馏节奏
    peer = "demo:private:10001"

    turns = _build_turns(a.turns)
    print(f"▶ 灌入 {len(turns)//2} 轮对话（{len(turns)} 条消息），每 8 轮蒸馏一次\n")
    n = 0
    for role, text, _ in turns:
        bot.store.add_msg(peer, role, text)
        if role == "user":                    # 与 bot.handle 一致的亲密度微小累加
            bot.store.save_state(
                peer, bond=round(min(1.0, bot.store.state(peer)["bond"] + 0.0008), 4))
        n += 1
        if n % 16 == 0:                       # 16 条 = 8 轮
            bot.distill_now(peer)
    bot.distill_now(peer)

    st = bot.store.state(peer)
    mems = bot.store.all_memories(peer)
    print(f"▶ 蒸馏结果：L2 记忆 {len(mems)} 条，L1/L2/L3 摘要 "
          f"{len(bot.store.summaries(peer))} 条，状态：{state_line(st)}\n")

    # ---------- 探针 ----------
    rows = []
    for nd in NEEDLES:
        asm = assemble(bot.store, peer, nd["probe"], persona=bot.persona)
        sys_text = asm.messages[0]["content"]
        in_ctx = nd["expect"] in sys_text
        # baseline：只保留最近 20 条原文（现在大多数"打包迁移"的实际做法）
        win = "".join(m.text for m in bot.store.recent(peer, limit=20))
        in_win = nd["expect"] in win
        hit_score = None
        for line in asm.report.get("recall", []):
            if nd["expect"] in line:
                try:
                    hit_score = float(line.split()[0])
                except (ValueError, IndexError):
                    hit_score = None
                break
        rows.append((nd, in_win, in_ctx, hit_score, asm.report["total"]))

    print("▶ 记忆探针（第 %d 轮之后回问早期细节）" % (len(turns) // 2))
    print("  %-10s %-8s %-10s %-8s %s" % ("类型", "最近20条", "本方案上下文", "召回分", "关键词"))
    ok = 0
    for nd, in_win, in_ctx, score, total in rows:
        ok += 1 if in_ctx else 0
        print("  %-10s %-8s %-10s %-8s %s" % (
            nd["kind"],
            "有" if in_win else "丢了",
            "有" if in_ctx else "丢了",
            f"{score:.3f}" if score is not None else "-",
            nd["expect"],
        ))
    print(f"\n  → 最近窗口方案保住 {sum(r[1] for r in rows)}/{len(rows)}；"
          f"本方案 {ok}/{len(rows)}")

    # ---------- 单轮装配明细 ----------
    probe = rows[2][0]["probe"]      # 用"待办事件"那条展示
    asm = assemble(bot.store, peer, probe, persona=bot.persona, with_evidence=True)
    print(f"\n▶ 单轮装配明细（提问：「{probe}」）")
    print(asm.debug())
    print(f"\n  实际 prompt 行数 {len(asm.messages)}，"
          f"字符 {sum(len(m['content']) for m in asm.messages)}")
    print("  （预算未用满是好事：剩余空间留给更长的摘要与更多召回，"
          "总上限卡住就不会让延迟和幻觉随对话长度漂移）")

    out_md = Path(a.out) if a.out else tmp / "snapshot.md"
    dump_md(out_md, bot.store, peer)
    print(f"\n▶ 记忆快照已导出：{out_md}")
    print(f"▶ 数据库：{db}")
    if a.show:
        print("\n--- 装配出的 system prompt ---\n" + asm.messages[0]["content"])
    bot.close()
    return 0


# ---------------------------------------------------------------- 其它命令
def cmd_chat(a) -> int:
    bot = Bot(a.db, persona_path=a.persona)
    print("输入消息开始聊天，:quit 退出，:probe 看本轮装配，:mem 看记忆，:distill 立即蒸馏")
    while True:
        try:
            text = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == ":quit":
            break
        if text == ":mem":
            dump_md(a.out or "snapshot.md", bot.store, a.peer)
            print(f"已导出 {a.out or 'snapshot.md'}")
            continue
        if text == ":distill":
            r = bot.distill_now(a.peer)
            print(f"新增记忆 {len(r.memories)} 条（{r.source}）")
            continue
        if text == ":probe":
            from .assemble import assemble
            asm = assemble(bot.store, a.peer, "看看我现在能看到什么", persona=bot.persona)
            print(asm.debug())
            continue
        r = bot.handle(a.peer, text)
        print("她 >", r.text)
        if a.debug:
            print("  [" + r.assembled.debug().replace("\n", "\n  ") + "]")
    bot.close()
    return 0


def cmd_probe(a) -> int:
    from .assemble import assemble
    bot = Bot(a.db, persona_path=a.persona)
    asm = assemble(bot.store, a.peer, a.text, persona=bot.persona, with_evidence=True)
    print(asm.debug())
    print("\n--- system ---\n" + asm.messages[0]["content"])
    print("\n--- messages ---")
    for m in asm.messages[1:]:
        print(f"{m['role']:9s} | {clip(m['content'], 160)}")
    bot.close()
    return 0


def cmd_export(a) -> int:
    bot = Bot(a.db, persona_path=a.persona)
    dump_md(a.out, bot.store, a.peer)
    print(f"已导出 {a.out}")
    bot.close()
    return 0


def cmd_stats(a) -> int:
    bot = Bot(a.db, persona_path=a.persona)
    peer = a.peer
    kinds: dict[str, int] = {}
    for m in bot.store.all_memories(peer):
        kinds[m.kind] = kinds.get(m.kind, 0) + 1
    print(dumps({
        "peer": peer,
        "消息数": bot.store.count_msgs(peer),
        "蒸馏水位": bot.store.watermark(peer),
        "记忆条目": kinds,
        "摘要条数": len(bot.store.summaries(peer)),
        "状态": bot.store.state(peer),
    }))
    bot.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台兜底
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="memcore")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="长程对话压力演示")
    d.add_argument("--turns", type=int, default=200)
    d.add_argument("--out", default=None)
    d.add_argument("--show", action="store_true", help="打印完整 system prompt")
    d.set_defaults(fn=cmd_demo)

    c = sub.add_parser("chat", help="交互聊天")
    c.add_argument("--db", default="persona.db")
    c.add_argument("--persona", default="persona.md")
    c.add_argument("--peer", default="local:private:0")
    c.add_argument("--out", default="snapshot.md")
    c.add_argument("--debug", action="store_true")
    c.set_defaults(fn=cmd_chat)

    p = sub.add_parser("probe", help="查看某轮的上下文装配")
    p.add_argument("text")
    p.add_argument("--db", default="persona.db")
    p.add_argument("--persona", default="persona.md")
    p.add_argument("--peer", default="local:private:0")
    p.set_defaults(fn=cmd_probe)

    e = sub.add_parser("export", help="导出人读快照")
    e.add_argument("out")
    e.add_argument("--db", default="persona.db")
    e.add_argument("--persona", default="persona.md")
    e.add_argument("--peer", default="local:private:0")
    e.set_defaults(fn=cmd_export)

    s = sub.add_parser("stats", help="看统计")
    s.add_argument("--db", default="persona.db")
    s.add_argument("--persona", default="persona.md")
    s.add_argument("--peer", default="local:private:0")
    s.set_defaults(fn=cmd_stats)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
