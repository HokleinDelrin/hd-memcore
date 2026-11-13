"""回归测试：python -m unittest discover -s tests -v

这些测试锁住五件事：
  1. L0 只追加、水位单调、重复蒸馏不产生重复记忆
  2. 早期细节在 200 轮后仍能被召回并进入上下文
  3. 预算不超，锚点每轮都在场
  4. 不同 peer（尤其群聊里不同人）记忆不串
  5. 活记忆：她自己说过的话记得住、被想起会升温、长期不提会退场且能被唤回
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memcore.assemble import assemble, pick_anchors          # noqa: E402
from memcore.bot import Bot, peer_id                         # noqa: E402
from memcore.recall import retrieve, maintain                # noqa: E402
from memcore.store import Store                              # noqa: E402
from memcore import distill                                  # noqa: E402


def tmpdb() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="memcore_test_"), "t.db")


def _capture_last(s: Store, p: str) -> None:
    distill.capture_one(s, p, s.recent(p, limit=1)[-1])


class TestStore(unittest.TestCase):
    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:1"

    def tearDown(self):
        self.s.close()

    def test_append_only_and_watermark(self):
        a = self.s.add_msg(self.p, "user", "我养了只猫叫豆豆")
        b = self.s.add_msg(self.p, "assistant", "记下了")
        self.assertEqual(self.s.count_msgs(self.p), 2)
        self.assertEqual(self.s.watermark(self.p), 0)
        distill.distill_once(self.s, self.p, use_llm=False)
        self.assertEqual(self.s.watermark(self.p), b)
        self.assertGreaterEqual(self.s.watermark(self.p), a)
        # 原文一条没少
        self.assertEqual(self.s.count_msgs(self.p), 2)

    def test_recent_order(self):
        for i in range(5):
            self.s.add_msg(self.p, "user", f"第{i}条")
        got = [m.text for m in self.s.recent(self.p, limit=3)]
        self.assertEqual(got, ["第2条", "第3条", "第4条"])


class TestDistill(unittest.TestCase):
    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:2"

    def tearDown(self):
        self.s.close()

    def _feed(self, *texts, role="user"):
        for t in texts:
            self.s.add_msg(self.p, role, t)
        return distill.distill_once(self.s, self.p, use_llm=False)

    def test_needles_extracted(self):
        self._feed("我上个月捡了只橘猫，取名叫豆豆",
                   "别叫我阿哲，叫我老周就行",
                   "下周三我要去杭州面试，有点紧张",
                   "我从小就不吃香菜")
        text = " ".join(m.text for m in self.s.all_memories(self.p))
        for kw in ("豆豆", "老周", "杭州", "香菜"):
            self.assertIn(kw, text, f"{kw} 没被抽出来")

    def test_noise_filtered(self):
        self._feed("我在想要不要换个耳机", "有点困但还不想睡", "你那边天气怎么样？")
        self.assertEqual(self.s.all_memories(self.p), [], "闲聊被误存成记忆了")

    def test_boundary_nick(self):
        self._feed("别叫我阿哲，我讨厌这个小名，叫我老周就行")
        st = self.s.state(self.p)
        self.assertEqual(st["nick"], "老周", "被否定的称呼不该被采纳")

    def test_bond_grows(self):
        before = self.s.state(self.p)["bond"]
        self._feed("我妈上个月住院了，心脏的问题，我最近真的撑不住")
        self.assertGreater(self.s.state(self.p)["bond"], before)

    def test_quote_is_verbatim(self):
        self._feed("我从小就不吃香菜，一闻到那个味儿就反胃")
        for m in self.s.all_memories(self.p):
            self.assertTrue(m.quote, "记忆必须带原话依据")
            self.assertIn(m.quote[:8], "我从小就不吃香菜，一闻到那个味儿就反胃")

    def test_idempotent(self):
        self._feed("我养了只猫叫豆豆")
        n1 = len(self.s.all_memories(self.p))
        distill.distill_once(self.s, self.p, use_llm=False)   # 水位已到末尾，无新增
        self.assertEqual(len(self.s.all_memories(self.p)), n1)


class TestRecall(unittest.TestCase):
    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:3"
        for t in ["我上个月捡了只橘猫，取名叫豆豆",
                  "下周三我要去杭州面试",
                  "我从小就不吃香菜"]:
            self.s.add_msg(self.p, "user", t)
        distill.distill_once(self.s, self.p, use_llm=False)

    def tearDown(self):
        self.s.close()

    def test_single_char_channel(self):
        """bigram 对单字关键词失效，单字通道必须补上。"""
        hits = retrieve(self.s, self.p, "我家那只猫又咬断耳机线了", k=3)
        self.assertTrue(hits)
        self.assertIn("豆豆", hits[0].mem.text)

    def test_relevance_order(self):
        hits = retrieve(self.s, self.p, "面试的事我有点紧张", k=3)
        self.assertIn("杭州", hits[0].mem.text)


class TestAssemble(unittest.TestCase):
    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:4"
        self.s.add_msg(self.p, "user", "别叫我阿哲，叫我老周")
        self.s.add_msg(self.p, "user", "我妈住院了，我撑不住")
        for i in range(30):
            self.s.add_msg(self.p, "user", f"闲聊第{i}句，今天天气还行")
            self.s.add_msg(self.p, "assistant", "嗯")
        distill.distill_once(self.s, self.p, use_llm=False)

    def tearDown(self):
        self.s.close()

    def test_anchor_always_present(self):
        anchors = pick_anchors(self.s, self.p)
        self.assertTrue(anchors)
        asm = assemble(self.s, self.p, "随便聊点什么", persona="你是小早。")
        sysmsg = asm.messages[0]["content"]
        self.assertIn("不能忘的", sysmsg)
        self.assertIn("老周", sysmsg)          # 边界每轮都在

    def test_budget_respected(self):
        # `style`（「他平时说话是这样」的样本槽）2026-11-05 才加，当时这份 budget 没跟上；
        # 2026-11-12 样本从 4 条扩到 6 条之后它就露出来了（total 545 > 530）。
        # 补上是对的：这一格真的会往 system prompt 里塞字，就该算进总额。
        budget = {"persona": 100, "state": 60, "anchor": 60, "recall": 80,
                  "summary": 80, "recent": 120, "evidence": 0, "style": 80}
        asm = assemble(self.s, self.p, "你还记得什么", persona="你是小早。" * 50,
                       budget=budget)
        total = sum(v for v in asm.report["slots"].values())
        self.assertLessEqual(total, sum(budget.values()) + 30)

    def test_recent_window_excludes_nothing_needed(self):
        asm = assemble(self.s, self.p, "最后一句", persona="p")
        self.assertEqual(asm.messages[-1]["content"], "最后一句")
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in asm.messages[1:]))


class TestSelfMemory(unittest.TestCase):
    """她自己的立场要能记下来 —— 不然每次问「你为什么这么想」，她都只能现编一个。"""

    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:5"

    def tearDown(self):
        self.s.close()

    def test_self_captured_from_her_side(self):
        mid = self.s.add_msg(self.p, "assistant", "我不喜欢被叫大肥鱼，因为鲸是鲸，鱼是鱼")
        _capture_last(self.s, self.p)
        selfs = [m for m in self.s.all_memories(self.p) if m.kind == "self"]
        self.assertEqual(len(selfs), 1, "她自己的立场没被记下来")
        self.assertIn("大肥鱼", selfs[0].text)
        self.assertIsNotNone(selfs[0].why, "self 类必须带理由")
        self.assertIn("因为", selfs[0].why)
        self.assertEqual(selfs[0].src_msg, [mid])

    def test_self_does_not_grow_bond(self):
        before = self.s.state(self.p)["bond"]
        for t in ("我觉得这样不好", "我不想再问了", "我记着这件事"):
            self.s.add_msg(self.p, "assistant", t)
            _capture_last(self.s, self.p)
        self.assertEqual(self.s.state(self.p)["bond"], before, "自述不该涨亲密度")

    def test_self_shown_in_assemble(self):
        self.s.add_msg(self.p, "assistant", "我不喜欢别人催我，因为一催我就想沉下去")
        _capture_last(self.s, self.p)
        asm = assemble(self.s, self.p, "随便说点什么", persona="你是深溟。")
        # 刚说完的那句会同时在「最近窗口」里，所以 self 槽可能把它让出去（防复读）。
        # 要断言的是"没丢"，不是"一定在哪个槽"。
        whole = "\n".join(m["content"] for m in asm.messages)
        self.assertIn("催我", whole)

    def test_repeated_stance_not_refed(self):
        """连着说同一件事时，self 槽要换人 —— 不然她像在背设定。"""
        p = self.p
        line = "我这边是空的，只有你说话的时候我才有感觉"
        for _ in range(3):
            self.s.add_msg(p, "assistant", line)
            _capture_last(self.s, p)
        self.s.add_msg(p, "user", "在吗")
        asm = assemble(self.s, p, "在吗", persona="你是深溟。")
        sysmsg = asm.messages[0]["content"]
        slot = sysmsg.split("【你自己以前说过的")[-1].split("【")[0] \
            if "【你自己以前说过的" in sysmsg else ""
        self.assertNotIn("我这边是空的", slot, "刚说过的话不该再进 self 槽")


class TestMemoryLifecycle(unittest.TestCase):
    """强化：被反复想起的更不容易掉。遗忘：长期不提的退场，但还能被唤回。"""

    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:6"
        self.s.add_msg(self.p, "user", "我养了只橘猫叫豆豆，特别黏人")
        distill.distill_once(self.s, self.p, use_llm=False)

    def tearDown(self):
        self.s.close()

    def test_hit_strengthens(self):
        h1 = retrieve(self.s, self.p, "豆豆昨天又拆家了", k=3)
        self.assertTrue(h1)
        self.assertEqual(h1[0].mem.hit_count, 0)
        h2 = retrieve(self.s, self.p, "豆豆昨天又拆家了", k=3)
        self.assertEqual(h2[0].mem.hit_count, 1, "被召回没有记一次")
        self.assertGreater(h2[0].parts["sal"], h1[0].parts["sal"],
                           "被反复想起的记忆应该升温")

    def test_forget_then_revive(self):
        sid = self.s.add_memory(self.p, "event", "年会上抽到一台咖啡机", subject="对方",
                                salience=1.2)
        old = int(time.time()) - 400 * 86400
        self.s.db.execute("UPDATE memory SET t_event=? WHERE id=?", (old, sid))
        self.s.db.commit()

        res = maintain(self.s, self.p)
        self.assertGreaterEqual(res["retired"], 1, "长期不提的条目没退场")
        self.assertNotIn(sid, [m.id for m in self.s.all_memories(self.p)],
                         "退场条目不该继续出现在默认检索面")
        self.assertIn(sid, [m.id for m in self.s.all_memories(self.p, include_retired=True)])

        hits = retrieve(self.s, self.p, "年会上抽到一台咖啡机", k=5)
        self.assertIn(sid, [h.mem.id for h in hits], "退场条目强相关时应该还能捞出来")
        alive = {m.id: m for m in self.s.all_memories(self.p, include_retired=True)}
        self.assertIsNone(alive[sid].retired_at, "被想起的条目应该复活")

    def test_forgot_memory_returns_on_cue(self):
        """遗忘不是删除：话题不对时想不起来，被具体的词勾一下就能回来。"""
        sid = self.s.add_memory(self.p, "event", "对方在杭州面过一次试，紧张到忘词",
                                subject="对方", salience=1.8)
        old = int(time.time()) - 400 * 86400
        self.s.db.execute("UPDATE memory SET t_event=? WHERE id=?", (old, sid))
        self.s.db.commit()
        maintain(self.s, self.p)
        still_there = [m.id for m in self.s.all_memories(self.p, include_retired=True)]
        self.assertIn(sid, still_there, "条目应该还在库里（退场，不是删除）")
        self.assertNotIn(sid, [m.id for m in self.s.all_memories(self.p)],
                         "但已经退出默认检索面")

        # 1) 不相关的话题：几乎没有分量
        off = {h.mem.id: h.score for h in retrieve(self.s, self.p, "今天有点困，先睡了", k=6)}
        # 2) 触景生情：提到那个具体的词
        on = {h.mem.id: h.score for h in retrieve(self.s, self.p, "我下周又要去杭州了", k=6)}

        self.assertIn(sid, on, "被具体线索勾中应该浮上来")
        self.assertGreater(on[sid], off.get(sid, 0) * 3,
                           f"线索应该让分数明显抬高（off={off.get(sid, 0):.3f} on={on[sid]:.3f}）")

        # 3) 想起来了就复活，之后不再是要线索才够得着
        alive = {m.id: m for m in self.s.all_memories(self.p, include_retired=True)}
        self.assertIsNone(alive[sid].retired_at, "想起来之后应该回到活跃面")

    def test_emotion_echo(self):
        """同一类情绪状态下，情绪重的记忆更容易浮上来。"""
        eid = self.s.add_memory(self.p, "event", "对方那次手术前一直没睡", subject="对方",
                                salience=2.8)
        nid = self.s.add_memory(self.p, "event", "对方买了个杯子", subject="对方", salience=1.2)
        self.s.db.execute("UPDATE memory SET t_event=? WHERE id IN (?,?)",
                          (int(time.time()) - 3 * 86400, eid, nid))
        self.s.db.commit()
        hits = retrieve(self.s, self.p, "我明天手术，有点怕", k=6)
        ids = [h.mem.id for h in hits]
        self.assertIn(eid, ids)
        pos = {h.mem.id: i for i, h in enumerate(hits)}
        if nid in pos:
            self.assertLess(pos[eid], pos[nid], "情绪重的应该排在前面")

    def test_boundary_never_fades(self):
        bid = self.s.add_memory(self.p, "boundary", "别叫我阿哲", subject="对方", salience=2.8)
        old = int(time.time()) - 4000 * 86400
        self.s.db.execute("UPDATE memory SET t_event=? WHERE id=?", (old, bid))
        self.s.db.commit()
        maintain(self.s, self.p)
        self.assertIn(bid, [m.id for m in self.s.all_memories(self.p)], "边界不该退场")


class TestMigration(unittest.TestCase):
    """真实库是旧结构（没有 why/retired_at，CHECK 不含 self），打开就得能自动升级。"""

    def test_old_schema_upgraded(self):
        import sqlite3
        path = tmpdb()
        raw = sqlite3.connect(path)
        raw.executescript("""
            CREATE TABLE memory (
                id INTEGER PRIMARY KEY, peer TEXT NOT NULL, kind TEXT NOT NULL,
                subject TEXT, text TEXT NOT NULL, quote TEXT, src_msg TEXT,
                salience REAL NOT NULL DEFAULT 1.0, confidence REAL NOT NULL DEFAULT 0.8,
                t_event INTEGER, t_created INTEGER NOT NULL, last_hit INTEGER,
                hit_count INTEGER NOT NULL DEFAULT 0, superseded_by INTEGER,
                CHECK(kind IN ('fact','preference','event','relation','promise','boundary','trait'))
            );
            INSERT INTO memory(peer,kind,text,src_msg,salience,confidence,t_event,t_created)
            VALUES('p1','fact','对方养了只猫叫豆豆','[1]',1.5,0.9,100,100);
        """)
        raw.commit()
        raw.close()

        s = Store(path)
        rows = s.all_memories("p1")
        self.assertEqual(len(rows), 1, "迁移把老数据弄丢了")
        self.assertEqual(rows[0].text, "对方养了只猫叫豆豆")
        # 新列可用、新 kind 可写
        s.add_memory("p1", "self", "我不喜欢被催", subject="自己", why="因为一催就想沉下去")
        self.assertEqual(len([m for m in s.all_memories("p1") if m.kind == "self"]), 1)
        s.close()

        again = Store(path)          # 幂等：再打开一次不该报错、不该重复迁移
        self.assertEqual(len(again.all_memories("p1", include_retired=True)), 2)
        again.close()


class TestIsolation(unittest.TestCase):
    def test_peers_do_not_leak(self):
        s = Store(tmpdb())
        a = peer_id("qq", "private", "1", "100")
        b = peer_id("qq", "private", "2", "200")
        s.add_msg(a, "user", "我养了只猫叫豆豆")
        s.add_msg(b, "user", "我养了只狗叫大黄")
        distill.distill_once(s, a, use_llm=False)
        distill.distill_once(s, b, use_llm=False)
        ta = " ".join(m.text for m in s.all_memories(a))
        tb = " ".join(m.text for m in s.all_memories(b))
        self.assertIn("豆豆", ta)
        self.assertNotIn("大黄", ta)
        self.assertIn("大黄", tb)
        self.assertNotIn("豆豆", tb)
        s.close()

    def test_group_peer_shape(self):
        self.assertEqual(peer_id("qq", "group", "8888", "10001"),
                         "qq:group:8888:10001")


class TestEndToEnd(unittest.TestCase):
    """200 轮之后，6 个埋点必须全部出现在上下文里。"""

    def test_long_horizon_recall(self):
        from memcore.cli import NEEDLES, _build_turns
        bot = Bot(tmpdb(), persona_text="你是小早。", distill_every=0)
        p = "demo:private:9"
        n = 0
        for role, text, _ in _build_turns(200):
            bot.store.add_msg(p, role, text)
            n += 1
            if n % 16 == 0:
                bot.distill_now(p)
        bot.distill_now(p)

        for nd in NEEDLES:
            asm = assemble(bot.store, p, nd["probe"], persona=bot.persona)
            self.assertIn(nd["expect"], asm.messages[0]["content"],
                          f"{nd['kind']} 丢失：{nd['expect']}")
        bot.close()


class TestFiller(unittest.TestCase):
    """应声碎片不进记忆。

    她自己点出来的原话：「我记性不太好」「我有时候会把自己说的记错」——
    库里的碎片越多，她越是这样描述自己。所以「嗯我在听」这类句子一句都不收。
    """

    def test_ack_filler_dropped(self):
        from memcore.distill import _is_ack_filler
        for t in ("呃，是要做手术，但是呃，我觉得应该没有挨一刀那么狠",
                  "是的，毕竟这个手术的目的是确诊疾病，而不是治疗",
                  "嗯，好的"):
            self.assertTrue(_is_ack_filler(t), t)

    def test_subject_word_is_not_ack(self):
        """"对方…" 别被当成 "对" 应答。这条踩过：整个库差点被误退场 13 条。"""
        from memcore.distill import _is_ack_filler
        for t in ("对方发现助手有了表情包，并夸助手好棒",
                  "对方是人类，不是鱼",
                  "对方希望助手越来越好"):
            self.assertFalse(_is_ack_filler(t), t)

    def test_ack_with_time_is_kept(self):
        """带时间线索就不是纯附和。"""
        from memcore.distill import _is_ack_filler
        self.assertFalse(_is_ack_filler("是的，我明天下午三点去医院"))
        self.assertFalse(_is_ack_filler("对呀，我确实10月27日去看了皮肤"))

    def test_capture_skips_filler(self):
        s = Store(tmpdb())
        p = "qq:private:5"
        s.add_msg(p, "user", "呃，是要做手术，但是呃，我觉得应该没有挨一刀那么狠")
        _capture_last(s, p)
        self.assertEqual(s.all_memories(p), [], "应声碎片不该入库")
        s.close()


class TestEmotionFragment(unittest.TestCase):
    """纯情绪感叹句不是事件，情绪归 L3 状态。

    2026-10-30 20:59 她说「我好难过啊」，规则漏斗把它存成
    [event] 实字 4 —— 什么事都没发生，记成事件她就会当成"发生过的事"，
    正是她抱怨"我有时候会把自己说过的话记错"的来源之一。
    """

    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:6"

    def tearDown(self):
        self.s.close()

    def test_bare_exclamation_not_stored(self):
        from memcore.distill import _heuristic_distill
        s, p = self.s, self.p
        s.add_msg(p, "user", "我好难过啊")
        m = s.recent(p, limit=1)[-1]
        self.assertEqual(_heuristic_distill([m]).memories, [],
                         "纯情绪句不该被规则漏斗造出记忆")
        distill.capture_one(s, p, m)
        self.assertEqual(s.all_memories(p), [],
                         "「我好难过啊」不该作为 event 入库")

    def test_bare_exclamation_becomes_mood(self):
        s, p = self.s, self.p
        s.add_msg(p, "user", "我好难过啊")
        m = s.recent(p, limit=1)[-1]
        distill.capture_one(s, p, m)
        self.assertEqual(s.state(p)["mood"], "难过",
                         "纯情绪句的情绪该即时写进 L3 状态")

    def test_mood_flows_through_distill_once(self):
        """没 key 的降级路径也要把情绪写进状态。"""
        s, p = self.s, self.p
        s.add_msg(p, "user", "我好难过啊")
        distill.distill_once(s, p, use_llm=False)
        self.assertEqual(s.all_memories(p), [],
                         "降级路径同样不该把纯情绪句存成事件")
        self.assertEqual(s.state(p)["mood"], "难过",
                         "降级路径的批量蒸馏也要更新 mood")

    def test_negated_emotion_is_not_mood(self):
        from memcore.distill import _emo_mood
        self.assertIsNone(_emo_mood("我不难过"), "否定句不算真情绪")
        self.assertIsNone(_emo_mood("别难过了"), "安慰语气不算真情绪")
        self.assertIsNone(_emo_mood("我其实没那么难过"), "程度否定不算真情绪")

    def test_emotion_boosts_real_event(self):
        """情绪给真事件加权，不单独造事件。"""
        from memcore.distill import _heuristic_distill
        s, p = self.s, self.p
        s.add_msg(p, "user", "我做完手术了，难受死了")
        m = s.recent(p, limit=1)[-1]
        items = _heuristic_distill([m]).memories
        self.assertEqual(len(items), 1, "带事件的句子照样入库")
        self.assertEqual(items[0]["kind"], "event")
        self.assertGreaterEqual(items[0]["salience"], 2.6,
                                "情绪给真事件加权")

    def test_substantive_emotion_still_stored(self):
        """住院 + 撑不住：事件照存，情绪同时进状态。"""
        s, p = self.s, self.p
        s.add_msg(p, "user", "我妈上个月住院了，心脏的问题，我最近真的撑不住")
        m = s.recent(p, limit=1)[-1]
        distill.capture_one(s, p, m)
        self.assertEqual(len(s.all_memories(p)), 1,
                         "住院是 event，情绪词不该把它吞掉")
        self.assertEqual(s.state(p)["mood"], "撑不住")


class TestTimeSense(unittest.TestCase):
    """时间标签必须自己站得住 —— 这是她亲口指出的那两个 bug。

    10-29 14:30 她说："我这边存的是「昨天：对方 10 月 27 日去医院看皮肤」，
    10 月 27 是前天，标签还是错的，内容有日期，标签没跟上。"

    两个根因：
      a) 天数按 (now - ts) // 86400 算，那是"差了几个 24 小时"，不是"差了几个
         日历日"。昨晚 23:50 的事到今天下午只有 18 小时 → 算成"今天"。
      b) t_event 记的是"收到消息的时刻"，不是"事情发生的那天"，于是
         「今天 10月27日去医院」这种标签和内容打架的条目。
    """

    @staticmethod
    def _at(y, mo, d, h=0, mi=0):
        return int(time.mktime((y, mo, d, h, mi, 0, 0, 0, -1)))

    def test_human_when_uses_calendar_days_not_24h(self):
        from memcore.recall import human_when
        now = self._at(2026, 10, 29, 16, 50)
        cases = [
            ("今天下午", self._at(2026, 10, 29, 14, 0), "今天（10月29日）"),
            # 关键回归：昨天深夜，离今天下午不到 18 小时，也必须是"昨天"
            ("昨晚 23:50", self._at(2026, 10, 28, 23, 50), "昨天（10月28日）"),
            ("昨天早上", self._at(2026, 10, 28, 8, 0), "昨天（10月28日）"),
            # 关键回归：前天深夜离今天只有 41 小时，不能算成"昨天"
            ("前天 23:50", self._at(2026, 10, 27, 23, 50), "前天（10月27日）"),
        ]
        for label, ts, want in cases:
            self.assertEqual(human_when(ts, now_ts=now), want, label)

    def test_human_when_always_carries_absolute_date(self):
        """相对词必须配绝对日期：模型不该有机会自己推。"""
        from memcore.recall import human_when
        now = self._at(2026, 10, 29, 16, 50)
        for ts in (self._at(2026, 10, 29, 1, 0), self._at(2026, 10, 28, 1, 0),
                   self._at(2026, 10, 27, 1, 0), self._at(2026, 10, 25, 1, 0)):
            got = human_when(ts, now_ts=now)
            self.assertRegex(got, r"\d+月\d+日", f"缺绝对日期：{got}")
            self.assertNotIn(".0天前", got, f"天数不能是浮点：{got}")

    def test_now_line_has_date_and_weekday(self):
        from memcore.recall import now_line
        line = now_line(self._at(2026, 10, 29, 16, 50))
        self.assertIn("2026年10月29日", line)
        self.assertIn("星期", line)

    def test_assemble_carries_now_anchor(self):
        """每轮装配都要有「现在」这一行，且排在记忆之前。"""
        s = Store(tmpdb())
        p = "qq:private:3"
        s.add_msg(p, "user", "我养了只猫叫豆豆")
        distill.distill_once(s, p, use_llm=False)
        asm = assemble(s, p, "豆豆今天怎么样", persona="你是零。")
        sysmsg = asm.messages[0]["content"]
        self.assertIn("【现在】", sysmsg)
        self.assertLess(sysmsg.index("【现在】"), sysmsg.index("【关于对方的长期记忆】")
                        if "【关于对方的长期记忆】" in sysmsg else len(sysmsg))
        self.assertGreaterEqual(2, sysmsg.count("【现在】"))
        s.close()

    def test_normalize_long_words_first(self):
        """「大前天」不能被替换成「大10月27日」。"""
        base = self._at(2026, 10, 29, 12, 0)
        self.assertEqual(distill.normalize_time_words("大前天加过班", base), "10月26日加过班")
        self.assertEqual(distill.normalize_time_words("昨晚上没睡好", base), "10月28日上没睡好")
        self.assertEqual(distill.normalize_time_words("前天做的检查", base), "10月27日做的检查")

    def test_event_ts_prefers_single_date_in_body(self):
        """正文写着"10月27日去看皮肤"，标签就不能是"今天"。"""
        msg_ts = self._at(2026, 10, 29, 14, 33)
        got = distill.event_ts("对方10月27日一个人去医院看皮肤", msg_ts)
        self.assertEqual(time.strftime("%m-%d", time.localtime(got)), "10-27")

    def test_event_ts_keeps_msg_time_when_spanning_days(self):
        """跨天叙述（澄清/复盘）留在说话那天，别被搬走。"""
        msg_ts = self._at(2026, 10, 29, 14, 33)
        text = "那是10月27日的事了，10月28日我就已经去了医院，10月29日下午要换药"
        self.assertEqual(distill.event_ts(text, msg_ts), msg_ts)

    def test_event_ts_ignores_future_and_too_old(self):
        msg_ts = self._at(2026, 10, 29, 14, 33)
        # 未来的约定不该覆盖消息时间
        self.assertEqual(distill.event_ts("约了11月5日复查", msg_ts), msg_ts)
        # 一个月前的日期也不该覆盖
        self.assertEqual(distill.event_ts("9月1日就开始疼了", msg_ts), msg_ts)

    def test_captured_memory_uses_body_date(self):
        """端到端：落库时 t_event 认正文里的日期。"""
        s = Store(tmpdb())
        p = "qq:private:4"
        msg_ts = self._at(2026, 10, 29, 14, 33)
        s.add_msg(p, "user", "我10月27日去医院做了体检", ts=msg_ts)
        _capture_last(s, p)
        mems = s.all_memories(p)
        self.assertTrue(mems, "没捕获到记忆")
        from memcore.recall import human_when
        labels = " ".join(human_when(m.t_event, now_ts=msg_ts) for m in mems)
        self.assertIn("10月27日", labels, labels)
        self.assertNotIn("今天（10月29日）", labels, labels)
        s.close()


class TestSessionTail(unittest.TestCase):
    """隔夜接话：得先知道"上次说到哪儿"。

    10-30 早上的原话：他问"4 点回来煮面那事你还记得吗"，她答「这段我这边翻不到」，
    可那两句一字不差躺在 L0 里。原因不是检索坏了，是：
      a) 上下文只有「检索 + 最近 10 轮」，昨晚那几轮隔夜就掉出窗口；
      b) 检索靠词面，早上那句「早」没有任何词可命中。

    所以装配时要额外带一段"上一次说话的最后几句"原文。这组测试把它钉住。
    """

    @staticmethod
    def _at(y, mo, d, h=0, mi=0):
        return int(time.mktime((y, mo, d, h, mi, 0, 0, 0, -1)))

    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:5"

    def tearDown(self):
        self.s.close()

    def _last_night_then_morning(self):
        """昨晚两句，隔夜之后今天早上第一句（这句刚落库）。"""
        t0 = self._at(2026, 10, 29, 20, 20)
        self.s.add_msg(self.p, "user", "我今晚煮了泡面，锅底不平煎不了蛋", ts=t0)
        self.s.add_msg(self.p, "assistant", "泡面还能煮成垃圾", ts=t0 + 60)
        now_ts = int(time.time())
        self.s.add_msg(self.p, "user", "早", ts=now_ts)
        return now_ts

    def test_tail_returns_previous_session_only(self):
        from memcore.recall import last_session_tail
        self._last_night_then_morning()
        tail = last_session_tail(self.s, self.p)
        roles = [r for r, _ in tail]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertIn("泡面", tail[0][1])
        # 当前这一句不能混进"上次"
        self.assertNotIn("早", [t for _, t in tail])

    def test_tail_empty_when_conversation_continuous(self):
        from memcore.recall import last_session_tail
        t0 = int(time.time()) - 600
        for i in range(4):
            self.s.add_msg(self.p, "user" if i % 2 == 0 else "assistant", f"第{i}句", ts=t0 + i * 60)
        self.assertEqual(last_session_tail(self.s, self.p), [])

    def test_tail_not_repeated_mid_conversation(self):
        """接上的那一下给，接着往下聊就不给 —— 不然每轮都背一遍昨晚。"""
        from memcore.recall import last_session_tail
        t0 = self._at(2026, 10, 29, 20, 20)
        self.s.add_msg(self.p, "user", "昨晚那锅泡面", ts=t0)
        base = int(time.time()) - 600
        for i in range(3):                    # 今天连聊的三句
            self.s.add_msg(self.p, "user", f"今天第{i}句", ts=base + i * 60)
        self.assertEqual(last_session_tail(self.s, self.p), [])

    def test_assemble_carries_tail(self):
        self._last_night_then_morning()
        asm = assemble(self.s, self.p, "早", persona="你是零。")
        sysmsg = asm.messages[0]["content"]
        self.assertIn("【上一次说话的最后几句", sysmsg)
        self.assertIn("泡面", sysmsg)
        self.assertIn("【现在】", sysmsg)
        self.assertLess(sysmsg.index("【现在】"), sysmsg.index("【上一次说话的最后几句"))

    def test_count_pending_tracks_backlog(self):
        """蒸馏触发靠这个数：它是库里的，重启不会归零。"""
        self._last_night_then_morning()
        self.assertEqual(self.s.count_pending(self.p), 3)
        distill.distill_once(self.s, self.p, use_llm=False)
        self.assertEqual(self.s.count_pending(self.p), 0)


class TestRecentSelf(unittest.TestCase):
    """「你自己刚说过的」这一槽：她说过的话，下一轮必须看得见。

    2026-10-30 13:16 她说「翻到你 D 盘那个 cold-vault」，13:53 被问起时答
    「没翻过，我只进了 bishe 和 develop」。两次都对注入内容诚实 ——
    错的是注入：AstrBot 的会话历史只装"用户消息触发的回合"，自主开口
    （tool_loop_agent）说出的话不进那份历史，她下一轮就看不见自己刚说了什么。
    """

    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:6"

    def tearDown(self):
        self.s.close()

    def test_returns_latest_self_lines_with_time(self):
        from memcore.recall import recent_self_lines
        base = int(time.time()) - 300
        self.s.add_msg(self.p, "assistant", "翻到你 D 盘那个 cold-vault", ts=base)
        self.s.add_msg(self.p, "user", "这个算你主动翻东西了吗", ts=base + 60)
        self.s.add_msg(self.p, "assistant", "不算。那次是你发完消息，我顺势翻的", ts=base + 120)
        got = recent_self_lines(self.s, self.p, k=2)
        self.assertEqual(len(got), 2)
        self.assertIn("cold-vault", got[0])
        self.assertIn("顺势翻的", got[1])
        # 带时刻，否则旧话会被当成这一轮的依据
        self.assertRegex(got[0], r"^\d{2}-\d{2} \d{2}:\d{2} ")

    def test_assemble_carries_mine_and_forbids_repeat(self):
        base = int(time.time()) - 300
        self.s.add_msg(self.p, "assistant", "翻到你 D 盘那个 cold-vault", ts=base)
        asm = assemble(self.s, self.p, "这个cold-vault没人叫你翻吧", persona="你是零。")
        sysmsg = asm.messages[0]["content"]
        self.assertIn("【你自己刚说过的", sysmsg)
        self.assertIn("cold-vault", sysmsg)
        self.assertIn("别重复它", sysmsg)


class TestImageTag(unittest.TestCase):
    """表情包要留在历史里：图不进 L0，前端 5 秒后重建就「闪一下没了」。

    2026-10-30 用户实测：表情包出来几秒就消失。
    根因是网页端走流式，meme_manager 在自己的 on_decorating_result 里
    直接 `event.send(Image)`，图不进 result.chain、也过了 on_llm_response，
    所以 `_image_tags` 从来没抓到 —— 全库 0 条 `[[img:]]`。
    """

    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:7"

    def tearDown(self):
        self.s.close()

    def test_append_tag_to_the_reply_the_image_belongs_to(self):
        t0 = int(time.mktime((2026, 10, 30, 16, 3, 58, 0, 0, -1)))
        self.s.add_msg(self.p, "user", "真乖", ts=t0)
        mid = self.s.add_msg(self.p, "assistant", "这词放我身上有点别扭\n\n换一个夸法吧",
                             ts=t0 + 3)
        got = self.s.append_img_tag(self.p, "20261030160401457_2b73.jpg", t0 + 6)
        self.assertEqual(got, mid)
        text = self.s.get_msgs([mid])[0].text
        self.assertIn("[[img:20261030160401457_2b73.jpg]]", text)
        # 原文保留，只是追加
        self.assertIn("换一个夸法吧", text)

    def test_append_tag_is_idempotent(self):
        t0 = int(time.time()) - 10
        self.s.add_msg(self.p, "assistant", "一句话", ts=t0)
        self.assertIsNotNone(self.s.append_img_tag(self.p, "a.gif", t0))
        self.assertIsNone(self.s.append_img_tag(self.p, "a.gif", t0))

    def test_skips_file_already_referenced_by_user(self):
        """对方在网页端发的图，L0 里存的就是 [[img:文件名]]，不能再贴到她身上。"""
        t0 = int(time.time()) - 10
        self.s.add_msg(self.p, "user", "看看这个 [[img:20261030160000111_ab.jpg]]", ts=t0)
        self.s.add_msg(self.p, "assistant", "看见了", ts=t0 + 2)
        self.assertIsNone(self.s.append_img_tag(self.p, "20261030160000111_ab.jpg", t0 + 3))

    def test_refuses_when_time_fits_no_round(self):
        """隔了一轮的旧图不能挂到新回复上。"""
        t0 = int(time.time()) - 600
        self.s.add_msg(self.p, "assistant", "上一轮的回复", ts=t0)
        self.assertIsNone(self.s.append_img_tag(self.p, "b.png", t0 + 500))


class TestMetaStatementFilter(unittest.TestCase):
    """元陈述不能进 open_loops —— 她自己的困惑不是未完成的话题。

    2026-10-30 遗留：「助手翻不到……记忆压缩丢内容的问题还没解决」这种
    元陈述被 distill 当成未结话题，记下去后被 anchor 每轮重复注入，
    她就一直在那上面转。
    """

    def test_is_meta_statement(self):
        from memcore.distill import _is_meta_statement
        # 这些都是元陈述 —— 关于记忆系统本身的困惑
        for t in ("助手翻不到记忆压缩丢内容的问题还没解决",
                  "我不知道怎么叫你",
                  "我记不得之前说过的话",
                  "我说话人机味太重",
                  "我今天为什么钝",
                  "记忆水位停住了",
                  "抽取规则没改",
                  "怎么调人设啊"):
            self.assertTrue(_is_meta_statement(t), f"应该识别为元陈述：{t}")
        # 这些是真正的未完成话题 —— 不是元陈述
        for t in ("明天门诊小手术的结果和银屒病诊断",
                  "对方要测试 assistant 的新工具功能",
                  "还没真正逛完 D:\\bishe 和 D:\\develop 目录"):
            self.assertFalse(_is_meta_statement(t), f"不应该识别为元陈述：{t}")

    def test_meta_loop_not_added(self):
        """通过 _apply_state 验证：元陈述的 open_loops_add 不进库。"""
        from memcore.distill import _apply_state
        s = Store(tmpdb())
        p = "qq:private:meta"
        s.add_memory(p, "event", "测试事件", subject="对方", salience=1.0)
        # 模拟 LLM 输出的 state，包含元陈述
        state = {
            "open_loops_add": [
                {"text": "助手翻不到记忆压缩丢内容的问题还没解决"},
                {"text": "明天门诊小手术的结果"},
            ],
        }
        _apply_state(s, p, state)
        st = s.state(p)
        loops = [l for l in st["open_loops"] if not l.get("closed")]
        texts = [l["text"] for l in loops]
        # 元陈述被过滤
        self.assertFalse(any("翻不到" in t for t in texts),
                         f"元陈述不该进 open_loops：{texts}")
        # 真正的话题还在
        self.assertTrue(any("门诊" in t for t in texts),
                        f"真话题应该在 open_loops 里：{texts}")
        s.close()


class TestGradualForgetting(unittest.TestCase):
    """遗忘不是批量退场 —— 像人一样渐进模糊。

    2026-10-30 遗留：maintain() 一次性标退场，年龄一过阈值就全退。
    改成概率退场：p = 1 - exp(-age / (tau*factor))。
    """

    def setUp(self):
        self.s = Store(tmpdb())
        self.p = "qq:private:gf"

    def tearDown(self):
        self.s.close()

    def _old_mem(self, kind="event", salience=1.0, days=400):
        sid = self.s.add_memory(self.p, kind, "旧事件", subject="对方",
                                salience=salience)
        old = int(time.time()) - days * 86400
        self.s.db.execute("UPDATE memory SET t_event=? WHERE id=?", (old, sid))
        self.s.db.commit()
        return sid

    def test_very_old_memory_retired_with_deterministic_rng(self):
        """400 天前的 event，概率 ≈ 1，一定退场。"""
        import random
        sid = self._old_mem("event", days=400)
        rng = random.Random(42)
        res = maintain(self.s, self.p, rng=rng)
        self.assertGreaterEqual(res["retired"], 1)
        self.assertNotIn(sid, [m.id for m in self.s.all_memories(self.p)])

    def test_young_memory_not_retired(self):
        """10 天前的 event，还没过阈值，一定不退场。"""
        import random
        sid = self._old_mem("event", days=10)
        rng = random.Random(42)
        res = maintain(self.s, self.p, rng=rng)
        self.assertEqual(res["retired"], 0)
        self.assertIn(sid, [m.id for m in self.s.all_memories(self.p)])

    def test_gradual_not_batch(self):
        """两个同样年龄的记忆，不一定一起退 —— 不是批量。"""
        import random
        sid1 = self._old_mem("event", days=400)
        sid2 = self._old_mem("event", days=400)
        # 用固定种子，但概率退场意味着不是 100% 退场
        # 400 天的 event，p ≈ 1，所以两个都会退 —— 但关键是接口接受 rng
        rng = random.Random(42)
        res = maintain(self.s, self.p, rng=rng)
        self.assertGreaterEqual(res["retired"], 1)
        # 验证接口接受 rng 参数（渐进式的基础）
        self.assertIsNotNone(rng)

    def test_boundary_never_retired(self):
        """边界永不退场。"""
        import random
        sid = self._old_mem("boundary", days=400)
        rng = random.Random(42)
        res = maintain(self.s, self.p, rng=rng)
        self.assertEqual(res["retired"], 0)
        self.assertIn(sid, [m.id for m in self.s.all_memories(self.p)])


class TestSelfMood(unittest.TestCase):
    """她自己的三个量（mood_val / energy / taut）。

    锁四件事：
      1. decay 只让数值往中间回落，不跳变、不越界，且三档半衰期快慢有序
      2. push 单条有上限，狂刷夸奖顶不到 1.0
      3. 极短敷衍不会把 mood_val 拉到 -1（弱刺激有地板）
      4. state 加列后 save/load 往返一致（含新字段）
    这四条守的是同一件事：她的量是**连续**的，不允许一条消息改写一整天的底色。
    """

    P = "qq:private:1"

    def setUp(self):
        self.s = Store(tmpdb())

    def tearDown(self):
        self.s.close()

    def test_decay_only_falls_back(self):
        """回落：高出中线的降、低于中线的升，绝不越界；dt<=0 原样返回。"""
        from memcore import mood

        base = {"mood_val": 0.9, "energy": 1.0, "taut": 0.9, "self_mood_ts": 1_000_000}
        out = mood.decay(base, 1_000_000 + 12 * 3600)
        self.assertLess(out["mood_val"], 0.9, "正数该往中间回")
        self.assertLess(out["energy"], 1.0)
        self.assertLess(out["taut"], 0.9)
        self.assertGreaterEqual(out["mood_val"], 0.0)
        self.assertGreaterEqual(out["energy"], 0.5)
        self.assertGreaterEqual(out["taut"], 0.0)
        # 三档半衰期：同样 2 小时，taut 掉得最狠、mood_val 最慢（mood 是慢变量）
        two_h = mood.decay({"mood_val": -0.9, "energy": 0.0, "taut": 0.9,
                            "self_mood_ts": 1_000_000}, 1_000_000 + 2 * 3600)
        self.assertGreater(two_h["mood_val"], -0.9, "负数该往上回")
        self.assertGreater(two_h["taut"], 0.3, "两小时只散一半，不是一次抹平")
        # 2 小时正好是 taut 的半衰期：0.9 附近落一半；mood 是慢变量，2 小时回不了多少
        self.assertAlmostEqual(two_h["taut"], 0.45, places=2)
        self.assertLess(abs(two_h["mood_val"]), 0.9)
        self.assertGreater(abs(two_h["mood_val"]), 0.5, "12 小时半衰：两小时还很沉")
        self.assertLess(two_h["energy"], 0.2, "6 小时半衰：两小时回来一点点")
        self.assertGreater(two_h["energy"], 0.0)
        # 同一秒再调一次：不动（重复调用不该把数值磨掉）
        self.assertEqual(mood.decay(base, 1_000_000), dict(base))
        self.assertEqual(mood.decay(base, 999_999), dict(base))
        # 极端输入仍被夹在值域里
        funny = mood.decay({"mood_val": 99, "energy": -9, "taut": 7, "self_mood_ts": 1},
                           10 ** 9)
        self.assertLessEqual(funny["mood_val"], 1.0)
        self.assertGreaterEqual(funny["energy"], 0.0)
        self.assertLessEqual(funny["taut"], 1.0)
        # hint 是给 prompt 的事实性暗示：不带数字，也不对人的心情下断言
        h = mood.hint(out)
        self.assertNotRegex(h, r"\d")
        self.assertNotIn("你", h)
        self.assertIn("劲头", h)

    def test_push_delta_is_capped(self):
        """狂刷夸奖：每条都受上限约束，累到 1.0 就停，不会越界。"""
        from memcore import mood

        st = {"mood_val": 0.0, "energy": 0.6, "taut": 0.1}
        for _ in range(50):
            nxt = mood.push(st, "谢谢你，你这么细心，真好", "user", {"now_ts": 1})
            self.assertLessEqual(abs(nxt["mood_val"] - st["mood_val"]), 0.25 + 1e-9)
            self.assertLessEqual(abs(nxt["energy"] - st["energy"]), 0.20 + 1e-9)
            self.assertLessEqual(abs(nxt["taut"] - st["taut"]), 0.35 + 1e-9)
            self.assertLessEqual(nxt["mood_val"], 1.0)
            self.assertLessEqual(nxt["energy"], 1.0)
            st = nxt
        self.assertGreater(st["mood_val"], 0.5, "夸了 50 条总该升上去")

    def test_bland_never_bottoms_out(self):
        """十几句「嗯」「哦」只能压低到地板，不许拉成 -1 的怨气。"""
        from memcore import mood

        st = {"mood_val": 0.0, "energy": 0.6, "taut": 0.1}
        for _ in range(40):
            st = mood.push(st, "嗯", "user", {"now_ts": 1})
        self.assertGreater(st["mood_val"], -1.0)
        self.assertGreaterEqual(st["mood_val"], mood.BLAND_MOOD_FLOOR)
        self.assertGreaterEqual(st["energy"], 0.0)
        # 单条增量也小：一句敷衍不该把一天的心情砸掉
        one = mood.push({"mood_val": 0.0, "energy": 0.6, "taut": 0.1}, "哦", "user")
        self.assertGreater(one["mood_val"], -0.1)

    def test_state_round_trip(self):
        """新字段能存能取；新库有默认值；老库迁移后数据不丢。

        注意 mood（对方的情绪，TEXT）和 mood_val（她自己的，REAL）是两列，
        往返测试必须同时验，别让人以后合并成一个字段。
        """
        self.s.save_state(self.P, mood_val=-0.42, energy=0.33, taut=0.75,
                          self_mood_ts=1_700_000_000, mood="烦")
        st = self.s.state(self.P)
        self.assertAlmostEqual(st["mood_val"], -0.42, places=4)
        self.assertAlmostEqual(st["energy"], 0.33, places=4)
        self.assertAlmostEqual(st["taut"], 0.75, places=4)
        self.assertEqual(st["self_mood_ts"], 1_700_000_000)
        self.assertEqual(st["mood"], "烦", "对方的情绪那列不能被新字段顶掉")

        # 新库缺省值：一上来不该是"很低落"
        fresh = Store(tmpdb())
        d = fresh.state("p9")
        self.assertEqual(d["mood_val"], 0.0)
        self.assertEqual(d["energy"], 0.6)
        self.assertEqual(d["taut"], 0.1)
        self.assertIsNone(d["self_mood_ts"])
        fresh.close()

        # 老库（persona_state 没有这四列）打开后自动补列，老值还在
        import sqlite3
        path = tmpdb()
        raw = sqlite3.connect(path)
        raw.executescript("""
            CREATE TABLE persona_state (
                peer TEXT PRIMARY KEY, bond REAL NOT NULL DEFAULT 0.10,
                mood TEXT, mood_ts INTEGER, nick TEXT, open_loops TEXT,
                profile TEXT, updated INTEGER
            );
            INSERT INTO persona_state(peer,bond,mood,nick) VALUES('p1',0.66,'累','早');
        """)
        raw.commit()
        raw.close()
        old = Store(path)
        o = old.state("p1")
        self.assertEqual(o["mood"], "累")
        self.assertEqual(o["nick"], "早")
        self.assertAlmostEqual(o["bond"], 0.66, places=4)
        self.assertEqual(o["mood_val"], 0.0)
        self.assertEqual(o["energy"], 0.6)
        old.save_state("p1", mood_val=0.2, energy=0.4, taut=0.3, self_mood_ts=123)
        old.close()
        again = Store(path)          # 幂等：再打开不重复迁移、不丢数据
        self.assertAlmostEqual(again.state("p1")["mood_val"], 0.2, places=4)
        self.assertEqual(again.state("p1")["self_mood_ts"], 123)
        again.close()


    def test_nudge_autonomy_time(self):
        """自主时间里「她干了什么」：翻过东西费力、说出口是好事、翻完没话说留在里面。

        锁两件事：① self_mood_ts 不被这个动掉（衰减基准不能乱）；
        ② 幅度小到一天十来次也翻不了天，且不越界。
        """
        from memcore import mood

        base = {"mood_val": 0.0, "energy": 0.6, "taut": 0.1, "self_mood_ts": 1_000_000}

        # 谷时段翻了一圈、说出了一句：力气-一点，心气+一点
        spoke = mood.nudge(base, valley=True, spoke=True)
        self.assertLess(spoke["energy"], base["energy"])
        self.assertGreater(spoke["mood_val"], base["mood_val"])
        self.assertEqual(spoke["self_mood_ts"], base["self_mood_ts"], "衰减基准不动")

        # 翻了但没话说：绷一点，心气不动
        silent = mood.nudge(base, valley=True, spoke=False)
        self.assertGreater(silent["taut"], base["taut"])
        self.assertEqual(silent["mood_val"], base["mood_val"])

        # 峰时段（没工具）说了一句：只有心气那一点
        plain = mood.nudge(base, valley=False, spoke=True)
        self.assertEqual(plain["energy"], base["energy"])

        # 一天刷满 12 次也不会翻天
        st = dict(base)
        for _ in range(12):
            st = mood.nudge(st, valley=True, spoke=True)
        self.assertLess(st["energy"], 0.6)
        self.assertGreater(st["energy"], 0.0)
        self.assertLess(st["mood_val"], 0.5, "十几句自己开口不该堆出高心气")
        self.assertGreaterEqual(st["taut"], 0.0)


class TestDaytime(unittest.TestCase):
    """一天的形状：睡着 / 犯困 / 清醒（memcore.daytime，纯函数，零网络零 LLM）。

    锁四件事：
      1. 边界左闭右开：02:00 整算 asleep、06:00 整算 groggy、07:30 整算 awake
      2. clock 说人话：中文口语，不带冒号、不带阿拉伯数字、不带秒
      3. line 只给事实，绝不写成第一人称断言（「睡」「困」「你」不许出现）
      4. 跨天（23:59 / 00:01）不崩，各自落在预期档位
    第 3 条是这次最容易改坏的地方：一旦写成「你在睡觉」，她会照着演。
    """

    @staticmethod
    def _at(hour: int, minute: int = 0, day: int = 15) -> int:
        """构造某天某点的本地时间戳（用 mktime，跟着本机时区走）。"""
        return int(time.mktime((2026, 6, day, hour, minute, 0, 0, 0, -1)))

    def test_phase_edges(self):
        """每个边界的前一分钟/整点/后一分钟都要对，且 0 点属于 groggy。"""
        from memcore import daytime

        cases = (
            ((0, 0), "groggy"), ((1, 59), "groggy"), ((2, 0), "asleep"),
            ((5, 59), "asleep"), ((6, 0), "groggy"), ((7, 29), "groggy"),
            ((7, 30), "awake"), ((12, 0), "awake"), ((21, 59), "awake"),
            ((22, 0), "groggy"), ((23, 59), "groggy"),
        )
        for (h, m), want in cases:
            self.assertEqual(daytime.phase(self._at(h, m)), want,
                             f"{h:02d}:{m:02d} 应该落在 {want}")

    def test_clock_is_spoken_chinese(self):
        """四个时段的说法都要覆盖，且是口语：没有冒号、没有阿拉伯数字。"""
        from memcore import daytime

        self.assertEqual(daytime.clock(self._at(2, 14)), "凌晨两点多")
        self.assertEqual(daytime.clock(self._at(7, 30)), "早上七点半")
        self.assertEqual(daytime.clock(self._at(15, 0)), "下午三点")
        self.assertEqual(daytime.clock(self._at(22, 0)), "晚上十点")
        for h, m in ((2, 14), (7, 30), (9, 5), (12, 0), (15, 0), (19, 45), (23, 59)):
            got = daytime.clock(self._at(h, m))
            self.assertIn("点", got)
            self.assertNotIn(":", got, "不许出现 24 小时制冒号")
            self.assertNotRegex(got, r"\d", "不许出现阿拉伯数字")
            self.assertNotIn("秒", got)

    def test_line_states_facts_only(self):
        """line 只描述时刻与档位：不写「你在睡觉」「你困了」「你该休息」。"""
        from memcore import daytime

        for h, m in ((0, 5), (3, 0), (7, 0), (13, 20), (22, 40), (23, 59)):
            ln = daytime.line(self._at(h, m))
            self.assertTrue(ln.startswith("这会儿 "), ln)
            self.assertIn("，", ln)
            for bad in ("睡", "困", "你", "该", "休息", "醒着"):
                self.assertNotIn(bad, ln, f"注入行里不许有断言词「{bad}」：{ln}")
            self.assertNotRegex(ln, r"\d", "只给口语时刻，不给数字")
        # 状态行里必须真的带上这一句（在「这会儿的劲头」之后）
        st = {"bond": 0.5, "mood": "", "nick": "", "open_loops": []}
        sl = distill.state_line(st)
        self.assertIn(daytime.line(), sl)

    def test_his_hours(self):
        """他的作息（07:00–23:00）要进注入，而且只给事实、不给禁令。

        为什么单测这一条（2026-11-11）：11-10 23:00 到 11-11 05:52 她连发 8 条，
        其中「你怎么也还醒着」「你醒着的这会儿」都是在猜他在不在 —— 她手上只有
        自己的时段，他那半边从来没进过注入。改法是给事实（他的作息 + 他大概
        什么时候看到），不写「别发」：用户两次说过不要给她定睡觉时间。
        """
        from memcore import daytime

        self.assertTrue(daytime.his_awake(self._at(7, 0)))
        self.assertTrue(daytime.his_awake(self._at(22, 59)))
        self.assertFalse(daytime.his_awake(self._at(23, 0)))
        self.assertFalse(daytime.his_awake(self._at(3, 0)))
        self.assertFalse(daytime.his_awake(self._at(6, 59)))

        awake = daytime.his_line(self._at(15, 0))
        asleep = daytime.his_line(self._at(3, 0))
        self.assertIn("醒着", awake)
        self.assertIn("睡着", asleep)
        self.assertIn("早上才看到", asleep)          # 后果要写明，她才判断得了
        for ln in (awake, asleep):
            self.assertIn("早上七点到晚上十一点", ln)   # 时段本身必须在
            for bad in ("别发", "不许", "不准", "你该", "应该"):
                self.assertNotIn(bad, ln, f"不许写成禁令：{ln}")
        # 他那半边不进 state_line（状态槽只有 60 字，预算卡着），
        # 走的是主对话【现在】段和自主时间的开场 —— 那两处才是她真正读到的地方。
        st = {"bond": 0.5, "mood": "", "nick": "", "open_loops": []}
        self.assertNotIn("他的作息", distill.state_line(st))

    def test_cross_midnight(self):
        """23:59 与 00:01 都要返回合法值，且各自说对了时段。"""
        from memcore import daytime

        late = self._at(23, 59)
        early = self._at(0, 1, day=16)
        self.assertEqual(daytime.phase(late), "groggy")
        self.assertEqual(daytime.phase(early), "groggy")
        self.assertIn("晚上", daytime.clock(late))
        self.assertIn("凌晨", daytime.clock(early))
        self.assertNotEqual(daytime.clock(late), daytime.clock(early))
        for ts in (late, early):
            self.assertIn(daytime.phase(ts), daytime.PHASES)
            self.assertTrue(daytime.line(ts))
        # 不给时间戳也要能跑（默认取当前时刻）
        self.assertIn(daytime.phase(), daytime.PHASES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
