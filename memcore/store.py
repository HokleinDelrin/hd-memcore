"""SQLite 存储层：L0 原始消息 / L1 分层摘要 / L2 记忆条目 / L3 关系状态。

设计要点（这是整个方案的根）：
  * L0 只追加，永不删除、永不覆盖。任何"记忆"都是它的派生视图。
  * 所有派生条目都带 src_msg_ids + quote，可回溯原话 —— 不许凭空生成。
  * 每个 peer 一个命名空间，群聊里不同人的记忆互不污染。
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .util import dumps, loads, now, tokenize


def backup_db(src: str | Path, dest: str | Path) -> Path:
    """在线备份一份记忆库（WAL 模式下唯一可靠的做法）。

    为什么不能用 `shutil.copy2`：库跑在 WAL 模式，写入先进 `-wal`，
    主文件只在 checkpoint 时更新。AstrBot 常驻握着连接，checkpoint 很少发生 ——
    于是 copy 出来的"备份"是几小时前的旧库：大小、时间戳都像新的，内容却没有今天的改动。
    2026-10-30 实测：四个备份全是 09:00 的快照，而 09:00 之后写了几十条消息。
    sqlite 的 backup API 走的是正在运行的库，拿到的一致快照包含 WAL 里的内容。

    Args:
        src: 源库路径。
        dest: 目标路径（父目录不存在会自动建）。

    Returns:
        目标路径。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(src))
    try:
        out = sqlite3.connect(str(dest))
        try:
            con.backup(out)
        finally:
            out.close()
    finally:
        con.close()
    return dest

# L2 表的 DDL 单独提出来：迁移（加列 / 放开 kind 的 CHECK）时要复用同一份定义。
# kind 多一个 self —— 记的是「她自己的判断」，第一人称，带理由，见 distill.py。
MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS memory (
    id        INTEGER PRIMARY KEY,
    peer      TEXT NOT NULL,
    kind      TEXT NOT NULL,     -- fact|preference|event|relation|promise|boundary|trait|self
    subject   TEXT,              -- 这条说的是谁（对方 / 自己 / 第三方）
    text      TEXT NOT NULL,     -- 一句话；kind=self 用第一人称「我……」
    why       TEXT,              -- 她给的理由（为什么这么想、为什么改主意）；可空
    quote     TEXT,              -- 原话依据
    src_msg   TEXT,              -- json [msg_id,...]
    salience  REAL NOT NULL DEFAULT 1.0,   -- 1..3，情绪强度/自我披露度
    confidence REAL NOT NULL DEFAULT 0.8,
    t_event   INTEGER,
    t_created INTEGER NOT NULL,
    last_hit  INTEGER,
    hit_count INTEGER NOT NULL DEFAULT 0,
    retired_at INTEGER,          -- 退场（遗忘）时间；非 NULL=已退场，强相关仍可唤回
    superseded_by INTEGER,       -- 被新事实取代则指向新条目，检索默认跳过
    CHECK(kind IN ('fact','preference','event','relation','promise','boundary','trait','self'))
);
CREATE INDEX IF NOT EXISTS idx_mem_peer ON memory(peer, kind, superseded_by);
CREATE INDEX IF NOT EXISTS idx_mem_text ON memory(peer, text);
"""

# L3 里她自己那一侧的状态量（mood.py 推）。单独列出来是为了：
#   1. SCHEMA 建新库、_migrate_state 补老库，两处用同一份定义；
#   2. 默认值与 mood.py 的 DEFAULT_* 对齐 —— 缺省不能让 she 一上来就"很低落"。
STATE_COLUMNS = (
    ("mood_val", "REAL NOT NULL DEFAULT 0.0"),
    ("energy", "REAL NOT NULL DEFAULT 0.6"),
    ("taut", "REAL NOT NULL DEFAULT 0.1"),
    ("self_mood_ts", "INTEGER"),
)

# L0 里她自己打的分（2026-11-12，M2）。同一份定义给两处用：
# SCHEMA 建新库、_migrate_msg 补老库。老库每行自动填默认值，不用搬数据。
MSG_COLUMNS = (
    ("importance", "INTEGER NOT NULL DEFAULT 0"),
    ("mark_why", "TEXT NOT NULL DEFAULT ''"),
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- ---------- L0：原始事件日志（append-only）----------
-- importance / mark_why 是 2026-11-12 加的（M2）：她自己对**某一条消息**打分。
-- 0 = 没标过（跟 3 分"标了但没表态"是两件事，别混）。
-- 只让她标 role='user' 的消息：标自己的会自我强化（她说一句"这条重要"，
-- 下一次召回又把那句抬上来，她自己看见就更像"这确实重要"）。
CREATE TABLE IF NOT EXISTS msg (
    id       INTEGER PRIMARY KEY,
    peer     TEXT NOT NULL,
    role     TEXT NOT NULL,          -- user | assistant
    ts       INTEGER NOT NULL,
    text     TEXT NOT NULL,
    raw      TEXT,                    -- 平台原始报文（可选）
    distilled INTEGER NOT NULL DEFAULT 0,  -- 0=待蒸馏 1=已蒸馏
    importance INTEGER NOT NULL DEFAULT 0, -- 她自己标的分 1..5；0=没标
    mark_why   TEXT NOT NULL DEFAULT ''    -- 她标分时写的理由（原话/一句话）
);
CREATE INDEX IF NOT EXISTS idx_msg_peer_ts ON msg(peer, ts);
CREATE INDEX IF NOT EXISTS idx_msg_distill ON msg(peer, distilled, id);

-- 词面召回（中文 bigram 空格串，unicode61 才能切开）
CREATE VIRTUAL TABLE IF NOT EXISTS msg_fts USING fts5(
    body, peer UNINDEXED, msg_id UNINDEXED, tokenize='unicode61'
);

-- ---------- L1：分层滚动摘要 ----------
CREATE TABLE IF NOT EXISTS summary (
    id       INTEGER PRIMARY KEY,
    peer     TEXT NOT NULL,
    level    INTEGER NOT NULL,       -- 1=近段 2=中段 3=远期
    msg_from INTEGER NOT NULL,
    msg_to   INTEGER NOT NULL,
    ts_from  INTEGER NOT NULL,
    ts_to    INTEGER NOT NULL,
    text     TEXT NOT NULL,
    ts       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sum_peer ON summary(peer, level, msg_to);

-- ---------- L2：结构化记忆条目（表结构在 MEMORY_DDL）----------
""" + MEMORY_DDL + """

-- ---------- L3：关系 / 人格状态（每 peer 一行，跨重启存活）----------
-- mood/mood_ts 是**对方的**情绪（从他话里抽的一个词）。
-- mood_val/energy/taut/self_mood_ts 是**她自己的**三个连续量，规则推进，见 mood.py。
-- 两组字段互不相干：别把 mood 当她的心情用，也别拿 mood_val 去覆盖 mood。
CREATE TABLE IF NOT EXISTS persona_state (
    peer      TEXT PRIMARY KEY,
    bond      REAL NOT NULL DEFAULT 0.10,   -- 0..1 亲密度
    mood      TEXT,
    mood_ts   INTEGER,
    mood_val  REAL NOT NULL DEFAULT 0.0,    -- -1..1 她好不好
    energy    REAL NOT NULL DEFAULT 0.6,    -- 0..1 还剩多少力气
    taut      REAL NOT NULL DEFAULT 0.1,    -- 0..1 绷没绷
    self_mood_ts INTEGER,                   -- 她这边最后一次更新的时刻（整数秒）
    nick      TEXT,                          -- 称呼
    open_loops TEXT,                          -- json [{id,text,ts,closed}]
    profile   TEXT,                           -- json 稳定画像缓存（小）
    updated   INTEGER
);

-- ---------- 她的「兴趣」：行为记录（2026-11-03 定论，见 curiosity.py）----------
-- 只记「她自主时间里翻了哪一类东西」，不记她看到了什么。
-- UNIQUE(peer,topic,day) 是刻意的：同一天翻同一类十次只算一天 ——
-- 定论里的「跨天出现 3 次以上才算兴趣」，靠这个唯一键直接数得出来。
-- 「两周没碰就淡化」不做删除、只做窗口过滤，所以老记录留着（她真回去翻时又亮）。
CREATE TABLE IF NOT EXISTS curiosity (
    id     INTEGER PRIMARY KEY,
    peer   TEXT NOT NULL,
    ts     INTEGER NOT NULL,
    day    TEXT NOT NULL,            -- 本地日历日 YYYY-MM-DD
    source TEXT NOT NULL,            -- zero_read | zero_grep | browse
    topic  TEXT NOT NULL,            -- 归一化主题键，如 Vibecoding/DVC 或 词:cold-vault
    detail TEXT,                     -- 原始路径/查询词（给人看，不注入）
    spoke  INTEGER NOT NULL DEFAULT 0,  -- 这一档她有没有说出口过（「会多出来」那一维）
    UNIQUE(peer, topic, day)
);
CREATE INDEX IF NOT EXISTS idx_cur_peer_topic ON curiosity(peer, topic, ts);

-- ---------- 人设漂移自检 ----------
CREATE TABLE IF NOT EXISTS drift_check (
    id     INTEGER PRIMARY KEY,
    peer   TEXT NOT NULL,
    ts     INTEGER NOT NULL,
    probe  TEXT NOT NULL,
    answer TEXT,
    score  REAL
);

-- ---------- 蒸馏水位（每 peer 已蒸馏到哪条消息）----------
CREATE TABLE IF NOT EXISTS watermark (
    peer TEXT PRIMARY KEY,
    msg_id INTEGER NOT NULL DEFAULT 0,
    ts INTEGER
);
"""


@dataclass
class Msg:
    id: int
    peer: str
    role: str
    ts: int
    text: str
    # 她自己给这条消息打的分（2026-11-12，M2）：0=没标，1..5=标过。
    # 带默认值是为了老调用点（测试里大量 `Msg(id, peer, role, ts, text)`）不用改。
    importance: int = 0
    mark_why: str = ""


@dataclass
class Mem:
    id: int
    peer: str
    kind: str
    subject: str | None
    text: str
    why: str | None
    quote: str | None
    src_msg: list[int]
    salience: float
    confidence: float
    t_event: int
    t_created: int
    last_hit: int | None
    hit_count: int
    retired_at: int | None = None

    def active_ts(self) -> int:
        """这条记忆最后一次"活过"的时间：发生时间与最后一次被想起，取较晚的。"""
        return max(self.t_event or 0, self.last_hit or 0)


def _migrate_memory(db: sqlite3.Connection) -> None:
    """把老库的 memory 表升到当前结构：加 why / retired_at，CHECK 放开 self。

    没有独立迁移脚本的代价是一次性重建表；这比"用户真实记忆报错"便宜得多。
    重建只搬数据，不删行，幂等（已是新结构就直接返回）。
    """
    row = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory'"
    ).fetchone()
    if row is None:
        return
    sql = row[0] or ""
    have = {r[1] for r in db.execute("PRAGMA table_info(memory)")}
    if "'self'" in sql and {"why", "retired_at"} <= have:
        return
    keep = [c for c in ("id", "peer", "kind", "subject", "text", "why", "quote", "src_msg",
                        "salience", "confidence", "t_event", "t_created", "last_hit",
                        "hit_count", "retired_at", "superseded_by") if c in have]
    cols = ",".join(keep)
    db.execute("ALTER TABLE memory RENAME TO memory_old")
    db.executescript(MEMORY_DDL)          # 建新表（索引名此刻还被 memory_old 占着）
    db.execute(f"INSERT INTO memory({cols}) SELECT {cols} FROM memory_old")
    db.execute("DROP TABLE memory_old")
    db.executescript(MEMORY_DDL)          # 旧表删掉、索引名释放后再建一次
    db.commit()


def _migrate_state(db: sqlite3.Connection, db_path: str | Path | None = None) -> None:
    """给老的 persona_state 补上她自己那四个状态量（幂等，加列不动数据）。

    为什么不是重建表：L3 每 peer 一行，重建要走全量搬运；加列（ALTER TABLE ADD COLUMN）
    对老行自动填 DEFAULT，风险小得多，而且跑两次也是同一个结果。

    **动 schema 之前先备份**：`persona.db` 是用户真实记忆，WAL 模式，
    所以只走 `backup_db()`（sqlite backup API 拿一致快照），不用 shutil.copy2 ——
    copy 出来的"备份"是几小时前的旧库（2026-10-30 实测过四次）。新库不需要备份（刚建就有列）。
    """
    have = {r[1] for r in db.execute("PRAGMA table_info(persona_state)")}
    if not have:
        return
    missing = [(c, d) for c, d in STATE_COLUMNS if c not in have]
    if not missing:
        return
    if db_path:
        src = Path(str(db_path))
        try:
            if src.is_file() and src.stat().st_size > 0:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup_db(src, src.with_name(src.name + f".bak-{stamp}"))
        except (OSError, sqlite3.Error):
            pass        # 备份失败不该挡住迁移：列加不上整个插件都起不来
    for name, ddl in missing:
        db.execute(f"ALTER TABLE persona_state ADD COLUMN {name} {ddl}")
    db.commit()


def _migrate_msg(db: sqlite3.Connection, db_path: str | Path | None = None) -> None:
    """给老的 msg 表补上她自己打分的两列（幂等，加列不搬数据）。

    为什么加列而不是重建：L0 只追加、有几十万行也不稀奇，重建要整表搬运；
    而两列的默认值都能直接给老行（0 / ''），加完读路径立刻可用。

    **动 schema 之前先备份**：这是他的真实对话原文，WAL 模式，
    所以只走 `backup_db()`（sqlite backup API 拿一致快照），不用 shutil.copy2。
    新库不需要备份（刚建就有列）。
    """
    have = {r[1] for r in db.execute("PRAGMA table_info(msg)")}
    if not have:
        return
    missing = [(c, d) for c, d in MSG_COLUMNS if c not in have]
    if not missing:
        return
    if db_path:
        src = Path(str(db_path))
        try:
            if src.is_file() and src.stat().st_size > 0:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup_db(src, src.with_name(src.name + f".bak-{stamp}"))
        except (OSError, sqlite3.Error):
            pass        # 备份失败不该挡住迁移：列加不上整条读路径都报错
    for name, ddl in missing:
        db.execute(f"ALTER TABLE msg ADD COLUMN {name} {ddl}")
    db.commit()


class Store:
    def __init__(self, path: str = "persona.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        _migrate_memory(self.db)
        _migrate_state(self.db, path)
        _migrate_msg(self.db, path)

    # ---------------- L0 ----------------
    def add_msg(self, peer: str, role: str, text: str, ts: int | None = None,
                raw: Any = None) -> int:
        ts = ts or now()
        cur = self.db.execute(
            "INSERT INTO msg(peer,role,ts,text,raw) VALUES(?,?,?,?,?)",
            (peer, role, ts, text, dumps(raw) if raw is not None else None),
        )
        mid = int(cur.lastrowid)
        self.db.execute(
            "INSERT INTO msg_fts(body,peer,msg_id) VALUES(?,?,?)",
            (" ".join(tokenize(text)), peer, mid),
        )
        self.db.commit()
        return mid

    def recent(self, peer: str, limit: int = 12, before_id: int | None = None) -> list[Msg]:
        sql = "SELECT * FROM msg WHERE peer=?"
        args: list[Any] = [peer]
        if before_id:
            sql += " AND id<?"
            args.append(before_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self.db.execute(sql, args).fetchall()
        return [_msg(r) for r in reversed(rows)]

    def msgs_range(self, peer: str, lo: int, hi: int) -> list[Msg]:
        rows = self.db.execute(
            "SELECT * FROM msg WHERE peer=? AND id>=? AND id<=? ORDER BY id",
            (peer, lo, hi),
        ).fetchall()
        return [_msg(r) for r in rows]

    def count_msgs(self, peer: str) -> int:
        return int(self.db.execute("SELECT COUNT(*) c FROM msg WHERE peer=?", (peer,)).fetchone()["c"])

    def search_lex(self, peer: str, query: str, limit: int = 60) -> list[int]:
        toks = tokenize(query)
        if not toks:
            return []
        # OR 查询：任意 bigram 命中即候选，排序交给 recall 打分
        expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in toks[:40])
        try:
            rows = self.db.execute(
                "SELECT msg_id FROM msg_fts WHERE msg_fts MATCH ? AND peer=? LIMIT ?",
                (expr, peer, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(r["msg_id"]) for r in rows]

    def get_msgs(self, ids: Iterable[int]) -> list[Msg]:
        ids = list(ids)
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT * FROM msg WHERE id IN ({q})", ids).fetchall()
        by_id = {r["id"]: _msg(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    # ---------------- L0 的自评分（M2）----------------
    def mark_msg(self, msg_id: int, level: int, why: str = "",
                 *, peer: str | None = None) -> tuple[bool, str]:
        """她给**某一条消息**打分（2026-11-12，M2）。写 msg.importance / mark_why。

        **只允许标 `role='user'` 的消息**（他的）。标自己那两条会自我强化：
        她说一句"这条重要"，下一次召回又把那句抬上来，她自己读着就更像"这确实重要"，
        两三轮之后她的自评就只剩下自己的回声。

        Args:
            msg_id: L0 里的消息 id（她看得到的是 recent / 引用里的 `#123`）。
            level: 1..5；0 = 撤销这条分数（当没标过）。
            why: 一句话理由，可空。**由她写，不代填**。
            peer: 给了就核对归属，防止她拿别的会话的 id 乱标。

        Returns:
            (ok, 一句话)。ok=False 时那句话是给她看的解释，不是异常。
        """
        try:
            mid = int(msg_id)
        except (TypeError, ValueError):
            return False, "要看哪一条？给个 msg_id（L0 里那个号）"
        lvl = 0
        try:
            lvl = int(level)
        except (TypeError, ValueError):
            return False, "level 给 1 到 5 之间的整数（0 = 撤销）"
        if lvl < 0 or lvl > 5:
            return False, "level 只能是 1..5（0 = 撤销这条分数）"
        row = self.db.execute("SELECT id,peer,role FROM msg WHERE id=?", (mid,)).fetchone()
        if row is None:
            return False, f"库里没有 #{mid} 这条消息"
        if peer and row["peer"] != peer:
            return False, f"#{mid} 不是这个会话里的消息"
        if (row["role"] or "") != "user":
            return False, (f"#{mid} 是你自己说的。你只能标他说的那些 —— "
                           f"标自己的会越标越像，我不想让你那样")
        clean = " ".join(str(why or "").split())[:200]
        self.db.execute("UPDATE msg SET importance=?, mark_why=? WHERE id=?",
                        (lvl, clean, mid))
        self.db.commit()
        return True, (f"#{mid} 标成 {lvl} 分了" if lvl else f"#{mid} 的分数撤销了")

    def msg_importance(self, ids: Iterable[int]) -> dict[int, int]:
        """这些消息她自己打了几分（没标过的给 0）。给蒸馏取 max 用。"""
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id, importance FROM msg WHERE id IN ({q})", ids).fetchall()
        return {int(r["id"]): int(r["importance"] or 0) for r in rows}

    def high_importance_msgs(self, peer: str, lo: int, hi: int, *,
                             min_level: int = 4) -> list[Msg]:
        """这一段区间里她标过 `min_level` 分以上的消息（原文，按 id 升序）。

        摘要合并（`distill._compact`）用它：>=4 分的句子在 L2/L3 里**原样保留**，
        不许被压成一行 —— 一条"他说的那个日期"和她随口一句寒暄，不该同等待遇。
        """
        rows = self.db.execute(
            "SELECT * FROM msg WHERE peer=? AND id>=? AND id<=? AND importance>=? ORDER BY id",
            (peer, int(lo), int(hi), int(min_level)),
        ).fetchall()
        return [_msg(r) for r in rows]

    # ---------------- L2 ----------------
    def add_memory(self, peer: str, kind: str, text: str, *, subject: str | None = None,
                   why: str | None = None, quote: str | None = None,
                   src_msg: list[int] | None = None,
                   salience: float = 1.0, confidence: float = 0.8,
                   t_event: int | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO memory(peer,kind,subject,text,why,quote,src_msg,salience,confidence,"
            "t_event,t_created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (peer, kind, subject, text, why, quote, dumps(src_msg or []), salience,
             confidence, t_event or now(), now()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def all_memories(self, peer: str, include_superseded: bool = False,
                     include_retired: bool = False) -> list[Mem]:
        sql = "SELECT * FROM memory WHERE peer=?"
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        if not include_retired:
            sql += " AND retired_at IS NULL"
        rows = self.db.execute(sql, (peer,)).fetchall()
        return [_mem(r) for r in rows]

    def touch_memory(self, mem_id: int) -> None:
        """被召回一次 = 想起一次。强化就是这一行。"""
        self.db.execute(
            "UPDATE memory SET last_hit=?, hit_count=hit_count+1 WHERE id=?", (now(), mem_id)
        )
        self.db.commit()

    def retire(self, mem_id: int) -> None:
        """遗忘：退出检索面，但不删。强相关时还能被唤回（revive）。"""
        self.db.execute(
            "UPDATE memory SET retired_at=? WHERE id=? AND retired_at IS NULL", (now(), mem_id)
        )
        self.db.commit()

    def revive(self, mem_id: int) -> None:
        self.db.execute("UPDATE memory SET retired_at=NULL, last_hit=? WHERE id=?",
                        (now(), mem_id))
        self.db.commit()

    def supersede(self, old_id: int, new_id: int) -> None:
        self.db.execute("UPDATE memory SET superseded_by=? WHERE id=?", (new_id, old_id))
        self.db.commit()

    # ---------------- L1 ----------------
    def add_summary(self, peer: str, level: int, msg_from: int, msg_to: int,
                    ts_from: int, ts_to: int, text: str) -> int:
        cur = self.db.execute(
            "INSERT INTO summary(peer,level,msg_from,msg_to,ts_from,ts_to,text,ts)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (peer, level, msg_from, msg_to, ts_from, ts_to, text, now()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def summaries(self, peer: str, level: int | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM summary WHERE peer=?"
        args: list[Any] = [peer]
        if level is not None:
            sql += " AND level=?"
            args.append(level)
        sql += " ORDER BY msg_to DESC"
        return self.db.execute(sql, args).fetchall()

    # ---------------- L3 ----------------
    def state(self, peer: str) -> dict:
        row = self.db.execute("SELECT * FROM persona_state WHERE peer=?", (peer,)).fetchone()
        if not row:
            self.db.execute("INSERT INTO persona_state(peer,updated) VALUES(?,?)", (peer, now()))
            self.db.commit()
            row = self.db.execute("SELECT * FROM persona_state WHERE peer=?", (peer,)).fetchone()
        return {
            "peer": row["peer"],
            "bond": row["bond"],
            "mood": row["mood"],
            "mood_ts": row["mood_ts"],
            "nick": row["nick"],
            "open_loops": loads(row["open_loops"], []) or [],
            "profile": loads(row["profile"], {}) or {},
            # 她自己的三个量（mood.py 用）。老库刚迁移完就是默认值。
            "mood_val": _col(row, "mood_val", 0.0),
            "energy": _col(row, "energy", 0.6),
            "taut": _col(row, "taut", 0.1),
            "self_mood_ts": _col(row, "self_mood_ts", None),
        }

    def save_state(self, peer: str, **fields) -> None:
        cur = self.state(peer)  # 保证行存在
        cols, args = [], []
        for k, v in fields.items():
            if k in ("open_loops", "profile"):
                v = dumps(v)
            cols.append(f"{k}=?")
            args.append(v)
        if not cols:
            return
        cols.append("updated=?")
        args.extend([now(), peer])
        self.db.execute(f"UPDATE persona_state SET {','.join(cols)} WHERE peer=?", args)
        self.db.commit()
        _ = cur

    # ---------------- 水位 ----------------
    def watermark(self, peer: str) -> int:
        row = self.db.execute("SELECT msg_id FROM watermark WHERE peer=?", (peer,)).fetchone()
        return int(row["msg_id"]) if row else 0

    def set_watermark(self, peer: str, msg_id: int) -> None:
        self.db.execute(
            "INSERT INTO watermark(peer,msg_id,ts) VALUES(?,?,?)"
            " ON CONFLICT(peer) DO UPDATE SET msg_id=excluded.msg_id, ts=excluded.ts",
            (peer, msg_id, now()),
        )
        self.db.commit()

    def pending(self, peer: str, limit: int = 200) -> list[Msg]:
        w = self.watermark(peer)
        rows = self.db.execute(
            "SELECT * FROM msg WHERE peer=? AND id>? ORDER BY id LIMIT ?", (peer, w, limit)
        ).fetchall()
        return [_msg(r) for r in rows]

    def count_pending(self, peer: str) -> int:
        """还没蒸馏的原文有多少条。

        这个数必须落库：它是「欠了多少活」的唯一持久信号。以前按进程内轮次
        计数触发蒸馏，AstrBot 一重启计数就归零，积压永远攒不到阈值 ——
        水位停在某一轮不再前进，之后说的话一条也没进记忆（10-29 14:32 那次）。
        """
        w = self.watermark(peer)
        row = self.db.execute(
            "SELECT COUNT(*) c FROM msg WHERE peer=? AND id>?", (peer, w)
        ).fetchone()
        return int(row["c"])

    # ---------------- 图片引用 ----------------
    def image_referenced(self, filename: str) -> bool:
        """这个文件名是不是已经出现在任何一条消息里（含对方的 `[[img:…]]`）。

        用途：她发图后要补描述，但对方的图早就在 caption 分支里写进 L0 了 ——
        先问一句，别为一张不该转述的图白调一次识图 API。

        Args:
            filename: 附件文件名，可带路径。

        Returns:
            True 表示已被引用。
        """
        fn = (filename or "").strip().split("/")[-1].split("\\")[-1]
        if not fn:
            return False
        return bool(self.db.execute(
            "SELECT 1 FROM msg WHERE text LIKE ? LIMIT 1", (f"%{fn}%",)).fetchone())

    def append_img_tag(self, peer: str, filename: str, file_ts: int | None = None,
                       *, window: int = 120, note: str = "") -> int | None:
        """把她这轮发出去的图补成 `[[img:文件名]]`（可选带一句画面描述）。

        为什么必须事后补：表情包不走 LLM 的回复链 —— 网页端是流式，
        meme_manager 在 `on_decorating_result` 里自己 `event.send(Image)`，
        那时文本早就写进 L0 了。于是 L0 只有字没有图，前端 5 秒后按 L0
        重建列表，图就「闪一下就没了」（2026-10-30 用户实测）。

        `note` 是 2026-11-01 加的：她说过「表情包我其实看不了画面，只看得到分类名，
        所以我发出去那些，多半是按名字猜的」。发出去之后把那张图的内容转述一句
        一起钉在 L0 里，她下一轮读到自己的话就能知道"我那张到底是什么画面"。
        前端会把 `[[imgnote:…]]` 当噪声去掉，界面上不多一行。

        定位规则：文件名开头就是时间戳（`20261030160401457_2b73.jpg`），
        找「落库时刻 <= 图片时刻」的最后一条回复；没时间戳就退回到最近一条
        且在 `window` 秒内。已经出现在任何消息里的文件不动 ——
        对方在网页端发图时，L0 里存的就是 `[[img:文件名]]`（见 main.py 的 caption 分支），
        那是他的图，不能贴到她身上。

        Returns:
            被补写的消息 id；没补（重复/找不到/不该补）返回 None。
        """
        fn = (filename or "").strip().split("/")[-1].split("\\")[-1]
        if not fn:
            return None
        if not (self.db.execute("SELECT 1 FROM msg WHERE text LIKE ? LIMIT 1",
                                (f"%{fn}%",)).fetchone()):
            ref = int(file_ts) if file_ts else now()
            row = self.db.execute(
                "SELECT id FROM msg WHERE peer=? AND role='assistant' AND ts<=? "
                "ORDER BY id DESC LIMIT 1", (peer, ref)
            ).fetchone()
            if not row:
                row = self.db.execute(
                    "SELECT id FROM msg WHERE peer=? AND role='assistant' "
                    "AND ts>=? ORDER BY id DESC LIMIT 1", (peer, ref - window)
                ).fetchone()
            if not row:
                return None
            mid = int(row["id"])
            body = self.db.execute("SELECT text FROM msg WHERE id=?", (mid,)).fetchone()
            cur = (body["text"] or "") if body else ""
            tag = f"[[img:{fn}]]"
            if tag in cur:
                return None
            # 只往「这轮」的回复上贴：同一条消息的时刻必须离图片足够近
            ts = self.db.execute("SELECT ts FROM msg WHERE id=?", (mid,)).fetchone()
            if ts and abs((file_ts or ts["ts"]) - (ts["ts"] or 0)) > window:
                return None
            clean_note = " ".join(str(note or "").split()).replace("]]", "）")
            tail = f"[[imgnote:{clean_note[:160]}]]" if clean_note else ""
            self.db.execute("UPDATE msg SET text=? WHERE id=?",
                            ((cur.rstrip() + "\n" + tag + tail).strip(), mid))
            self.db.commit()
            return mid
        return None

    def close(self) -> None:
        self.db.close()


def _col(row: sqlite3.Row, name: str, default=None):
    """取列，列不存在或为 NULL 时给默认值。

    为什么要兜：老库还没跑 _migrate_state 时 SELECT * 里没有这几列；
    迁移是幂等的，但读路径不该因为"这一瞬间还没迁"就 KeyError。
    """
    try:
        v = row[name]
    except (IndexError, KeyError):
        return default
    return default if v is None else v


def _msg(r: sqlite3.Row) -> Msg:
    return Msg(r["id"], r["peer"], r["role"], r["ts"], r["text"],
               int(_col(r, "importance", 0) or 0),
               str(_col(r, "mark_why", "") or ""))


def _mem(r: sqlite3.Row) -> Mem:
    keys = r.keys()
    return Mem(
        id=r["id"], peer=r["peer"], kind=r["kind"], subject=r["subject"], text=r["text"],
        why=r["why"] if "why" in keys else None,
        quote=r["quote"], src_msg=loads(r["src_msg"], []) or [], salience=r["salience"],
        confidence=r["confidence"], t_event=r["t_event"], t_created=r["t_created"],
        last_hit=r["last_hit"], hit_count=r["hit_count"],
        retired_at=r["retired_at"] if "retired_at" in keys else None,
    )
