"""memcore —— 零依赖的上下文重装配层。

只做三件事：

    原文永久存（L0，只追加）
    记忆做派生（L1/L2/L3，随时可从原文重算）
    每一轮按预算重新装配（窗口是工作台，不是档案柜）

最小用法：

    from memcore import Store, assemble
    st = Store("mem.db")
    st.add_msg("local:private:0", "user", "我养了只猫叫豆豆")
    ctx = assemble(st, "local:private:0", "我家猫又把耳机线咬了").messages

需要「喂一轮 → 拿回复」的完整回路时再用 `Bot`（生成那一步会去读环境变量里的模型
配置；没有 key 就走离线自述，不发任何网络请求）。
"""
from __future__ import annotations

from .assemble import Assembled, assemble
from .bot import Bot, Reply, dump_md, peer_id
from .store import Mem, Msg, Store

__all__ = [
    "Store", "Msg", "Mem",
    "assemble", "Assembled",
    "Bot", "Reply", "dump_md", "peer_id",
]
