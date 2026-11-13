"""她抄下来的"像人话的句子"（2026-11-04）。

**为什么单列一槽**：她的腔调来自"读什么"—— 读注入里那些条目式公文，写出来就是公文；
读真实的人话，才像人。所以这一槽只放**她自己抄的**句子，不放我们挑的：
我们挑的就变成"教她说话"，她自己抄的才是她的口味。

**筛选放在进门，不放在出门**：生成端设违禁词只会让她更僵（已经踩过：改过头她说话变整齐）。
所以规则全在 `pick()` 里 —— 抄的时候不合格的进不来，进来之后不再管。

文件是她自己的：`<她家>/语感.md`，一行一句，她自己用 zero_home_write 往里加。
"""
from __future__ import annotations

import pathlib

NAME = "语感.md"
MIN_LEN = 4          # 太短的不是句子（"嗯""好"）
MAX_LEN = 60         # 太长的不像随手说的
KEEP = 5             # 每轮只放几句：多了她会变成复读机
_NOISE = ("http://", "https://", "www.", "```")
# 文件开头的模板说明不是"她抄的句子"。真正的原因是文件带 BOM（\ufeff），
# `startswith("#")` 判断失效，于是标题「# 语感」和这句说明一起被当成句子注进去了
# （2026-11-05 在 last_block.txt 里看到 `- # 语感`）。两头都堵：去 BOM + 认模板行。
_TEMPLATE = {
    "# 语感",
    "这里是别人说话里我觉得像人话的句子，一行一句。",
}
_BOM = "\ufeff\u200b\u200c\u200d\u2060"


def path_of(root) -> pathlib.Path:
    return pathlib.Path(root) / NAME


def _clean(line: str) -> str:
    s = (line or "").strip().lstrip(_BOM).strip()
    for mark in ("- ", "* ", "> ", "「", "」", "\"", "“", "”"):
        if s.startswith(mark):
            s = s[len(mark):].strip()
    return s


def pick(lines, k: int = KEEP) -> list[str]:
    """从一堆行里挑出能当"语感样本"的句子：最近 k 条，去重，过滤噪音。

    故意不做"像不像人"的判断 —— 那是她抄的时候该做的，不是我们代劳的。
    这里只挡掉明显不是句子的东西：链接、代码块、超长、超短。
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        s = _clean(raw)
        if not s or s.startswith("#") or s in _TEMPLATE:
            continue
        if any(n in s for n in _NOISE):
            continue
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[-k:] if k else out


def read(root, k: int = KEEP) -> list[str]:
    """读她抄的那份；没有这个文件就是她还没抄，不是错误。"""
    try:
        text = path_of(root).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return pick(text.splitlines(), k=k)


def block(root, k: int = KEEP) -> str:
    """注入用的那一段；空字符串表示这一轮不放。"""
    lines = read(root, k=k)
    if not lines:
        return ""
    return ("【你抄下来的句子（你自己觉得像人话的那种；不是要你学谁，是让你知道人是怎么说的）】\n"
            + "\n".join(f"- {s}" for s in lines))
