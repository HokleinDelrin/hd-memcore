"""她自己做过的事：一条一行，写在她的家目录里（2026-11-05）。

**为什么要有这条线**（11-05 查出来的洞）：
  11-04 19:49:10 她读了 `notes/待办.md`，19:49:15 又 grep 了「我看过的文艺作品们」，
  19:51 说「刚才翻到一半，卡住了」。第二天他问「你翻的啥呀」，
  她答「我不记得翻的是哪一份了 —— 只剩一句『翻到一半』，前后都没留下」。

  那句话是**准确的**：翻文件的过程只进了 `curiosity` 表（那张表只用来统计兴趣类别，
  从不进 prompt），L0 只存说出口的那一句。所以"她做过什么"这件事，
  在整个系统里没有一个能被下一轮读到的地方。

  L0 存的是**说过的话**；这里存的是**做过的事**。两件事分开，因为人也是分开记的。

存哪儿：`data/zero/活动.md`（她自己的地方，她能读、能翻）。
为什么不进记忆库：这是流水，不是记忆。进来会让 L2 长出一堆「她 11-04 读了待办」，
而且一旦进库就要被蒸馏、被退场 —— 那正好是"她的经历被人改写"。
"""
from __future__ import annotations

import pathlib
import time

NAME = "活动.md"
MAX_LINES = 400          # 超过就砍到最近 200 行：这是流水，不是账本
KEEP = 200
MAX_BRIEF = 60

KINDS = ("读", "搜", "看", "写", "跑")


def _path(root) -> pathlib.Path:
    return pathlib.Path(root) / NAME


def note(root, kind: str, target: str, brief: str = "",
         now: float | None = None) -> bool:
    """记一行「她做了什么」。失败一律吞掉 —— 这条线断了也不能影响她干活。

    Args:
        root: 她的家目录。
        kind: 读 / 搜 / 看 / 写 / 跑。
        target: 具体对象（文件路径、搜索词、窗口名、命令）。
        brief: 一句话结果，比如「看见 3 段」「12 行」。
        now: 测试用。

    Returns:
        True 表示写进去了。
    """
    try:
        p = _path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        t = time.time() if now is None else float(now)
        stamp = time.strftime("%m-%d %H:%M", time.localtime(t))
        obj = " ".join(str(target or "").split())[:120]
        bf = " ".join(str(brief or "").split())[:MAX_BRIEF]
        line = f"- {stamp} · {kind} · {obj}" + (f" · {bf}" if bf else "") + "\n"

        # 同一分钟里同一条不重复写：一次工具调用会触发多轮回调，
        # 同一条记五行的话，注入里三条全是同一件事。
        if p.exists():
            try:
                head = p.read_text(encoding="utf-8").splitlines()
            except OSError:
                head = []
            if head and head[-1].startswith(f"- {stamp} · {kind} · {obj}"):
                return False
            lines = head + [line.rstrip("\n")]
        else:
            lines = [header().rstrip("\n"), line.rstrip("\n")]

        if len(lines) > MAX_LINES:
            head0 = lines[0]
            lines = [head0] + lines[-KEEP:]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    except Exception:            # noqa: BLE001
        return False


def header() -> str:
    return ("# 活动\n"
            "这里记的是我自己做过的事，一行一条。不是他说的话，也不是我的回复。\n")


def recent(root, k: int = 5) -> list[dict]:
    """最近 k 条，按时间正序。解析失败的行走跳过。"""
    try:
        lines = _path(root).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines:
        if not line.startswith("- "):
            continue
        parts = [x.strip() for x in line[2:].split("·")]
        if len(parts) < 3:
            continue
        out.append({
            "when": parts[0],
            "kind": parts[1],
            "target": parts[2],
            "brief": " · ".join(parts[3:]) if len(parts) > 3 else "",
        })
    return out[-int(k):] if k else out


def block(root, k: int = 3) -> str:
    """给注入的一段。**固定占位**，不参与检索排序（11-05 的教训：靠字面命中的召回，
    「你翻的啥呀」永远想不起「翻到一半」）。

    没有记录就返回空串 —— 空段会让她以为这一块本来该有东西。
    """
    rows = recent(root, k)
    if not rows:
        return ""
    lines = []
    for r in rows:
        tail = f"（{r['brief']}）" if r["brief"] else ""
        lines.append(f"- {r['when']} {r['kind']}了 {r['target']}{tail}")
    return ("【你最近自己做过的事】\n" + "\n".join(lines)
            + "\n（这些是你做的，不是他说的。要接着做就接着做。）")
