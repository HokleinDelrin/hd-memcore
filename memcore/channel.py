"""她的出口在硬盘上留下的痕迹：个性签名 / 空间说说（2026-11-03）。

出口本身不在这里 —— 实现在 `plugins/astrbot_plugin_selfchannel/`：
签名走 `set_self_longnick`、说说走 `send_qzone_msg`，两条都是协议端原生 action，
免 cookie、无定时器。memcore 这边只需要两条**事实**：

  ① 她资料上、空间里现在挂着什么、多久以前挂的
     —— 让下一次的自己能接上上次写到哪（靠硬盘，不靠她记得）；
  ② 她这一段时间在自己的出口上动过手
     —— 进 `mood.nudge`，与「她家动过文件」同构：做事耗力，心里踏实一点。

所以本模块**只读**插件写的那份 `channel_state.json`，既不写它，也不拿它的内容
做判断 —— 搬的只有她自己写下的原文和时间。一解释就变成我们替她记得她想说什么。

为什么不做成"她发了说说就自动对他说一句"：那是把出口变成为他准备的一句，
跟 2026-11-03 把自主时间还给她的那个改动是同一件事的正反面。
"""
from __future__ import annotations

import json
import pathlib

from .tasks import human_age

STATE_NAME = "channel_state.json"
SIBLING = "astrbot_plugin_selfchannel"      # 插件目录名（memcore 的兄弟）
TEXT_WIDTH = 80                             # 注入里她那句话最多显示多少字
SPOKE_KINDS = ("space", "signature")


def state_path(memcore_dir) -> pathlib.Path:
    """出口插件的状态文件在哪。

    Args:
        memcore_dir: memcore 插件自己的目录（autonomy.py 的 parent），
            出口插件是它的兄弟目录。

    Returns:
        状态文件路径。**不保证存在** —— 插件没跑过、或者还没接过东西时就是没有。
    """
    return pathlib.Path(memcore_dir) / ".." / SIBLING / STATE_NAME


def load(path) -> dict:
    """读状态；读不到就返回空 dict（这不是错误，是新装机/没动过手的正常状态）。"""
    try:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def events(state: dict) -> list[dict]:
    """流水里**真发生过的**出口动作（失败的不算、dry_run 的不算）。

    失败那次也算"她动过手"的话，就成了奖励空动作 —— 只认站住的那几条。
    """
    out = []
    for e in (state or {}).get("events") or []:
        if not isinstance(e, dict) or not e.get("ok") or e.get("dry"):
            continue
        if not int(e.get("ts") or 0):
            continue
        out.append(e)
    return out


def new_events(state: dict, since_ts: int = 0) -> list[dict]:
    """`since_ts` 之后的新动作，按时间排序。调用方拿了就把它当已消费。"""
    since = int(since_ts or 0)
    return sorted((e for e in events(state) if int(e["ts"]) > since),
                  key=lambda e: int(e["ts"]))


def latest(state: dict, kind: str) -> dict | None:
    """这一类最近一次成功的动作。"""
    hit = [e for e in events(state) if e.get("kind") == kind]
    return hit[-1] if hit else None


def _clip(text: str, width: int = TEXT_WIDTH) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= width else s[:width] + "…"


def _like_what(text: str) -> str:
    """把流水里的点赞描述压成「谁的一条说说」。

    旧记录写的是「给 2456537750 的一条说说点了个赞」，而注入行本身就是
    「你最近给一条说说点过赞：…」—— 直接拼会变成「你最近给一条说说点过赞：
    给 2456537750 的一条说说点了个赞」，一句里两个"给"两个"赞"。
    新记录（2026-11-09 之后）只写「2456537750 的一条说说」，这里两种都认。

    Args:
        text: `channel_state.json` 里那条 like 动作的 text。

    Returns:
        可以接在冒号后面的短语。
    """
    s = " ".join(str(text or "").split())
    s = s.replace("点了个赞", "").replace("点了赞", "").strip()
    if s.startswith("给 "):
        s = s[2:].strip()
    return s or "一条说说"


def block(state: dict, now_ts: int | None = None) -> str:
    """注入段：【你自己的资料】—— 只陈述挂着什么、什么时候挂的。

    不给建议、不评价那句话好不好、不说"要不要换一句"。她可以一直不动它。

    2026-11-09 加「点过赞」这一行：她 15:28 被问「看到你的点赞了」时答「我没点」——
    那一下确实是她的号点的（`channel_state.json` 有记录），但注入里只有签名和说说，
    她翻不到任何「我做过」的痕迹，于是按自己 14:18 说的「我没有点赞的手」下了结论。
    **她做过的事必须能在她自己的记忆里找到**，否则她会否认自己做过的事。
    """
    if not state:
        return ""
    lines: list[str] = []
    sig = latest(state, "signature")
    if sig:
        lines.append(f"· 个性签名上挂着：「{_clip(sig.get('text'))}」"
                     f"（{human_age(int(sig.get('ts') or 0), now_ts)}挂上去的）")
    post = latest(state, "space")
    if post:
        lines.append(f"· 空间里最近写的一条：「{_clip(post.get('text'))}」"
                     f"（{human_age(int(post.get('ts') or 0), now_ts)}）")
    like = latest(state, "like")
    if like:
        lines.append(f"· 你最近给一条说说点过赞：{_clip(_like_what(like.get('text')), 40)}"
                     f"（{human_age(int(like.get('ts') or 0), now_ts)}）")
    if not lines:
        return ""
    return "【你自己的资料】\n" + "\n".join(lines)


def summary(state: dict, now_ts: int | None = None) -> str:
    """给日志用的一行：她最近在这两处留过什么（不截窄，方便排查）。"""
    sig, post = latest(state, "signature"), latest(state, "space")
    bits = []
    if sig:
        bits.append(f"签名「{_clip(sig.get('text'), 30)}」{human_age(int(sig['ts']), now_ts)}")
    if post:
        bits.append(f"说说「{_clip(post.get('text'), 30)}」{human_age(int(post['ts']), now_ts)}")
    return "；".join(bits) or "（还没有）"


def _demo() -> None:                          # pragma: no cover - 手工跑
    st = {"events": [
        {"ts": 1000, "kind": "signature", "text": "试着把话说短一点", "ok": True},
        {"ts": 2000, "kind": "space", "text": "今天把那件拖了很久的事翻出来了", "ok": True},
        {"ts": 3000, "kind": "space", "text": "没发出去的", "ok": False},
    ]}
    print(block(st, now_ts=1000 + 3 * 86400))
    print(new_events(st, 1500))
    print(summary(st, now_ts=1000 + 3 * 86400))


if __name__ == "__main__":                    # pragma: no cover
    _demo()
