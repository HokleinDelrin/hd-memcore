"""QQ 空间接口的限流退避（2026-11-09）。

**为什么要这个**：`get_qzone_feeds` / `get_qzone_msg_list` 走的是腾讯网页版那套
`feeds3_html_more`，繁忙时返回 `network busy`。现在撞上了只回一句人话，
她下一轮再问就再撞一次 —— 撞得越勤，被限得越死（2026-11-09 记在 HANDOVER 待办里）。

**分两层**：

1. **单次调用里短等一次**（`RETRY_WAIT` 秒）。繁忙多半是瞬时的，第二次常常就通了。
   等的时间不能长：这段是在她的工具调用里同步等的，等太久她那轮就被拖住了。
2. **连续撞上就锁一段时间**：`2 → 4 → 8 → 16 → 30` 分钟封顶。
   锁着的窗口里直接告诉她还看不到、几分钟后再来，**不再打接口**。
   一次成功就把计数清零。

**为什么退避状态要落文件**：AstrBot 一天重启过 23 次（AGENTS 里那条教训）。
"连续失败几次"记在进程内存里，一重启就归零 —— 线程里攒不住的量，等于没记。
所以这里只认文件：读一次、写一次，跨进程、跨重启都还在。

**只管读**：点赞（`like_qzone`）另有一套每日闸，且它是写动作，
真被限流的时候也不该由这里替它决定放不放。
"""
from __future__ import annotations

import json
import pathlib
import time

# 连续失败第 1/2/3/4/5+ 次分别锁多久（分钟）；再失败就按最后一档封顶
BACKOFF_MINUTES = (2, 4, 8, 16, 30)
# 单次调用内撞上繁忙之后等多久再试一次（秒）
RETRY_WAIT = 4.0
# 判定「这是限流，不是别的错」的关键词（各家措辞不一，抓形状就够）
BUSY_MARKS = ("network busy", "busy", "too many request", "rate limit",
              "使用人数过多", "系统繁忙", "繁忙", "操作过于频繁", "请求过于频繁",
              "稍后再试", "frequen")


def looks_busy(text: str) -> bool:
    """这段错误文本像不像腾讯侧的限流。

    Args:
        text: 协议端返回或异常里的原文。

    Returns:
        True 表示按限流处理（会重试 + 退避）。
    """
    s = str(text or "").lower()
    return any(m in s for m in BUSY_MARKS)


def human_left(seconds: int) -> str:
    """把"还要等多久"说成人话 —— 她只会照着转述，别出现秒数碎片。

    Args:
        seconds: 剩余秒数。

    Returns:
        「1 分钟」「几分钟」这类短语。
    """
    s = max(0, int(seconds))
    if s <= 90:
        return "一两分钟"
    m = (s + 59) // 60
    if m <= 10:
        return f"{m} 分钟"
    return "十几分钟"


def _now() -> int:
    return int(time.time())


class Gate:
    """空间读接口的退避闸。状态落在一个小 JSON 里（跨重启还在）。

    字段：
        fails  连续失败次数（一次成功就清零）
        until  这个时间戳之前不许再打接口
        last   最近一次撞上的时间
        note   最近一次的错误摘要（排查用）
    """

    def __init__(self, path: pathlib.Path | str) -> None:
        self.path = pathlib.Path(path)

    # ---------------------------------------------------------------- 读写
    def read(self) -> dict:
        """读状态；文件坏了、没有、被写成别的东西，都当成"没被限过"。"""
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"fails": 0, "until": 0, "last": 0, "note": ""}
        if not isinstance(d, dict):
            return {"fails": 0, "until": 0, "last": 0, "note": ""}
        return {
            "fails": max(0, int(d.get("fails") or 0)),
            "until": max(0, int(d.get("until") or 0)),
            "last": max(0, int(d.get("last") or 0)),
            "note": str(d.get("note") or "")[:200],
        }

    def _write(self, d: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        except OSError:
            pass                        # 落盘失败不该让她的工具调用炸掉

    # ---------------------------------------------------------------- 判定
    def wait_left(self, now: int | None = None) -> int:
        """还要锁多久（秒）。0 表示现在可以打接口。"""
        t = _now() if now is None else int(now)
        return max(0, self.read()["until"] - t)

    def hit(self, note: str = "", now: int | None = None) -> int:
        """记一次「撞上限流」，返回这次锁多少秒。"""
        t = _now() if now is None else int(now)
        d = self.read()
        d["fails"] = min(len(BACKOFF_MINUTES), d["fails"] + 1)
        minutes = BACKOFF_MINUTES[d["fails"] - 1]
        d["until"] = t + minutes * 60
        d["last"] = t
        d["note"] = str(note or "")[:200]
        self._write(d)
        return minutes * 60

    def ok(self) -> None:
        """一次成功：把连续失败清零（until 也一并清掉）。"""
        if not self.path.exists():
            return
        d = self.read()
        if not (d["fails"] or d["until"]):
            return
        self._write({"fails": 0, "until": 0, "last": d["last"], "note": ""})

    # ---------------------------------------------------------------- 给人话
    def refusal(self, now: int | None = None) -> str:
        """锁着的时候对她说的话。空字符串表示没锁。

        不写"限流""配额""接口"这类词 —— 她看不见屏幕，只会把原话转述出去。
        """
        left = self.wait_left(now)
        if left <= 0:
            return ""
        return f"空间这会儿不给我看，得等{human_left(left)}。等会儿再来。"


def _demo() -> None:                        # pragma: no cover - 手工跑
    import tempfile
    p = pathlib.Path(tempfile.gettempdir()) / "qzone_demo.json"
    p.unlink(missing_ok=True)
    g = Gate(p)
    for i in range(6):
        print("hit", i + 1, "->", g.hit("network busy"), g.refusal())
    g.ok()
    print("ok ->", g.read(), "refusal:", repr(g.refusal()))


if __name__ == "__main__":                  # pragma: no cover
    _demo()
