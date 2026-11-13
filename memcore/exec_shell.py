"""她「申请执行一条命令」的那一层（L4，2026-11-04）。

他拍板的权限分级：L1 只读 / L2 她家写 / L3 白名单写（未做） /
**L4 执行命令 = 她申请、他回「准 <编号>」才跑** / L5 截屏读屏（只读，已做）。

**这一层先只放开只读命令。** 理由不是"读没风险"，而是"读错了还能回头"：
一次误写就是桌面上的真实后果 —— 覆盖掉他没保存的稿子、删掉一个目录、
把某个服务停掉。而她真正缺的那部分能力（看一眼磁盘上到底有什么、机器什么状态）
在"读"这一层已经拿得到。所以批准之后也只跑白名单里的只读动词，
其余在**申请那一刻**就拒掉，不让他批一个注定失败的请求。

三件事：
  1. `check()`  —— 只读白名单 + 元字符/写操作/危险路径的硬拒（纯函数，可单测）
  2. `Requests` —— 待批单：编号自增、过期作废、批准后落结果文件
  3. `reply_of()` —— 认出他那句「准 3」

**没做的事**（别误会）：不加 shell 到她的常驻工具集、不做通配、不解析重定向。
`subprocess` 只跑 `powershell -NoProfile -NonInteractive -Command <cmd>`，
命令串在进来时已经被字符集和动词两道筛过。
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time

MAX_CMD = 200
MAX_OUT = 4000
TTL_S = 30 * 60          # 半小时没批就作废：他要是一天后才回「准 3」，那件事早不是那件事了
KEEP = 60                # 状态文件里最多留 60 条，够回头看，不至于长成日志
OUTDIR = "执行"

# 只读动词白名单。收的是"看了不会改任何东西"的那些：
# 文件与目录、系统信息、进程与服务、网络状态、磁盘与卷。
# 别名一并收（dir/ls/cat/type 都是 PowerShell 别名），否则她写习惯了会被误拒。
READONLY = {
    # 文件 / 目录（只读）
    "get-childitem", "gci", "dir", "ls",
    "get-item", "gi",
    "get-content", "gc", "cat", "type",
    "test-path", "resolve-path",
    "get-location", "pwd", "gl",
    "select-string", "sls",
    "measure-object", "measure",
    "compare-object", "diff", "fc",
    "get-filehash", "fciv",
    # 系统信息
    "get-date", "whoami", "hostname", "systeminfo", "ver",
    "get-process", "gps", "get-service", "gsv",
    "get-volume", "get-psdrive", "get-command", "gcm",
    "get-computerinfo", "get-hotfix",
    "get-ciminstance", "gcim", "get-wmiobject", "gwmi",
    "get-netadapter", "get-nettcpconnection", "get-netipaddress",
    "get-dnsclientserveraddress", "get-netroute",
    "where.exe", "where", "echo", "write-output",
    # 网络（只读）
    "netstat", "ipconfig", "nslookup", "ping", "tracert", "arp", "route", "getmac",
}

# 写/改/执行类动词：出现在命令任何位置都拒。
# 只收**带连字符的 cmdlet**和少数缩写 —— 这套检查是第二道筛，第一道是首词白名单。
# 别往里加 `md ` / `del ` / `py ` / `git` 这种短词：他盘里的路径本身就带这些
# （`Softwares&Tools`、`persona.md`、`git-repo`），一刀切会把正常申请全拒掉。
# 而它们出现在第二位置时并不会被执行 —— 没有管道、没有分号，PowerShell 只把
# 后面那串当参数。
WRITE_WORDS = (
    "set-", "remove-", "new-", "add-", "clear-", "stop-", "start-",
    "restart-", "resume-", "suspend-", "invoke-", "iex", "export-", "import-",
    "copy-", "move-", "rename-", "format-",
    "out-file", "out-host", "tee-object",
    "set-content", "add-content", "remove-item", "remove-variable",
)

# 元字符：管道、重定向、串联、变量、表达式、脚本块。允许成对引号（路径里有空格）。
# `&` 不在里面：他的路径里就带着它（`Softwares&Tools`），一刀拒掉等于半个 C 盘看不了。
# 单独成词的 `&` 另外查（PowerShell 里那是调用运算符）。
BAD_CHARS = set(";|<>`$%^(){}[]!\n\r\t")
STANDALONE_AMP = re.compile(r"(?:^|\s)&{1,2}(?:\s|$)")

PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*")
APPROVE_RE = re.compile(
    r"^\s*(?:准|批准|同意|可以)\s*(?:#|第)?\s*(\d{1,6})\s*(?:号)?\s*[。.!！]?\s*$")
REJECT_RE = re.compile(
    r"^\s*(?:不准|驳回|不行|否)\s*(?:#|第)?\s*(\d{1,6})\s*(?:号)?\s*[。.!！]?\s*$")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def check(cmd: str, is_blocked=None) -> tuple[bool, str]:
    """这条命令能不能进待批队列。**申请那一刻就判**，不等他批。

    Args:
        cmd: 她想跑的整条命令（不含 shell 包装）。
        is_blocked: 可选 ``str -> str|None``，给一个路径返回拒绝理由。
            插件传的是 safe_fs 的黑名单判定，防止她借 `Get-Content` 读自己的源码。

    Returns:
        (True, "") 表示可申请；否则 (False, 原因)，原因会原样回给她。
    """
    text = (cmd or "").strip()
    if not text:
        return False, "要跑什么？命令是空的。"
    if len(text) > MAX_CMD:
        return False, f"太长了（{len(text)} 字），一次最多 {MAX_CMD} 字。"
    bad = sorted({c for c in text if c in BAD_CHARS})
    if bad:
        shown = " ".join({";": "；", "|": "|", "<": "<", ">": ">"}.get(c, c) for c in bad)
        return False, (f"里面有 {shown} 这类符号。这一层只跑一条命令，不拼管道、不重定向、"
                       "不取变量。想要什么就直接说那一条。")
    if STANDALONE_AMP.search(text):
        return False, "里面有单独的 & —— 那是把两条命令接在一起的写法，这一层不跑。"

    head = text.split()[0].lower().strip("\"'")
    if head not in READONLY:
        return False, (f"「{head}」不在只读清单里。现在只放开只读命令（看目录、看文件、看进程、"
                       "看网络、看磁盘）；要改东西得等下一级权限。")

    low = " " + text.lower() + " "
    for w in WRITE_WORDS:
        if w in low:
            return False, (f"里面有「{w.strip()}」—— 那一类会改东西，这一层不放开。"
                           "只读的申请我照转，他会批。")

    if is_blocked is not None:
        for raw in PATH_RE.findall(text):
            why = is_blocked(raw.rstrip("\\/"))
            if why:
                return False, f"{raw} 不能看：{why}"

    return True, ""


def reply_of(text: str) -> tuple[str, int] | None:
    """认出他那句回话。返回 ("approve"|"reject", 编号) 或 None。

    Args:
        text: 他的原话，如「准 3」「准#3」「不准 3」。

    Returns:
        None 表示这不是一条审批回话（照常进对话）。
    """
    t = (text or "").strip()
    for kind, rx in (("approve", APPROVE_RE), ("reject", REJECT_RE)):
        m = rx.match(t)
        if m:
            return kind, int(m.group(1))
    return None


def run(cmd: str, *, timeout: int = 15, cwd=None) -> dict:
    """真跑一条已经被批准的命令。

    Args:
        cmd: 已经过 `check()` 的命令串。
        timeout: 秒。超时直接掐，不让他等一个卡住的 `dir /s`。
        cwd: 工作目录（默认给她的家目录）。

    Returns:
        {"ok": bool, "out": str, "err": str, "ms": int}
    """
    started = time.time()
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-Command", cmd],
            capture_output=True, timeout=max(1, int(timeout)),
            cwd=str(cwd) if cwd else None,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "out": "", "ms": int((time.time() - started) * 1000),
                "err": f"超过 {timeout} 秒还没跑完，掐了。"}
    except OSError as e:
        return {"ok": False, "out": "", "err": f"起不来：{e}",
                "ms": int((time.time() - started) * 1000)}
    out = _decode(p.stdout)[:MAX_OUT]
    err = _decode(p.stderr)[:MAX_OUT]
    return {"ok": p.returncode == 0, "out": out, "err": err,
            "ms": int((time.time() - started) * 1000)}


def _decode(raw: bytes) -> str:
    """Windows 控制台输出可能不是 UTF-8（中文系统常见 GBK）。"""
    if not raw:
        return ""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


class Requests:
    """待批单。一个 JSON 文件，编号自增；批过的落一份结果到她家目录。"""

    def __init__(self, state_path, out_dir, *, ttl: int = TTL_S,
                 keep: int = KEEP, timeout: int = 15, cwd=None):
        self.state_path = pathlib.Path(state_path)
        self.out_dir = pathlib.Path(out_dir)
        self.ttl = int(ttl)
        self.keep = int(keep)
        self.timeout = int(timeout)
        self.cwd = cwd

    # ---------------------------------------------------------------- 存取
    def _load(self) -> dict:
        try:
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        if not isinstance(st, dict):
            st = {}
        st.setdefault("next", 1)
        items = st.get("items")
        st["items"] = items if isinstance(items, list) else []
        return st

    def _save(self, st: dict) -> None:
        st["items"] = st["items"][-self.keep:]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    @staticmethod
    def _find(st: dict, n: int) -> dict | None:
        for it in st["items"]:
            if int(it.get("n", 0)) == int(n):
                return it
        return None

    def _expire(self, st: dict, now: float) -> None:
        for it in st["items"]:
            if it.get("status") == "pending" and now - float(it.get("at", 0)) > self.ttl:
                it["status"] = "expired"
                it["why_out"] = "过了半小时没批，作废了"

    # ---------------------------------------------------------------- 动作
    def request(self, cmd: str, why: str = "", *, is_blocked=None,
                now: float | None = None) -> dict:
        """她申请一条。校验不过就不落单（别让他批一个注定失败的）。"""
        ok, reason = check(cmd, is_blocked)
        if not ok:
            return {"ok": False, "error": reason}
        now = time.time() if now is None else float(now)
        st = self._load()
        self._expire(st, now)
        n = int(st["next"])
        st["next"] = n + 1
        st["items"].append({
            "n": n,
            "cmd": cmd.strip(),
            "why": (why or "").strip()[:300],
            "at": now,
            "status": "pending",
            "out": "",
            "ms": 0,
            "doc": "",
        })
        self._save(st)
        return {"ok": True, "n": n}

    def pending(self, *, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else float(now)
        st = self._load()
        self._expire(st, now)
        rows = [it for it in st["items"] if it.get("status") == "pending"]
        self._save(st)
        return rows

    def reject(self, n: int, *, why_out: str = "他说不准", now: float | None = None) -> dict:
        now = time.time() if now is None else float(now)
        st = self._load()
        self._expire(st, now)
        it = self._find(st, n)
        if it is None:
            return {"ok": False, "error": f"没有 #{n} 这个编号。"}
        if it.get("status") != "pending":
            return {"ok": False, "error": f"#{n} 已经不是待批状态（{it.get('status')}）。"}
        it["status"] = "rejected"
        it["why_out"] = why_out
        self._save(st)
        return {"ok": True, "n": n, "cmd": it.get("cmd", "")}

    def approve(self, n: int, *, now: float | None = None) -> dict:
        """他回「准 N」之后：跑掉、落结果文件、返回给他看的摘要。"""
        now = time.time() if now is None else float(now)
        st = self._load()
        self._expire(st, now)
        it = self._find(st, n)
        if it is None:
            return {"ok": False, "error": f"没有 #{n} 这个编号。"}
        status = it.get("status")
        if status == "expired":
            return {"ok": False, "error": f"#{n} 已经作废了（等了超过 {self.ttl // 60} 分钟）。"}
        if status != "pending":
            return {"ok": False, "error": f"#{n} 已经处理过（{status}）。"}
        cmd = str(it.get("cmd") or "")
        # 批的时候再验一次：状态文件在她的家目录旁边，理论上她动不了，
        # 但"批准的一条命令"值得再确认它是只读的 —— 这一层没有第二次机会。
        ok, reason = check(cmd)
        if not ok:
            it["status"] = "blocked"
            it["why_out"] = reason
            self._save(st)
            return {"ok": False, "error": f"#{n} 跑不了：{reason}", "n": n, "cmd": cmd}

        res = run(cmd, timeout=self.timeout, cwd=self.cwd)
        it["status"] = "done" if res["ok"] else "failed"
        it["out"] = (res["out"] or res["err"])[:MAX_OUT]
        it["ms"] = res["ms"]
        doc = ""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            doc_p = self.out_dir / f"{n}-{time.strftime('%m%d-%H%M')}.txt"
            doc_p.write_text(
                f"#{n} {time.strftime('%Y-%m-%d %H:%M')}\n"
                f"他说：准\n命令：{cmd}\n"
                f"耗时：{res['ms']} ms\n\n"
                f"--- 输出 ---\n{res['out'] or '（空）'}\n"
                + (f"\n--- 错误 ---\n{res['err']}\n" if res["err"] else ""),
                encoding="utf-8")
            doc = f"{OUTDIR}/{doc_p.name}"
        except OSError:
            doc = ""
        it["doc"] = doc
        self._save(st)
        # 她做过什么，也留一行给她自己（下一轮注入里能读到）。失败不影响结果。
        try:
            from . import activity
            if self.cwd:
                first = " ".join((res["out"] or res["err"] or "").split())[:40]
                activity.note(self.cwd, "跑", cmd, f"{res['ms']} ms {first}".strip())
        except Exception:            # noqa: BLE001
            pass
        return {"ok": True, "n": n, "cmd": cmd, "out": res["out"],
                "err": res["err"], "ms": res["ms"], "doc": doc,
                "status": it["status"]}

    # ---------------------------------------------------------------- 给注入
    def recent(self, k: int = 3, *, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else float(now)
        st = self._load()
        self._expire(st, now)
        self._save(st)
        rows = [it for it in st["items"] if it.get("status") != "pending"]
        return rows[-int(k):]

    def block(self, k: int = 3, *, now: float | None = None) -> str:
        """给注入用的一段：她申请过什么、批没批、结果是什么。

        只在真有记录时才返回非空 —— 空段会让她以为"这一块本来该有东西"。
        """
        now = time.time() if now is None else float(now)
        st = self._load()
        self._expire(st, now)
        self._save(st)
        pend = [it for it in st["items"] if it.get("status") == "pending"][-3:]
        done = [it for it in st["items"]
                if it.get("status") in ("done", "failed", "rejected", "blocked")][-int(k):]
        if not pend and not done:
            return ""
        lines: list[str] = []
        for it in pend:
            mins = max(0, int((now - float(it.get("at", now))) // 60))
            lines.append(f"- #{it['n']} 还在等他准（申请于 {mins} 分钟前）：`{it.get('cmd','')}`")
        for it in done:
            stt = it.get("status")
            if stt == "rejected":
                lines.append(f"- #{it['n']} 他没准：`{it.get('cmd','')}`")
            elif stt == "blocked":
                lines.append(f"- #{it['n']} 批了但跑不了：{it.get('why_out','')}")
            else:
                body = " ".join(str(it.get("out") or "").split())[:160]
                tail = f"\n  {body}" if body else "（没有输出）"
                keep = f"（全文在你自己的 {it['doc']}）" if it.get("doc") else ""
                lines.append(f"- #{it['n']} {'跑完了' if stt == 'done' else '跑了但报错'}："
                             f"`{it.get('cmd','')}`{tail}{keep}")
        return "【你申请跑过的命令】\n" + "\n".join(lines)
