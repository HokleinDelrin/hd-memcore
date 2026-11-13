"""LLM 客户端：DeepSeek OpenAI 兼容接口，只用标准库 urllib。"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict[str, str]], *, model: str | None = None,
         temperature: float = 1.0, max_tokens: int = 800,
         json_mode: bool = False, timeout: int = 90,
         base: str | None = None, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LLMError("缺少 DEEPSEEK_API_KEY 环境变量")
    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        (base or DEFAULT_BASE).rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read()[:400].decode('utf-8', 'replace')}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"网络错误: {e.reason}") from e
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as e:
        raise LLMError(f"响应结构异常: {str(data)[:400]}") from e


def available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))
