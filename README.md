# hd-memcore

**The context window is a workbench, not a filing cabinet.**
`hd-memcore` is a dependency-free context re-assembly layer: raw turns are stored forever, memories are derived (and recomputable), and the context for every turn is assembled from scratch under a budget.

## Try it

```bash
python -m memcore.cli demo
```

No API key. Nothing to install. Pure standard library, Python 3.10+.

The demo replays 200 turns, then asks back six details planted early in that conversation. A rolling-window approach keeps 0/6 of them; this keeps 6/6. It then prints one turn's assembly — the 14 context segments with their token counts and the recall scores behind each hit — and exports a human-readable `snapshot.md`.

## What it is for

Long conversations lose their beginning: a rolling window forgets turn 3, and a "summarize and migrate" bundle goes stale and cannot be re-derived. So keep the raw turns and re-assemble the working context every turn.

## Comparison

| | rolling window | summarize & migrate | **hd-memcore** |
|---|---|---|---|
| Early details | gone once out of window | lost at summary time | **raw text kept, recallable anytime** |
| Memories recomputable | — | ❌ summaries can't be recomputed | ✅ derived layer can be rebuilt |
| Assembly visible | ❌ | ❌ | ✅ per-turn token breakdown + recall scores |
| Dependencies | — | vendor SDKs | **stdlib only** |

The track is crowded, so, honestly: mem0 / Zep / Letta cover far more ground than this does. hd-memcore only does the re-assembly layer — and tries to do that one layer visibly.

## Known limits

- Text memory only. No vector store, no multimodal, no distributed setup.
- SQLite, one file, one machine: fine for a single process, not for a fleet.
- Recall scoring is lexical bigrams + TF-IDF cosine, not embeddings. Real vectors slot in where `retrieve()` calls `recall._vec` / `recall._cos`.
- Distillation is heuristic by default; an LLM is optional, and every derived layer can be recomputed from the raw turns.

## API

```python
from memcore import Store, assemble

store = Store("mem.db")                                   # one SQLite file
store.add_msg("local:private:0", "user", "我养了只猫，叫豆豆")
asm = assemble(store, "local:private:0", "我家猫今天又把耳机线咬了")
print(asm.messages)            # hand straight to any chat API
print(asm.report["total"])     # tokens spent on this turn
```

For the whole loop (store → assemble → generate → store) use `Bot`. With no model credentials it answers offline by describing what it assembled, so nothing leaves the machine.

## Install & test

```bash
pip install -e .                            # optional — the demo runs without installing
python -m unittest discover -s tests
```

On a GBK console (Windows) set `$env:PYTHONIOENCODING='utf-8'` first.

## Layout

| file | what |
|---|---|
| `memcore/store.py` | L0 raw turns (append-only) + derived tables |
| `memcore/distill.py` | L1/L2/L3 derivation from the raw turns |
| `memcore/recall.py` | hybrid scoring: bigrams + TF-IDF + recency + salience, MMR dedupe |
| `memcore/assemble.py` | per-turn budgeted assembly, 14 segments |
| `memcore/cli.py` | `demo` / `chat` / `probe` / `export` / `stats` |

---

## 中文

**上下文窗口是工作台，不是档案柜。**
`hd-memcore` 是一个零依赖的上下文重装配层：原文永久存、记忆做派生（可重算）、每一轮按预算重新装配。

### 一条命令

```bash
python -m memcore.cli demo
```

不需要任何 API key，不需要装任何东西，纯标准库。

demo 会灌入 200 轮对话，然后回问早期埋下的 6 个细节：滚动窗口方案保住 0/6，这里 6/6。接着打印单轮装配明细（14 段各占多少 token、召回分各是多少），最后导出一份 `snapshot.md`。

### 它解决什么

长对话里滚动窗口会忘掉第 3 轮，打包迁移的摘要会过期而且没法重算；这个方案是「原文永久存 + 按需重算装配」。

### 对比

| | 滚动窗口 | 打包迁移摘要 | **hd-memcore** |
|---|---|---|---|
| 早期细节 | 超窗即丢 | 摘要时丢 | **原文在，随时可召回** |
| 记忆可重算 | — | ❌ 摘要无法重算 | ✅ 派生层可重建 |
| 装配可见 | ❌ | ❌ | ✅ 每轮输出 token 分解与召回分 |
| 依赖 | — | 各家 SDK | **纯 stdlib** |

mem0 / Zep / Letta 覆盖的场景比它广，它只做「重装配」这一层。

### 已知边界

只做文本记忆：没有向量库、没有多模态、没有分布式；SQLite 单文件，适合单机。召回打分为词面 bigram + TF-IDF 余弦，没有 embedding（要换真向量，替掉 `retrieve()` 里的 `recall._vec` / `recall._cos` 即可）。蒸馏默认走启发式，模型可选，派生层随时能从原文重算。

### 最小 API

```python
from memcore import Store, assemble

store = Store("mem.db")
store.add_msg("local:private:0", "user", "我养了只猫，叫豆豆")
asm = assemble(store, "local:private:0", "我家猫今天又把耳机线咬了")
print(asm.messages)          # 直接喂给任何聊天 API
```

### 跑测试

```bash
pip install -e .                            # 可选，demo 不装也能跑
python -m unittest discover -s tests
```

Windows GBK 控制台先设 `$env:PYTHONIOENCODING='utf-8'`。
