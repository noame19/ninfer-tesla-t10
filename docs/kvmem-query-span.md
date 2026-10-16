# KVMem: the retrieval query span

## The defect

The raw-prompt endpoint never retrieved anything useful. With the user's own flags
(`--max-context 262144 --kvmem-budget 53248 --kvmem-gen-reserve 30720 --kvmem-host-mib 5120
--kvmem-sessions 1`), a 226,886-token prompt that carried a probe in its first fifth answered
`BEGIN=MISS MID=HIT END=HIT` — only the probe that happened to land inside the retained tail.

The census line for that run was the whole story:

```
KVMEM_RETRIEVAL logical=45 query_tokens=185772 cold=0 selected=0
```

`selected=0`: the selector picked nothing, so the window degraded to the mandatory sink pages plus
the newest pages.

## Root cause

`src/models/qwen3_5/frontend/frontend.cpp:1395` (was `:1347`) marks the query for a raw token
prompt as the **entire prompt**:

```cpp
result.memory_query = TokenSpan{0, result.token_ids.size()};
```

Retrieval ranks pages by the mean key over the query span. A mean taken over a whole document
carries no signal, so nothing scores above the mandatory pages. The structured path does not have
this problem: `frontend/chat_template.cpp:438` sets `result.memory_query = content_regions[i - 1]`
for the last real user message, which is why `/v1/chat/completions` retrieved correctly
(`logical=225718 query_tokens=50 cold=610` → 3/3) while `/completion` did not.

Both lines are upstream's, not this port's: they arrive in `ninfer-kvmem.patch`
(hunks `@@ -1188,6 +1201,8 @@` and `@@ -1328,6 +1343,7 @@`), and upstream master has no commit that
changes them. But upstream's llama.cpp backend bounds the span from the driver side —
`--kvmem-query-max-tokens` (512 tokens from the tail of the span) and `--kvmem-query-last`
(64-token fallback) — and the NInfer backend ported neither, as upstream's own
`docs/multi-backend-differences.md` states ("没有对应的可配置长度上限").

## The fix

`apply_kvmem_query_policy(PreparedPromptData&, cap, fallback)` in
`src/models/qwen3_5/frontend/frontend.cpp`, called at the end of both `Frontend::prepare` and
`Frontend::prepare_tokens` (so a graft has already shifted the token indices), implements the two
upstream limits as **separate** paths:

| span | limit applied |
|---|---|
| covers the whole prompt (raw-token fallback) | `--kvmem-query-last`, default **64** |
| structured span, or a clamp of one | `--kvmem-query-max-tokens`, default **512** |
| absent / empty | `--kvmem-query-last` |
| `cap == 0` | policy disabled |

The split matters and was measured. Using the 512 cap on the raw case leaves 512 tokens of trailing
prose around an ~80-token question, and the mean key stays diluted:

| query window | recall (small rig, probes evicted) |
|---|---|
| 0 — whole prompt, the pre-fix behaviour | **0/3** |
| 48 | 3/3 |
| **64 — the default** | **3/3** |
| 96 | 3/3 |
| 128 | 2/3 |
| 256 | 2/3 |
| 512 | 1/3 |

New flags: `--kvmem-query-max-tokens N` (0 disables) and `--kvmem-query-last N`.

## A second, older bug fixed alongside

Neither `prepend_graft` nor `apply_direct_graft` shifted `memory_query` while shifting
`token_ids`, `positions`, `vision_items` and the rewrite frontiers. A graft inserts `n` tokens in
front of the prompt, so the span addressed the wrong tokens — for a whole-prompt query it silently
stopped covering the prompt's tail. Both now shift `memory_query->begin` like every other index.

## Why the PPL gate does not apply here

Retrieval changes *which* KV pages are resident, so the output legitimately differs from a run with
different retrieval. The gate for this change is therefore recall, not a PPL number:
the small rig moved 0/3 → 3/3 with `/v1/chat/completions` unaffected, and the 242k scenario is the
acceptance test.

## Upstream divergence

This is a deliberate local divergence and is recorded here (and in the commit message) as such.
The alternative — changing selection semantics so a whole-prompt span means "average everything" —
would alter ranking for the structured path too, so the clamp was preferred.
