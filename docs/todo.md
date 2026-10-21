# The working queue for this line

This is the *execution* list. `docs/remaining-sm75-work.md` records the P-numbered tasks and their
history; this file records what is being done now, in what order, and how each item is judged. An
item leaves this file when it has merged into `sm75` (with its measurement) or when its branch has
been deleted after a recorded failure.

The merge rule is in `AGENTS.md` under *What may and may not reach `sm75`*: a change reaches the
main line when it is measured to be a real speedup, a real fix or a real capability with no
compensating cost, **and** the user has agreed to it. A branch that was only tested is not a merge
candidate.

## Branch grouping

Branches group items that share a *class* of change and one build, so a long compile is paid once.
Separability is the constraint on grouping: two items may share a branch only when their effects
land on **different models or different workloads**, so a regression is still attributable to one
of them by testing that model alone.

| Branch | Items | Why they may share it | Build cost |
|---|---|---|---|
| `sm75-p10-gdn-routes` | A1, A2 | Same route-key mechanism, different models: A1 is the 9B/4B (32 value heads), A2 is the 0.8B (16). No compile at all — both are `docs/device-profiles/t10-sm75.json`. | none |
| `sm75-p11-calibration` | A3 | One file pair (`route_catalog.cpp`, `device_calibration.cu`); the digest bump touches the same JSON A1/A2 edit, so it must land after them. | ~10 min |
| `sm75-p12-vision` | B1, B2, B3, B4 | All four are needed for the same acceptance ("this model encodes an image on the GPU"), and they separate by model: B1 is 9B only, B2/B3 are 4B and 0.8B only, B4 is the CPU fallback. Test 9B after B1, then 4B, then 0.8B. | ~10 min per step, incremental |
| `sm75-p13-router-kvmem` | E1, E2, E4 | One mechanism: the router's admission and option-merging rules. E1/E2 must land before E4 opens the gate. | ~5 min |
| `sm75-p14-kvmem` | D1, D2, D3 | Serve- and planning-level KVMem robustness; none of them changes numerics. | ~10 min |
| `sm75-p15-attn-h16` | C1 | Depends on A3 (the calibration instrument cannot name an `h16` key before it); long GPU sweep. | ~2 min + GPU |
| `sm75-p16-attn-race` | C3 | Independent kernel investigation; needs its own branch because it may touch the prompt kernels. | ~30-100 min if a kernel edit is needed |

## Status

**A1, A2, A3, B1-B4, D2, D3, E1, E2 and E4 are merged into `sm75` (A2 at `f72e8c6f`),** verified
there by rebuilding and re-measuring every model:

| Model | PPL after the merge | Was |
|---|---|---|
| Qwen3.8-27B | 3.00762 | 3.00762 (unchanged) |
| Ornith-1.5-9B | **3.96913** | 3.98352 (A1 improved it) |
| Qwen3.5-4B | **4.51396** | 4.52763 (A1 improved it) |
| Qwen3.5-0.8B | 6.35792 | 6.3419 (A2's cost, the operator accepted it) |

The three GDN keys are live and independent, proven by `NINFER_GDN_TWO_STAGE=0` on the merged build:
it returns the 0.8B to 6.3419 and the 9B to 3.98352, i.e. each tower consults its own key. Vision was
re-verified after the merge too -- the 9B encodes an image in 2.8 s on the GPU.

Nothing else is merged. A branch that was only tested is not a merge candidate:

| Item | Branch | State | Measured |
|---|---|---|---|
| C1 | `sm75-p15-attn-h16` | **kept unmerged** | the calibrated h16 table moves prefill −0.4 % / −0.2 %, decode +0.2 %: inside this card's spread |
| C3 | `sm75-p16-attn-race` | **kept unmerged** | not exposed: this tree has no `ggml_fattn.cu`, so the 512-wide tile it raced on does not exist here |
| D1 | (none — merged as documentation) | **no clamp to add** | a 147k-token prompt answered at five needle positions against a 51,200-token pool; 168k of history recalled in turn 3 |

## Items

### A1 — T10 profile: `gdn_two_stage/h32` (9B, 4B) · branch `sm75-p10-gdn-routes`

The T10 profile names only `gdn_two_stage/h48`. The three 9B towers and the 4B have
`linear_num_value_heads = 32`, so `two_stage_route()` at
`src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp:217` looks up
`gdn_two_stage/h32`, finds nothing, and returns false. The compiled profiles carry `h32` for the
3090/4090/5090 (`src/runtime/engine/device_profiles.json`) but those entries are keyed by
`hardware_class`, which this port never matches, so on the T10 the two-stage GDN prefill has been
off for every 9B and 4B.

Measured before the change (Ornith-1.5-9B, Q6_K, rk4v4-e8, `--spec mtp --draft-tokens 2`,
interleaved A/B, 2-3 rounds each):

| Workload | A (shipped profile) | B (+`h32`) | Delta |
|---|---|---|---|
| 11.2k prefill | 1.55 / 1.54 k tok/s | 1.60 / 1.60 | **+3.9 %** |
| 11.2k decode | 295.9 / 291.4 tok/s | 288.7 / 292.4 | within noise |
| 47.6k prefill | 1.41 / 1.39 | 1.44 / 1.43 | **+2.1 … +2.9 %** |
| 47.6k decode | 234.6 / 226.2 | 228.1 / 227.3 | within noise |
| PPL (8,190 tokens) | 3.98352 | 3.96913 | slightly better (FP32 sum order) |

Acceptance: keep the prefill gain, decode inside the card's ~2.4 % spread, and PPL not worse on the
same build. Also measure QwenPaw-Flash-9B and Qwen3.5-4B once, since they share the key.

### A2 — T10 profile: `gdn_two_stage/h16` (0.8B) · branch `sm75-p10-gdn-routes`

The 0.8B has `linear_num_value_heads = 16`, so it looks up `gdn_two_stage/h16`, which is not in the
calibration catalog at all (`src/calibration/route_catalog.cpp` lists only `h48` and `h32`).
A3 adds the catalog entry; A2 adds the profile route.

Acceptance: measured prefill/decode on `Qwen3.5-0.8B.ninfer` against the shipped profile, PPL
unchanged or better. The 0.8B is small enough that the GDN share of prefill is worth measuring
first with nsys — if it is under ~2 %, this item is dropped rather than carried.

### A3 — The calibration instrument covers all four head geometries · branch `sm75-p11-calibration`

`src/calibration/route_catalog.cpp:62-90` hard-codes `constexpr int query_heads_per_kv_head = 24 / 4`
and spells every key as `attn_i8_small/h24/<coding>/w<n>`. `calibration_candidates()` throws
`route "…" is not in the calibration route catalog` for anything else, so
`--device-profile calibrate` cannot produce a profile for a 9B (16 over 4), a 4B (16 over 4) or a
0.8B (8 over 2). `src/calibration/device_calibration.cu:704-705` has the same two constants.

Change: generate the key and its candidate tiers per geometry, from the geometry's own
`GroupSize = QHeads / KVHeads` and the row-tile count that follows from it, for
`(24,4) (16,4) (16,2) (8,2)` — the four pairs `with_causal_geometry` already admits. Template the
calibration body on the geometry the way `launch_small_t_i8` is templated.

**The digest must be bumped.** `calibration_route_catalog_digest()` hashes every key and candidate,
and `t10-sm75.json` records `"route_catalog": "ac2d7b3e311831b0"`; a build whose digest differs
*ignores* the file with a log line (`device_profile.cpp:287-292`). The digest is reproduced exactly
by hashing the catalog the way `route_catalog.cpp:120-140` does — verified against the shipped value
before this item was written, so the new value can be computed rather than guessed.

Acceptance: `ninfer-calibrate --print` runs on the T10 for all four geometries without throwing;
`t10-sm75.json` carries the new digest and still loads (the startup log line names the profile);
the h24 route entries it produces are unchanged from today's.

### B1 — The 9B vision merger projects to 4096 rows · branch `sm75-p12-vision`

All three 9B towers are byte-identical to the 27B's (depth 27, hidden 1152, ffn 4304, 16 heads of
72, position table 2304×1152, `q4_g64_fp16 [3456,1152]`×81, `q5 [1152,1152]`/`[1152,4304]`×27,
`q6 [1152,1536]`, merger fc1 `q8 [4608,4608]`). The **only** difference is the merger's second
projection: `q8_g32_fp16 [5120,4608]` on the 27B against **`[4096,4608]`** on the 9B, because the 9B
text tower is 4096 wide.

`src/ops/linear/q8/q8_dispatch.cpp` `kShapes` has `5120x4608`, `4608x4608` and `2048x4608`, not
`4096x4608`, so the load throws `q8 linear: unsupported shape` — reproduced on the T10:

```
$ ninfer-serve Ornith-1.5-9B-MTP-Q6_K.ninfer --vision --vision-residency resident
FATAL server failed during startup | q8 linear: unsupported shape
```

Change: one shape file, its selector declaration, one `kShapes` entry, one `sources.cmake` line. The
declaration goes in a **new** header included only by `q8_dispatch.cpp`: `q8_shapes.h` is included
by 36 translation units, and a one-line addition there costs a full rebuild for no reason.

The attention path needs nothing: 1152 / 16 = 72, which is exactly the geometry
`src/ops/softmax_attention/dense/packed/packed_softmax_attention.cpp:16-27` admits.

Acceptance: the 9B starts with `--vision-residency resident` and encodes one image; the answer is
sensible; `--vision-residency cpu` still works; PPL of the text path unchanged.

### B2 — The 4B and 0.8B vision linear shapes · branch `sm75-p12-vision`

Neither tower shares a shape with anything in the tables. Measured on the T10, both throw
`q4 linear: unsupported shape`.

| Model | q4 | q5 | q6 | q8 |
|---|---|---|---|---|
| 4B (hidden 1024, ffn 4096) | `3072x1024` (q/k/v), `4096x1024` (fc1) | `1024x1024` (attn out), `1024x4096` (fc2) | `1024x1536` (patch embed) | `4096x4096` (merger fc1), `2560x4096` (merger fc2) |
| 0.8B (hidden 768, ffn 3072) | `2304x768`, `3072x768` | `768x768`, `768x3072` | `768x1536` | `3072x3072`, `1024x3072` |

Same mechanical shape as B1, eight shapes per family set. Per-family selector declarations go in
new headers so `q4_shapes.h` (16 TUs), `q5_shapes.h` (9), `q6_shapes.h` (6) and `q8_shapes.h` (36)
are not touched.

Acceptance: both models start with vision resident and encode an image correctly; text PPL
unchanged.

### B3 — `packed_softmax_attention` admits a 64-wide head · branch `sm75-p12-vision`

`packed_softmax_attention.cpp:16-19` fixes `kHeadDim = 72`, `kHeads = 16`,
`kExpectedScale = 1/sqrt(72)`, and `kernel.cuh:14,116` fixes `kPackedAttentionHeadDim = 72` and
`QKKs = ceil(72/16) = 5`. The 4B tower is 1024/16 = **64** and the 0.8B is 768/12 = **64 with 12
heads**, so neither can use the op even once B2 gives it weights.

Change: template the head dimension and head count on the existing hot instantiation and add the
64-wide/16-head and 64-wide/12-head ones. `QKKs` becomes `ceil(D/16)`, which is 4 at D=64 — a
tighter fit than the 72-wide tower. The whole change is inside `packed/` (one `.cuh`, one `launch.cu`,
one `.cpp`, ~7 translation units).

**This touches the 27B and 9B path**, so it lands after B1 is verified and must re-verify 9B
afterwards: keep D=72/16 heads as a distinct instantiation whose arithmetic is unchanged.

Acceptance: 4B and 0.8B encode an image and answer sensibly; 9B and 27B still start and encode;
text PPL unchanged everywhere.

### B4 — The CPU vision encoder is also 72-wide only · branch `sm75-p12-vision`

`src/models/qwen3_5/execution/vision_cpu.cpp:17-23` fixes `kHeadDim = 72`, `kAxisPairs = 18`,
`kRotaryPairs = 36` and bakes `kRopeInverseFrequency[18]`. Its guard at
`src/models/qwen3_5/load/vision_cpu.cpp:147` throws *"CPU Vision needs the two-axis rotary tower
geometry (72-wide heads)"* — that string, not the GPU path, is the source of the `72-wide heads`
notes in `~/local_model/ninfer.ini`. So the 4B and 0.8B have no vision at all today, CPU or GPU.

Change: derive the head dimension, the axis pairs and the rotary frequency table from the tower's
config instead of baking them.

Acceptance: 4B and 0.8B start with `--vision-residency cpu` and encode an image; 9B and 27B CPU
encoding unchanged.

### C1 — A T10 attention tier table for `h16` · branch `sm75-p15-attn-h16`

Depends on A3. The 9B/4B have `GroupSize = 16/4 = 4`, so for a verify width of 3 the row-tile count
is `(3*4+15)/16 = 2`; the compiled fallback picks `<8 warps, 2 CTAs>` and the first hand-written
attempt at an `h16` table replaced it with a single-CTA tier and lost decode:

| Workload | A | B (+ naive `h16` table) |
|---|---|---|
| 11.2k prefill | 1.54 k tok/s | 1.60 (+3.9 %) |
| 47.6k prefill | 1.40 | 1.43 (+2.1 %) |
| 47.6k decode | 236.1 / 229.5 tok/s | **217.8 / 216.7 (−6 … −9 %)** |

The naive table is therefore **not** a merge candidate. What is left is a per-width sweep like
`db404f24` did for `h24`: measure every tier the catalog offers at every width, on the prefill *and*
the decode workload, and keep a table only if the combined effect is a gain. The judgement is on
overall speed, not on prefill alone.

Acceptance: a table whose measured effect is positive on both workloads, or a recorded rejection.

### C3 — Does this port carry the 512-wide Turing tile race? · branch `sm75-p16-attn-race`

Upstream `0fed5054` says its combine barrier "fixes the 512-wide tile race on every Turing part",
and `ebac85f7` before it selected measured tiles. Both are absent from this tree, and this port's
attention work is the INT8 family where upstream's is FP8/BF16, so the two do not overlap — which
means the race, if it is real here, is still live.

This starts as an investigation, not a change: read upstream's two commits, locate the equivalent
barrier in this tree's prompt and chunked kernels, and decide whether a race exists. Only then
decide whether a fix is in scope.

Acceptance: either a concrete defect with a reproducer, or a statement with evidence that this
port's kernels are not exposed.

### D1 — KVMem: clamp a prefill step to the pages that are available · branch `sm75-p14-kvmem`

`planning/startup.cpp:1390` `effective_prefill_chunk()` is `min(options.prefill_chunk,
options.max_context)`; it never looks at the KV pool. `ninfer-fusion-kvmem`'s
`NINFER_PREFILL_CHUNK_AUTO` clamps a step to `(usable − 8 pages) * 64` so a large chunk against a
small pool cannot ask for more pages than exist — the failure mode it removes is a 429
(`this step needs N free, M available`) at admission.

Acceptance: a pool smaller than the prefill chunk serves the request instead of refusing it; the
chunk actually used is logged; T10 prefill speed on the acceptance artifact unchanged.

**Result: the premise does not hold here, and the failure mode it describes does not exist in this
tree.** Upstream's clamp removes a 429 at admission, `this step needs N free, M available`. That
message is not in this tree: `grep -rn "this step needs" src/` finds nothing, and the only admission
refusal is `context cache exhausted: an idle Engine found no placement`
(`src/runtime/engine/engine_core.h:2505`), which fires when no active lane can release anything --
a different condition, and not one a prefill clamp would change.

What this tree does instead was measured rather than assumed. With a 51,200-token window (budget
20,480 + gen-reserve 30,720, `--prefill-chunk 1024`) on the acceptance artifact:

- A single 147,447-token prompt answered correctly with a needle at 10%, 25%, 50%, 75% and 90% of
  its length -- all five found, 367 s each. So a prompt three times the pool is served in full.
- A three-turn conversation whose history reached 168,080 tokens: turn 1 planted the needle at 40%
  of an over-pool prompt (417 s); turns 2 and 3 recalled it exactly (168,110 and 168,141 tokens of
  history, 3.6 s and 3.1 s). So the history survives eviction and restore, not just the live window.

That is the opposite of upstream's documented behaviour, where a prompt longer than the pool loses
its middle silently. Their engine and this one differ in more than the clamp: they run the ring with
`--max-shared-prefixes 0` and four protected buckets, this tree runs a retrieval-based window that
keeps the history in host memory.

So there is no clamp to add. What was worth adding instead is the observability that would have made
this question answerable from a log rather than by benchmark -- the `[kvmem]` startup line in D3,
which prints the window, the pool, the host budget and the prefill chunk against the reserve.

### D2 — KVMem: the startup line that says what speculation is doing · branch `sm75-p14-kvmem`

`[spec] backend=… draft_tokens=… adaptive=… attention_window=… ngram=…`, printed once at startup,
`backend=none` when off. Pure observability, no behaviour change; it is what makes "is the
speculation I asked for actually on" answerable without a benchmark.

Acceptance: the line appears with the values the flags imply, for MTP on and off.

### D3 — KVMem: the budget identity · branch `sm75-p14-kvmem`

`ninfer-fusion-kvmem` splits the pool into `protect / skeleton / evidence / recent / free` and hard-
throws when `sum != pool`. This tree has `sink_blocks()` and `recent_blocks()` against
`selected_tokens` (`include/ninfer/types.h:558-591`) with a validator that checks sink+recent fit,
but nothing that proves the four buckets add up at runtime, so a bookkeeping slip shows up as a
wrong answer rather than a refused start.

Acceptance: the invariant is asserted where the window is planned; no change to measured speed; PPL
unchanged on the acceptance artifact with KVMem on.

### E1 — Router admission must know how much VRAM is free · branch `sm75-p13-router-kvmem`

`ModelRegistry::make_room_locked` (`src/serve/model_registry.cpp:168-199`) decides eviction from
`loaded_count_locked() < policy_.models_max` alone; `grep` for any memory probe in
`model_registry.cpp`/`generation_service.cpp` returns nothing. So with `models-max 2` and a 27B
already holding 95 % of the card, a request for a second model evicts nothing, tries to load, and
fails deep inside the Program's allocation with a message that does not say "the card is full".

Change: before creating a service, compare `cudaMemGetInfo` free against the artifact's payload
(`artifact::schema.h:92` `payload_bytes`) plus the fixed stage budget already defined for exactly
this purpose (`src/models/qwen3_5/load.cpp:141-143`, 1.5 GiB per stage + 1.0 GiB for the head —
1.0 GiB on the host-pinning path), and keep evicting while a victim exists. If it still does not
fit, throw `ModelLoadFailed` naming the requirement and the free amount.

Acceptance: with a 27B loaded, a request for a second 27B is refused with that message instead of a
raw allocation failure; with room, a second small model still loads; `models-max 1` unchanged.

### E2 — `--kvmem-*` on the command line is refused in router mode · branch `sm75-p13-router-kvmem`

`build_catalog` (`src/serve/model_catalog.cpp:207-212`) appends the command line to **every** model's
arguments, last and therefore winning. A `--kvmem-budget` on the router's command line is applied to
models whose preset never asked for KVMem, and each of those then fails the large precondition block
at `src/serve/serve_options.cpp:1914` — twenty lines of `KVMem requires …` for a mistake whose cause
is the command line.

Change: while building the catalog in router mode, refuse a command-line KVMem flag and name the
fix: it belongs in that model's own section of `--models-preset`.

Acceptance: a router started with `--kvmem-budget` on the command line fails immediately with that
message; the same flag inside `[id]` loads normally.

### E4 — Open KVMem to the router · branch `sm75-p13-router-kvmem`

Two hard refusals stand in the way and nothing else does: `serve_options.cpp:1649-1652`
(`options.router()` is one disjunct of a three-way refusal) and `apps/serve/main.cpp:352-354`
(`if (options.router() && model_options.kvmem.selected_tokens != 0) throw`). The runtime is already
per-model — `generation_service.cpp:273` copies `engine_options.kvmem = options.kvmem;` into each
Engine, and the KVMem window is owned by the Program.

This lands **after** E1 and E2, so that opening the gate cannot expose a card-full hard failure or a
command-line leak.

Known asymmetric side effect, a design property and not a defect: KVMem forces
`context_cache.enabled = false` and `kv_capacity = explicit(window * max_concurrency)`
(`serve_options.cpp:1918-1919`), so a KVMem model in the router gets no prefix reuse while a
non-KVMem one does.

Acceptance: a router whose preset has one KVMem model and one without, `models-max 1`, serves both;
switching A → B → A works; host memory returns after an unload (RSS back to baseline); `/props`
reports `kvmem` for the KVMem model only.

### E3 — The unload path with KVMem attached · branch `sm75-p13-router-kvmem` (measurement)

**Measured while verifying E4**, which is the only way this could be measured: two 27B/9B KVMem
models behind one router on a 16 GiB card do not fit together, so every request crosses the eviction
path.

**The coexistence question, answered by three configurations rather than one.** What matters to an
operator is not whether KVMem and the router can be made to start together, but whether two models
stay resident:

| second model | both load-on-startup | `models-max 2` | result |
|---|---|---|---|
| 9B KVMem + 0.8B ordinary | yes | 2 | **both resident**, alternating requests answer in 0.53/0.79 s then 0.30/0.07 s |
| 9B KVMem + 0.8B KVMem | yes | 2 | **both resident**, same pattern (0.59/0.48 s then 0.26/0.08 s) |
| 27B KVMem + 9B KVMem | yes | 2 | does not fit in 16 GiB; each request evicts and reloads the other |

So KVMem does not force serial residency, and it is not the KVMem that fills the card -- 16 GiB holds
a 9B plus a 0.8B with KVMem on both. The rule is the ordinary one: `models-max 2` keeps two models
resident whenever the card can hold them, and evicts by unload (never sleep, because a KVMem Engine
cannot be suspended) only when it cannot.

    21:13:40  capacity | q8mtp   loaded        runtime 1.99 GiB | free 1.32 GiB
    21:14:04  capacity | ornith15 loaded       runtime 1.46 GiB | free 7.13 GiB    (q8mtp unloaded)
    21:14:47  capacity | q8mtp   reloaded      runtime 1.99 GiB | free 1.32 GiB    (ornith15 unloaded)

    request 1  q8mtp      43.1 s
    request 2  ornith15   25.5 s
    request 3  q8mtp      44.6 s

Each request pays a full weight load, because a KVMem model cannot be suspended -- `model_instance`
refuses `--model-suspend` with KVMem, so the registry's `suspendable()` answers no and it takes the
unload path. The numbers are the load times of these two artifacts, not a KVMem penalty: the host
payload budget is released with the Engine and re-pinned on load, and nothing is left behind. The
knob that decides how often this is paid is `--unload-idle-seconds`; a router with two KVMem models
and no eviction headroom is a serial reload loop, and should set it high enough that a conversation
does not cross it.

Not a code change: a probe that loads a KVMem model, forces eviction by requesting another model,
then requests the first again, watching RSS, VRAM and the KVMem counters. What is being falsified is
the teardown assumption: `third_party/kvmem/src/host/*.cpp` has no static mutable state, `~NvmeKvTier`
closes all three descriptors, and the pinned tiers are owned by the Program, so the reading is that
it is clean — but the sequencing across an eviction is exactly what reading cannot prove.

### D4 — Answering a prompt longer than the pool (awaiting the user's decision)

Plain language: today, if you give a model a prompt longer than the KV pool, the engine keeps the
beginning and the end and throws away the middle. It does not report an error — you get HTTP 200 and
a fluent, confident, wrong answer, and the warning is a line in the server log. The engine's own
comment at `src/serve/generation_service.cpp:299-337` says so: the middle "measured to go missing
with no error line, HTTP 200 and a plausible wrong answer; the root cause is not localized. Treat
every answer to this request as unverified".

`ninfer-fusion-kvmem` has the same bug and did not fix it either.

The proposed change is not a fix, it is honesty: refuse the request (HTTP 4xx with the pool size and
the prompt size) instead of answering it wrongly. The trade is that a client which today receives a
plausible answer would receive an error; the benefit is that no answer is silently wrong.

This needs the user's decision because it changes observable behaviour. Not scheduled.
