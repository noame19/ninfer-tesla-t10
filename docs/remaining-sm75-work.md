# Remaining sm_75 (Tesla T10) work

The work list for the sm_75 port. `AGENTS.md` holds the branch, build, verification and reporting
rules that apply to every item here; this file holds the items themselves and the measurements they
rest on.

**Numbering.** Two different things have a number and they must not be confused:

* **`P<n>` is a task's identity.** It never changes, and it matches the branch that carries it
  (`sm75-p3` for `P3`). Two tasks that must ship in one build share a branch only when they share a
  directory, which is called out where it applies.
* **`#<n>` is the order of execution** as of the last revision of this table. Reorder it freely;
  the `P` numbers stay put.

The measurements quoted below were taken on the T10 in this repository's history; **do not spend
another build re-measuring them**. Where a number is missing, that is stated rather than guessed.

## Order of execution

| # | Task | Branch | Build cost | What it is |
|---|---|---|---|---|
| **#1** | **P1** | `sm75-kvmem-official` | ~15 min incremental | KVMem long context: make the branch serve 242k |
| **#2** | **P5a** | `sm75-p5` | **none** (JSON) | The nine remaining false attention tiers |
| **#3** | **P4** | `sm75-p4` | ~5 min/round | A Turing prompt-attention tile (98k: attention is 47–51%) |
| **#4** | **P0a** | `sm75-p0` | ~2 min | The three validators; unblocks 4B/9B startup |
| **#5** | **P5b** | with P4 | 60–100 min | The hard-coded fallback in `small_t_i8_launch.cuh`, which JSON cannot reach |
| **#6** | **P0b** | `sm75-p0` | ~5 min | Attention geometry `(16,4)` |
| **#7** | **P0c** | `sm75-p0` | ~10 min | GDN gating geometry for 9B/4B |
| **#8** | **P3** | `sm75-p3` | ~20 min | GDN cooperative split-K, sized from the runtime SM count |
| **#9** | **P0d/P0e** | `sm75-p0` | ~10 min | MTP row offsets, vision tower |
| **#10** | **P6** | `sm75-p6` | **1–2 h/round** | bf16→f16 hoist, TF32 saturation |
| **#11** | **P7** | `sm75-p7` | ~10 min | MTP shortlist draft head |

Why this order: **P1** is the only item that may already be fixed upstream and is the one that
unlocks the long context this port exists for. **P5a** costs no build at all and finishes a job
that is most of the way done. **P4** has the cheapest iteration cost of anything that needs a build.
**P0a** makes a local 4B artifact load, which turns every later P0 item into a runnable check.
**P5b shares `causal_cache/` with P4**, so it rides P4's build instead of waiting for its own.
**P6 goes last among the cheap ones** because `third_party/ggml-quants/src/ggml-cuda/mma.cuh` is
included by 76 translation units, i.e. 1–2 hours per round — the same wall-clock as P4's whole

> **Status correction (2026-10-21).** This table predates the merge of the Qwen3.5 tower line.
> P0a, P0b, P0c and P0d are **already on `sm75`** (`4bc9b5ad`, `64c41822`, `6c1a26e5`, merged by
> `1fd86015`) and the working tree confirms each one; only **P0e (vision)** is still open, and the
> detail below says why. P5a landed as `66b4bb61` + `054c56ef`, and P4's prompt route landed as
> `420dabe4` + `bac38fc8`. P3 and P6 remain unmerged test branches by the user's decision, not by
> omission. The live execution queue is [`todo.md`](todo.md); this file keeps the P-numbered history.
implementation, for one hoist.

## Where the port stands

Working and measured on the card: `rk4v4-e8` KV (int8 gave +31% decode over bf16 at 32k; rk4v4-e8
raised the ceiling from ~24k to ~98k), the hand-written T10 device profile, and decode is at the
memory wall and closed as such (see `docs/device-profiles/`). Prefill attribution at 11.2k: MMQ
71%, attention 14%, GDN 9%, other 6%.

Current acceptance figures (Tesla T10, the 27B acceptance artifact, `rk4v4-e8`, 56 SMs):

| Workload | Before | Now |
|---|---|---|
| 11.2k prefill | 455.5 tok/s | **467 tok/s** |
| 47.6k prefill | 358.4 | **386** |
| 98k prefill | 276.2 | **299** |
| PPL (8190 scored tokens) | 3.01334 | **3.01334** (bit-exact) |

The gain is commit `db404f24` — the w6/w7/w8 attention tiers past 8192 keys moved off `6x2x32qe`,
which cannot place two CTAs per SM (16,448 B static + 32,768 B dynamic shared memory = 49,216 B
against Turing's 65,536 B). **P5a is the same fix for the nine tiers not yet touched**, and **P5b**
is the same bug in the compiled fallback.

Three routes were tried and killed, with the evidence in their commit messages — do not restart them:

- **FP16 dequantize + cuBLAS prefill** (reverted in `d30b20f2`): cuBLAS really does run FP16 at
  43 TFLOP/s here against MMQ's ~33, but staging every weight through FP16 costs 108 GB per prefill
  chunk (~12.6 s at 394 GB/s) against a 12.2 s measured regression. The lever is not the GEMM peak.
- **Staging the GGUF vector kernel's loads through shared memory** (reverted in `31716fa6`): L1
  sectors fell 35%, DRAM bytes did not move (30.12 → 30.21 MB), speed 20.5 → 19.3 t/s.
- **A register-staged MMQ prefill pipeline** (P2b, branch `sm75-p2b`, not merged): every changed
  kernel grew by up to 26 registers on a kernel already at `REG:255`, and `__syncthreads()` doubled
  from 2 to 4 per K step. Measured **−1.6 to −1.7%** at 11.2k and 47.6k with PPL bit-exact. The
  MMQ is register/issue bound, not stalled on the weight fetch — which falsifies the "fixed 6.8 µs is
  the x fetch" reading in `docs/sm75-prefill-bottleneck.md`. See that file's "Why it lost".

---

## P1 — KVMem long context

**Goal.** Make `sm75-kvmem-official` serve a 242k context under the flags this work is specified
with:

```
--max-context 262144 --kvmem-budget 53248 --kvmem-gen-reserve 30720 \
  --kvmem-host-mib 5120 --kvmem-sessions 1
```

The reported failure is that the probe is lost in the second half of the context (around 242k).

**The fix is probably already in the branch and simply absent from the binary.**
`sm75-kvmem-official` HEAD is `7eb0e516`, titled `fix(kvmem): restore the ring's position
arithmetic, which the merge had flattened`. The stage3 binary was built from run 38509724894 at
`aa07252b`, two commits earlier. The fix's own message describes the symptom exactly: the merge
dropped `compact_position` from the ordinary batch position, leaving KV addressing correct **only
while the ring is idle** — its recorded measurement is the same 70k prompt with the probe at the
start, `stage1, no KVMem, 98k pool -> HIT` against `stage3, KVMem, 40,960 pool -> MISS`, and
"every short-context test looked fine: with the pool at least as large as the context the ring never
compacts and the two expressions agree".

So this starts as **a build, not a bug hunt**: rebuild the branch at `7eb0e516` and re-run the
242k probe. Only if that fails does the ring arithmetic get read.

**Build cost is low.** Relative to the merge base (`df125470`), the branch changes only **five**
`.cu`/`.cuh` files (3 under `src/core`, 1 `src/ops/token_sums`, 1
`src/ops/softmax_attention/dense/causal_cache`) out of 103 changed files. Note that
`git merge-base sm75-p4 sm75-kvmem-official` is `df125470`, so the branch does **not** contain
`db404f24`; merging this line's work in is a separate step.

**What "done" looks like.** The 242k probe hits under the flags above, short-context behaviour is
unchanged, and the existing 98k figures are not regressed.

**Size.** Small if the fix is the one already committed; unbounded if the ring arithmetic is wrong
in some other way.

---

## P0 — dense geometries other than the 27B

**Branch `sm75-p0`.** Items P0a–P0e below are one task for build purposes: they share the acceptance
object (a converted 4B/9B artifact starting and running) and most of them are plain C++.

**Goal.** Serve a converted GGUF of any dense Qwen3.5/3.8 tower, not just the 27B. The converter is
already general: `qwen35_dense_gguf` reads the geometry from the base config, validates the GGUF's
header and tensor set against it, and stores each block row unchanged, so a 9B, 4B or 0.8B release
converts today and its stored rows are byte-identical to the source (tools/convert/gguf_fidelity.py;
see [GGUF block formats](gguf.md#converting-any-dense-qwen35qwen38-release)). **Loading is what is
still 27B-shaped**, in the places below. Positions are on `sm75`.

| Tower (hidden / layers / attention / GDN heads / conv) |
|---|
| 27B: 5120 / 64 / 24q-4kv / 16-48 / 10240 — supported |
| 9B and 4B: 4096 or 2560 / 32 / **16q-4kv** / 16-32 / 8192 |
| 0.8B: 1024 / 24 / **8q-2kv** / 16-16 / 6144 — not needed |

### What refuses a load (fix these first, they are validator-only) — **P0a, #4**

**DONE on `sm75` (`4bc9b5ad`).** The three validators are now derived from the tensors rather than
from the 27B's literals: `src/ops/weight_input.cpp:172-199` uses the structural rule
`attention ? third == q && fourth == k && q[1] == k[1] : k == q && third == fourth && q[1] == third[1]`,
`src/ops/weight_input.cpp:262-280` only requires `shape == matrix(b)`, and
`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh:42-71` carries
`Bf16Gdn9Geometry {32,4096}`, `Bf16Gdn4Geometry {32,2560}` and `Bf16Gdn08Geometry {16,1024}`
next to the 27B's and 35B's. `src/ops/softmax_attention/dense/causal_cache/geometry.cuh:23-25` adds
`CausalD256H16Kv4` and `CausalD256H8Kv2`, and `src/ops/kernel/mtp_pack.cuh:25-31` takes the four row
counts as parameters. Verified on the T10: 9B, 4B and 0.8B all start and serve text.

Three local artifacts are already converted and wait for exactly this:
`/mnt/disk238g/ninfer_model/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M-MTP.ninfer` (2.9 GB),
`ornith-1.5-9b-uncensored.ninfer` (5.7 GB), `swift-27b-ud-q3_k_xl.ninfer` (12.8 GB). No conversion
is needed. The 4B fails today with:

For the record, these are the three places that used to refuse a load, and what each one became:

```
FATAL server failed during startup | text/layers/0: input projection: unsupported logical projection geometry
```

1. ~~`src/ops/weight_input.cpp:169-190` — the fused input projection's geometry whitelist.~~ **Done:**
   the two branches it accepted (`dense` 27B, `moe`) were the *same structural rule* with different
   numbers, and the literals added no information, so they merged into one rule in
   `src/ops/weight_input.cpp:172-199`.
2. ~~`src/ops/weight_input.cpp:262-266` — `prepare_gdn_gating_proj_weights` takes `{48,5120}` or
   `{32,2048}`; a 9B/4B is `{32,4096}`/`{32,2560}`.~~ **Done:** `src/ops/weight_input.cpp:262-280`
   only requires `shape == matrix(b)` and names the mismatch instead.
3. ~~`src/ops/wrapper/gdn_input_proj.cpp:1305-1309` — `kGgufHidden 5120` and friends.~~ **Done:** the
   five constants are derived from the tensors the Op is handed (`qkv = 2*key + value`, `value = z`).

### What has no kernel or route (the real work) — **P0b (#6), P0c (#7), P0d/P0e (#9)**

Items 4, 5 and 6 are **done** on `sm75`; item 7 is the one still open.

4. ~~**GDN gating projection**~~ **Done (`6c1a26e5`).** `src/ops/gdn_gating_proj/bf16/` now carries
   `Bf16Gdn9Geometry {32,4096,BlockN 64}`, `Bf16Gdn4Geometry {32,2560,64}` and
   `Bf16Gdn08Geometry {16,1024,64}` alongside the 27B's and 35B's (`..._gemm_mma.cuh:42-71`). The
   closed route tables are still what makes a geometry runnable, so a **new** tower still needs a
   route table and a T10 measurement — the difference is that the four shipped towers no longer do.
5. ~~**Causal attention**~~ **Done (`64c41822`).** `src/ops/softmax_attention/dense/causal_cache/geometry.cuh:23-25`
   now defines `CausalD256H24Kv4`, `CausalD256H16Kv4`, `CausalD256H16Kv2` and `CausalD256H8Kv2`, with
   `kCausalSmallTSplitScale<KVHeadsValue> = 4 / KVHeadsValue`, and `with_causal_geometry` enumerates
   the four pairs. The *tiers* those geometries launch are still a T10 measurement — that is item C1
   in [`todo.md`](todo.md), and it is where a naive table already lost 6–9 % on decode.
6. ~~**MTP**~~ **Done.** `src/ops/kernel/mtp_pack.cuh:25-31` takes `q_rows` and `kv_rows` and computes
   `attn_rows = 2*q_rows + 2*kv_rows`; the 27B's `14336` is no longer baked in.
7. **Vision — still open (P0e), and now precisely located.** The 9B's tower is byte-identical to the
   27B's except the merger's second projection, which is `q8_g32_fp16 [4096,4608]` against the 27B's
   `[5120,4608]`; `src/ops/linear/q8/q8_dispatch.cpp` has no `4096x4608` entry, so the load throws
   `q8 linear: unsupported shape` — reproduced on the T10 with `--vision-residency resident`. The 4B
   and 0.8B towers are genuinely new (`{depth 24, hidden 1024}` and `{depth 12, hidden 768}`): they
   need new q4/q5/q6/q8 shape entries *and* a `packed_softmax_attention` that admits a 64-wide head,
   because this tree's op is fixed at `kHeadDim 72` / `kHeads 16`
   (`packed/packed_softmax_attention.cpp:16-19`). The CPU fallback is 72-wide too
   (`execution/vision_cpu.cpp:17-23`), which is where the `72-wide heads` notes in the operator's
   `ninfer.ini` come from. Work items B1–B4 in [`todo.md`](todo.md).

Already general, no work needed: the GGUF linear/vector/MMQ products (`gguf_project` requires only
`k % 512 == 0`), the GDN recurrent and chunked routes (`head_map::of` derives the head mapping from
the operands), the conv split geometry for 9B/4B sizes
(`CausalConvSplitGeometry::Rows2048x2048x4096` exists), the GDN fold records for `16x32x8192`, and
embedding — a GGUF artifact applies no Hadamard rotation, so the 5120-wide rotation routes are not
on this path.

**What "done" looks like.** A converted 9B and a converted 4B start under `ninfer-serve`, run the
serving smoke test with vision and MTP, and match llama.cpp's own perplexity on the same GGUF within
that quant level's noise — plus the existing 27B artifacts still starting unchanged. **Built locally**
(see `AGENTS.md`): this host has a working Docker toolchain now, so no cloud round is needed. The
T10 has to be free for the run.

**Size.** Moderate to large: two validator generalizations, one kernel-geometry instantiation with
its route table, one attention geometry, and one kernel parameterization.

---

## P4 — a Turing prompt-attention tile

**Branch `sm75-p4`, shared with P5b.** P5b also edits `causal_cache/`, so it rides this branch's
build rather than paying for its own.

**Goal.** Give the prompt path an attention kernel of its own, so the long-context prompt does not
fall into the 8-token chunked fallback.

**Why it is hard.** Two hard limits in the current code: `static_assert(TokenTile * GroupSize <= 48)`
with `GroupSize = 6` caps the chunk at 8 tokens, and the prompt kernel's `Bc` is pinned to the
64-key KV page (`K + V + FP16V = 64 KiB`), which is why `causal_softmax_attention.cpp` routes the
prompt path to `ChunkedSmallT`. Neither is a tuning problem; both need a new kernel.

At 11.2k this is only ~14% of prefill, so it was deprioritised. **At 53k it is 31–43%** (from the
prefill fit: time/token = 1.90 ms + 2.65e-5 · T), so its value grows with context — which matters
more now that KVMem pushes the context past 98k.

**What "done" looks like.** 47.6k and 98k prefill tok/s plus PPL, and a check that no route is
silently falling back (log the chosen kernel).

**Size.** ~400–700 lines of new kernel. Quality-critical: attention numerics changing under a
degraded softmax would be silent.

## P5 — the attention tiers that promise a residency the kernel cannot reach

**Branch `sm75-p5`.** The profile half (`P5a`) needs no build; the fallback half (`P5b`) is compiled
and is done with P4.

**Goal.** Stop the small-T int8 kernel from asking for two CTAs per SM when its shared memory admits
one, and stop the profile from shipping tier names whose middle field is a lie.

**Evidence.** `small_t_i8.cuh:62` declares the kernel
`__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm)`, and `docs/device-profiles/README.md` documents
the tier spelling as `"<warps>x<CTAs per SM>x<key block>"`. The kernel's footprint, read back from
nsys rather than computed, is **16,448 B of static shared memory plus a 32,768 B dynamic arena**
(TokenTile 8, KeyBlock 32) = 49,216 B. Turing has 65,536 B per SM, so `MinBlocksPerSm > 1` is
**unsatisfiable**: every launch runs a single CTA, and `__launch_bounds__` has already made the
compiler sink registers for a residency that never happens.

Every tier in `docs/device-profiles/t10-sm75.json` whose middle field is 2 or 4 is affected --
`2x4x32`, `4x2x32`, `8x2x32`, `2x2x32q`, `4x2x32q`, `8x2x32q`, `4x2x32e`, `8x2x32e`, `4x2x32qe`,
`8x2x32qe`, `6x2x32`, `6x2x32qe`. Only the x1 tiers are honest. `12x1x64d` is worse than dishonest
and is already disabled under `NINFER_SM75` (65,536 B of dynamic arena is all of Turing's shared
memory).

**This is not only a profile problem.** `small_t_i8_launch.cuh:192-210` is a hard-coded fallback
taken whenever the profile routes nothing, and it selects `6x2x32` for `implementation_window >
8198`. Because it is a fallback it covers **every member of the INT8 family**
(`kv_cache_is_int8_family`: `Int8Group64`, `RotatedInt8KeyInt4ValueGroup64`,
`RotatedLloyd4KeyInt4Value`, `RotatedInt4KeyInt4ValueE8`, `RotatedE8RootKeyInt4Value`), not just the
`rk4v4-e8` the profile names. Editing JSON alone cannot reach those four.

The w6/w7/w8 tiers were fixed this way on 2026-10-15 (commit `db404f24`, +7.7% at 47.6k, +8.3% at
98k, PPL bit-exact 3.01334); the remaining tiers are the same fix, unapplied.

**Why the w1-w5 tiers matter too.** They cover the verify width, 9 to 64 columns -- the path the PPL
gate itself runs on, and the one decode's speculative rounds use. A `4x2x32qe` tier that really runs
four warps is at 6.25% occupancy.

**What "done" looks like.** Prefill tok/s at 11.2k, 47.6k and 98k for the tiers that change, tok/s on
the verify width (9-64 columns) where `4x2x32qe`/`2x2x32q` are used, and PPL on the same build. The
second half needs a build, because the fallback is compiled in; the profile half does not.

**Size.** Small. A profile edit plus one fallback branch, but it must be validated at both widths.

---

## P3 — GDN gating cooperative split-K

**Goal.** Let the GDN gating projection split its K dimension cooperatively across blocks, instead
of one block per output tile.

**State.** The device-profile work already delivered the two-stage GDN part (`gdn_two_stage`,
`gdn_h48`), which is what the profile ships today. What is left is the cooperative split-K routing,
which must size itself from the runtime SM count (`device_sm_count()`), not a hardcoded 68 — a
hardcoded count is wrong on any card that is not the one it was tuned on, and this port must stay
correct on other sm_75 parts.

**What "done" looks like.** GDN's share of prefill (9% at 11.2k) measured before/after at 11.2k and
47.6k, plus PPL.

**Size.** Moderate; the risk is the hardcoded-count class of bug, not the arithmetic.

## P6 — bf16→f16 conversion hoist, and TF32 saturation

**Goal.** Stop converting bf16 to f16 per MMA operand in the inner loop; convert once when the tile
is loaded. Separately, make the TF32 path saturate rather than silently wrap.

**Evidence.** The conversion is at `mma.cuh:101-113` (`mma_bf16`). Turing has no native bf16 MMA, so
this conversion is on the hot path; hoisting it to the load stage is arithmetic-neutral.

**What "done" looks like.** Speed at the tensor-core-using shapes, plus a numeric test that the
converted tile is bit-identical to the per-operand conversion it replaces. For the TF32 saturation
part, a test that feeds the saturating input and asserts saturation instead of wrap.

**Size.** Small-to-moderate. The saturation half is a correctness fix, not a speed one.

## P7 — an MTP shortlist draft head

**Goal.** Stop each MTP draft step from re-reading the whole output head.

**Why.** With MTP at draft 4 the artifact's proposal head is absent, so each draft step reads the
full 248320×5120 output head — roughly 1.35 GB per round, K times per generated token. A shortlist
head (a small draft matrix over a candidate token subset) would cut that to the shortlist width.

**What it is not.** This is not `--lm-head-draft`; that flag is correctly refused on this artifact
because there is no proposal head to use.

**What "done" looks like.** Decode tok/s with `--spec mtp --draft-tokens 4`, accepted-draft rate
before/after (a shortlist that proposes worse tokens loses more than it saves), and PPL.

**Size.** Moderate plus a weight-conversion layer. Touches the artifact pipeline, so it is the item
most likely to need an artifact round-trip.

## P2b — a register-staged prefill kernel

**CLOSED — attempted and failed.** Kept for the evidence, not as a candidate. The branch `sm75-p2b`
is on GitHub as a record and is deliberately not merged.

**What was done.** The MMQ weight tile was split into its two existing 128-value halves and one half
was filled while the tensor cores read the other (5 half-tile loaders for Q4_K, Q5_K, IQ3_XXS,
IQ3_S, IQ4_XS — 94% of this artifact's weight bytes), with a four-barrier pipeline per K step.

**Result: −1.6 to −1.7%, with PPL bit-exact 3.01334.** Correct but slower, at both 11.2k (452.8
against 460.7) and 47.6k (346.4 against 351.5), consistently across 12 interleaved rounds.

**Why it lost** (from `cuobjdump --dump-resource-usage` over the 29 changed kernels): every changed
kernel *grew* — `<Q4_K,J=16>` 128→138 registers, `<Q5_K,J=16>` 140→150, `<IQ3_XXS,J=16>` 124→144,
up to +26 registers on a kernel that was already at `REG:255` — and `__syncthreads()` doubled from
2 to 4 per K step. **The MMQ is register/issue bound, not stalled on the weight fetch.** That
falsifies the "fixed 6.8 µs per K step is the x fetch" reading in
`docs/sm75-prefill-bottleneck.md`; see that file's "Why it lost".

**For any future attempt:** start from the register budget (`REG:255` is already the ceiling) and
instruction issue, and do not add `__syncthreads()`.

**Original statement of the task, kept for context.** Replace the int8 MMQ kernel's blocking
global→shared tile load with a software-pipelined one. The vendored MMQ compiles for sm_75 at
`REG:255`, `STACK:48`, so one block is resident per SM (25% occupancy) and its tile wants ~56 KiB of
shared memory against Turing's 64 KiB, so there is no room for a second stage anyway. The tree's
vendored ggml also has **no `cp.async` infrastructure at all**: `CP_ASYNC_AVAILABLE` is defined and
never used, so there is nothing to switch on.

**Blocker specific to this host.** The MMQ kernel cannot be profiled here: `ncu` with kernel replay
on it hard-froze this machine twice (no swap, 23 GB, weights resident). Use `nsys` for attribution;
`ncu` only on small, short kernels, single section, `-c 1`.

## Dropped

**P9 (qsa_indexer TF32)** — the code path only exists for the Qwen4Exp / Flash-Next sparse models.
This artifact is dense, so there is nothing to port. Closed.

---

## Verification rule for anything above

Every item needs the same two-sided gate this port has used throughout: **speed or context measured
on the T10**, *and* **`ninfer-perplexity` PPL on the same build**. A change that is faster and
dumber is a regression, not a win. Compare against the numbers above, not against a fresh baseline,
and record both sides in the commit message.
