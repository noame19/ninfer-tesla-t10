# Remaining sm_75 (Tesla T10) work

Everything that is **not** the KVMem effort, written down so it can be picked up later without
re-deriving it. KVMem is being handled separately and is deliberately not described here.

The measurements quoted below were taken on the T10 in this repository's history; **do not spend
another CI round re-measuring them**. Where a number is missing, that is stated rather than guessed.

## Where the port stands

Working and measured on the card: `rk4v4-e8` KV (int8 gave +31% decode over bf16 at 32k; rk4v4-e8
raised the ceiling from ~24k to ~98k), the hand-written T10 device profile (+4.3% prefill at 47.6k
against PPL 3.01132 → 3.01334), and decode is at the memory wall and closed as such (see
`docs/device-profiles/`). Prefill attribution at 11.2k: MMQ 71%, attention 14%, GDN 9%, other 6%.

Two routes were tried and killed, with the evidence in their commit messages — do not restart them:

- **FP16 dequantize + cuBLAS prefill** (reverted in `d30b20f2`): cuBLAS really does run FP16 at
  43 TFLOP/s here against MMQ's ~33, but staging every weight through FP16 costs 108 GB per prefill
  chunk (~12.6 s at 394 GB/s) against a 12.2 s measured regression. The lever is not the GEMM peak.
- **Staging the GGUF vector kernel's loads through shared memory** (reverted in `31716fa6`): L1
  sectors fell 35%, DRAM bytes did not move (30.12 → 30.21 MB), speed 20.5 → 19.3 t/s.

---

## P0 — dense geometries other than the 27B

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

### What refuses a load (fix these first, they are validator-only)

1. `src/ops/weight_input.cpp:169-190` — the fused input projection's geometry whitelist. The two
   branches it accepts (`dense` 27B, `moe`) are the *same structural rule* with different numbers:
   attention wants `third == q && fourth == k` (gate shaped like the query, value like the key) and
   GDN wants `k == q && third == fourth` (key like query, z like value). The literals add no
   information, so generalizing them merges the branches into one rule.
2. `src/ops/weight_input.cpp:262-266` — `prepare_gdn_gating_proj_weights` takes `{48,5120}` or
   `{32,2048}`; a 9B/4B is `{32,4096}`/`{32,2560}`. The rows are the GDN value heads.
3. `src/ops/wrapper/gdn_input_proj.cpp:1305-1309` — `kGgufHidden 5120`, `kGgufQkvRows 10240`,
   `kGgufZRows 6144`, `kGgufQueryKey 2048`, `kGgufValue 6144` validate every GGUF GDN projection and
   the conv composition. All five follow from the tensors the Op is handed (`qkv = 2*key + value`,
   `value = z`), so they can be derived instead of declared.

### What has no kernel or route (the real work)

4. **GDN gating projection** — the largest item. `src/ops/gdn_gating_proj/bf16/` has exactly two
   instantiations: `Bf16Gdn27Geometry {48,5120,BlockN 128}` and `Bf16Gdn35Geometry {32,2048,BlockN 64}`
   (`..._gemm_mma.cuh:38-49`), selected by `is_27`/`is_35`
   (`..._plan.cpp:179-185`) against two closed route tables (`k27Routes`, `k35Routes`) with
   sm_86 occupancy asserts. A 32-head/4096-hidden or 32-head/2560-hidden tower needs a third
   geometry struct, the same schedule set instantiated for it, and its own route table. The kernel
   is templated on the geometry, so this is instantiation plus table work; the route bounds are
   tuning constants and want the T10's own measurement, not the sm_86 ones.
5. **Causal attention** — `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:51-55`
   accepts `(query_heads, kv_heads) ∈ {(24,4), (16,2)}` only, against
   `CausalD256H24Kv4`/`CausalD256H16Kv2` (`geometry.cuh:17-18`). A 9B/4B is `(16,4)`; the
   `small_t_i8_w*_h16_*` kernels are already instantiated per head count, so this is likely an alias,
   the whitelist entry and the `SmallTSplitScale` choice — but that scale is a tuning constant, so it
   needs a measurement rather than a guess.
6. **MTP** — `src/ops/kernel/mtp_pack.cuh:9-11` hard-codes the 27B's `14336 = 6144+1024+6144+1024`
   rows for `mtp_split_attn_in_kernel`, so a 4B (10240) or 9B (10240) MTP block would split at the
   wrong offsets. The four row counts are the block's own attention shapes; pass them in.
7. **Vision** — the 9B's tower is the 27B's (depth 27, hidden 1152, ffn 4304, 16 heads) except the
   merger's `projection_dim` 4096 vs 5120, so it may already fit; the 4B's (depth 24, hidden 1024,
   ffn 4096) and the 0.8B's (depth 12, hidden 768) are new. The vision projections take the official
   Vision formats, whose plans are shape-literal too, so this needs its own pass.

Already general, no work needed: the GGUF linear/vector/MMQ products (`gguf_project` requires only
`k % 512 == 0`), the GDN recurrent and chunked routes (`head_map::of` derives the head mapping from
the operands), the conv split geometry for 9B/4B sizes
(`CausalConvSplitGeometry::Rows2048x2048x4096` exists), the GDN fold records for `16x32x8192`, and
embedding — a GGUF artifact applies no Hadamard rotation, so the 5120-wide rotation routes are not
on this path.

**What "done" looks like.** A converted 9B and a converted 4B start under `ninfer-serve`, run the
serving smoke test with vision and MTP, and match llama.cpp's own perplexity on the same GGUF within
that quant level's noise — plus the existing 27B artifacts still starting unchanged. Requires one
sm_75 build (GitHub Actions; this host has no nvcc) and the T10, so it needs the card free.

**Size.** Moderate to large: two validator generalizations, one kernel-geometry instantiation with
its route table, one attention geometry, and one kernel parameterization.

---

## P2b — a register-staged prefill kernel

**Goal.** Replace the int8 MMQ kernel's blocking global→shared tile load with a software-pipelined
one, so prefill stops being the thing that is 71% of prefill time.

**Why it is hard, and why config cannot fix it.** The vendored MMQ compiles for sm_75 at `REG:255`,
`STACK:48`, so one block is resident per SM — 25% occupancy — and its tile wants ~56 KiB of shared
memory against Turing's 64 KiB, so there is no room for a second stage anyway. The tree's vendored
ggml also has **no `cp.async` infrastructure at all**: `CP_ASYNC_AVAILABLE` is defined and never
used, so there is nothing to switch on. This is a from-scratch pipeline in vendored code.

**What "done" looks like.** Prefill tok/s at 11.2k and at 47.6k, measured against the current
figures, with a `ninfer-perplexity` PPL check on the same build. **A speed number alone does not
qualify it.**

**Size.** Large. Not attempted.

**Blocker specific to this host.** The MMQ kernel cannot be profiled here: `ncu` with kernel replay
on it hard-froze this machine twice (no swap, 23 GB, weights resident). Use `nsys` for attribution;
`ncu` only on small, short kernels, single section, `-c 1`.

## P4 — a Turing prompt-attention tile

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

## Dropped

**P9 (qsa_indexer TF32)** — the code path only exists for the Qwen4Exp / Flash-Next sparse models.
This artifact is dense, so there is nothing to port. Closed.

---

## Verification rule for anything above

Every item needs the same two-sided gate this port has used throughout: **speed or context measured
on the T10**, *and* **`ninfer-perplexity` PPL on the same build**. A change that is faster and
dumber is a regression, not a win. Compare against the numbers above, not against a fresh baseline,
and record both sides in the commit message.
