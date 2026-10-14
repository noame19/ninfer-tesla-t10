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
