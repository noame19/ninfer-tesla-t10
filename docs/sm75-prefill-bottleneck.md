# P2b — why the int8 MMQ kernel is slow, measured on the T10

> **Outcome (2026-10-15): the pipeline this document argues for was written and measured, and it
> is ~1.7% SLOWER. P2b is closed as a failed optimisation — do not restart it.** The half-tile
> split was numerically exact (`PPL 3.01334` on every run, 12 interleaved A/B rounds), so the idea
> was implemented correctly; it simply does not pay. The reason is in "Why it lost" at the end:
> the kernel is register/occupancy bound, not stalled on the weight fetch, so overlapping the fetch
> buys nothing while the doubled `__syncthreads()` count costs synchronisation. The load of these
> numbers is a correction to the *hypothesis* in "Why it is slow", not to the measurements.

Measurement host: NVIDIA Tesla T10 (TU102, 56 SMs, 4 MiB L2, 16 GB GDDR6), Debian 12, driver
615.71.09 / CUDA UMD 13.4. Artifact
`Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-Q8MTP.ninfer` (mixed IQ3_S/IQ4_XS/Q4_K/Q5_K/
IQ3_XXS/IQ2_S/Q2_K/Q4_0, 15.29 GiB of weights, 27.32 G weight-MACs per token column).

## Reproduced baseline (commit 7cdafc8d, profile on)

| measure | value |
|---|---|
| prefill @ 11,200 tokens | 479.0 tok/s (TTFT 23.4 s) |
| prefill @ 47,600 tokens | 363.2 tok/s (TTFT 131.2 s) |
| decode @ 47,600 tokens | 17.8 tok/s |
| PPL `--text` 8,190 scored tokens | 3.01334 (matches the port's recorded 3.01334) |

`--prefill-chunk` sweep at 11.2k: 128 → 395.7, 256 → 426.3, 512 → 450.1, **1024 (default) → 479.0**
tok/s. Smaller chunks are worse, so keeping the activation working set inside L2 by shrinking the
chunk does not pay. 2048 and 4096 refuse to start (`requested Engine runtime reservation requires
3988188416 bytes, but only 3557097472 bytes are available for runtime capacity`).

## Why it lost (P2b, measured)

`sm75-p2b` split the x tile into its two 128-value halves and filled one while the MMA read the
other (see `mmq-load-tiles.cuh`, `ggml_cuda_mmq_load_tiles_half_*`). Four `__syncthreads()` per K
step against the original two. Result, four interleaved A/B rounds at 11.2k and four at 47.6k:

| build | 11.2k prefill | 47.6k prefill | PPL |
|---|---|---|---|
| `stage1` (CI, no P2b) | 460.7 / 459.0 | 351.5 / 351.2 | 3.01334 |
| `stage2` (CI, no P2b) | 474.0 / 467.0 / 460.3 | — | — |
| local `sm75-p2b` | 452.8 / 452.1 | 346.4 / 346.4 | **3.01334** |

The binary diff says exactly why. `cuobjdump --dump-resource-usage` over the whole executable:
the P2b vs no-P2b builds differ in **29 functions, all of them MMQ kernels**, and every one of them
**grew**. `mul_mat_q<Q4_K,J=16>` REG 128→138, `mul_mat_q<Q5_K,J=16>` 140→150,
`mul_mat_q<IQ3_XXS,J=16>` 124→144, up to **+26 registers** on the `J=16` shapes, with the matching
`moe_mul_mat_q` variants growing too. The kernel already compiles at `REG:255`, so the extra live
values across the pipeline are exactly what the schedule could not absorb — occupancy and issue
slots, not the weight fetch, are what the T10 MMQ is short of.

**The wider lesson.** "Why it is slow" below derives a per-K-step cost of 6.8 µs fixed + 0.086 µs
per column and attributes the fixed part to a latency-chained weight fetch. That attribution is
what P2b tested, and P2b's failure falsifies *the fix*, so treat the fixed 6.8 µs as unexplained
rather than as fetch latency. Any further attempt on this kernel should start from the register
budget (`REG:255` is already the ceiling) and from instruction issue, the two resources P2b
overran, and should not add `__syncthreads()`.

## Kernel attribution (nsys, 2,100-token prefill, 4.52 s of GPU kernel time)

| kernel | share |
|---|---|
| `mul_mat_q` IQ3_S J=128 | 31.2% |
| `mul_mat_q` Q4_K J=128 | 13.8% |
| `mul_mat_q` IQ4_XS J=128 | 13.8% |
| `mul_mat_q` IQ3_XXS J=128 | 9.0% |
| `mul_mat_q` IQ2_S J=128 | 7.6% |
| `mul_mat_q` Q2_K J=64 | 1.0% |
| MMQ total (all types, all J) | **~80%** |
| `causal_attention_small_t_i8_tiled_kernel` | 4.7% |
| GDN two-stage (recurrence + prepare) | 3.8% |
| `store_plane_kernel` | 3.0% |
| `bf16_gdn_gating_proj_gemm_mma_kernel` | 2.2% |

## Why it is slow: latency, not bandwidth

`nsys --gpu-metrics-set=tu10x` (needs sudo: `ERR_NVGPUCTRPERM` otherwise), 2 kHz, sampled inside
the three slowest `mul_mat_q` launches:

| metric | IQ2_S J=128 (6.99 ms) | Q2_K J=64 (9.57 ms) |
|---|---|---|
| SMs active | 97.7% | 97.8% |
| SM issue | 30.1% | 31.6% |
| Tensor (INT8) active | 15.8% | 18.6% |
| **DRAM read** | **4.8%** | **6.5%** |
| DRAM write | 2.3% | 2.1% |
| Unallocated warps in active SMs | 73.2% | 73.1% |

Calibration: `ninfer-hbm-bandwidth-probe`'s `read_u128` kernel sustains 394.1 GB/s (97.7% of the
403.3 GB/s the device advertises) and reports **88%** on the same DRAM-read metric. So 1% ≈ 4.5
GB/s and MMQ moves only ~22-29 GB/s.

With 8 warps per SM (256 threads, one 56 KiB block resident → 25% occupancy) and a
`__syncthreads()`-separated load phase and MMA phase, the SM is idle on both pipes at once. The
kernel is stall-bound; every resource has 3x or more headroom.

## The traffic model that the measurement refutes

A naive model says each weight element is read once per column tile (`ntx = ceil(T/J)`) and each
activation element once per row tile (`nty = R/I`). At T=1024, J=128, I=128 that predicts, per
1024-token chunk: 131.4 GB of weights and 245.6 GB of activations — 957 ms at 394 GB/s, against
1212 ms measured, i.e. 79% of a DRAM bound. **The DRAM metric refutes it**: the reuse is caught
before DRAM, and the kernel is nowhere near the bandwidth wall. Do not re-derive this model as a
reason to change the tile ordering or to shrink the prefill chunk; both were measured and neither
helps (the chunk sweep above, and the 4 MiB L2 is smaller than any tile working set worth keeping).

## What the numbers say to build

Per K-step (256 values) with I=J=128, one block per SM:

| | bytes | at 7.04 GB/s per SM |
|---|---|---|
| x tile (IQ4_XS weights, 128 rows) | 17,408 | 2.5 us |
| y tile (activations, 128 columns, 2 k-blocks) | 36,864 | 5.2 us |
| total | 54,272 | 7.7 us |
| MMA (128x128x256 int8) | — | 3.6 us at the INT8 peak share |

So a perfectly overlapped kernel is bandwidth-bound at 7.7 us per K-step, against ~12.3 us
measured. The ceiling is ~2-3x on MMQ, which is ~80% of prefill.

Shared memory is the hard constraint: Turing has 64 KiB per SM and the current tile wants 57,856 B
(`ids 512 + x 128*76*4 + y 128*144`), so there is no room for a second stage and only one block
fits per SM. Any pipeline has to buy its second buffer by shrinking a tile.

## How to measure it again

```bash
cd /home/agent/work-rom/t10-perf-run
./ab.sh  <serve> 800 16 tag                 # prefill @ 11.2k
./ab.sh  <serve> 3400 16 tag                # prefill @ 47.6k
./ppl.sh <ninfer-perplexity> tag            # the quality gate, expect 3.01334
./nsys_pf.sh <serve> 150 tag                # kernel attribution -> /tmp/nsys_<tag>.nsys-rep
./dram.sh <serve> 70 tag                    # GPU metrics; needs sudo, calibrate against the probe
```

`nsys stats --report cuda_gpu_kern_sum --format csv /tmp/nsys_<tag>.nsys-rep` for the roll-up; the
GPU-metric samples live in the `.sqlite` nsys writes beside the report (`GPU_METRICS` table,
metricId 22 = DRAM read, 23 = DRAM write, 9 = tensor, 8 = SM issue, 17/20 = warps).

Do not run `ncu` kernel replay on `mul_mat_q`: it hard-froze this host twice (no swap, 23 GB,
weights resident).
