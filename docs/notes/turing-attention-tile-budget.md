# What a Turing attention tile can actually afford (sm_75, TU102 / Tesla T10)

Reconnaissance for P4, plus the two external findings that change its shape. All numbers are
computed from this tree's own constants, not estimated.

## The limit is 64 KiB, but the prompt kernel's footprint is not 91 KiB

`causal_softmax_attention.cpp:420-429` routes the whole prompt path to `ChunkedSmallT` under
`NINFER_SM75`, with the comment that the prompt kernels "need 91-96 KiB" and "no tile shape fits".

That comment is about `prompt_bf16.cuh` / `prompt_i8.cuh` as they are currently instantiated. The
fast variant, `prompt_i8_fast.cuh`, was added later and its budget is much smaller. From its own
constants (`kCausalPromptI8FastBc = 64`, `kCausalPromptHeadDim = 256`, `kCausalPromptI8FastGroups = 4`):

```
one stage = K codes 64*256                       = 16384 B
          + packed int4 V 64*128                 =  8192 B
          + G64 key scales 64*4*2                =   512 B
          + value scales 64*8*2                  =  1024 B
                                                 = 26112 B
```

| Warps | Br | buffers | Q bytes | smem | vs 65536 |
|---|---|---|---|---|---|
| 8 | 128 | 2 (as instantiated) | 32768 | 84992 | over by 19456 |
| **8** | **128** | **1** | 32768 | **58880** | **fits** |
| 4 | 64 | 2 | 16384 | 68608 | over by 3072 |
| **4** | **64** | **1** | 16384 | **42496** | **fits** |

So **a single-buffered prompt kernel does fit** and the blocker is not shared memory — it is three
`static_assert`s in `prompt_i8_fast.cuh` that pin the shape to the double-buffered `Warps=8` case:

- `:82 static_assert(kCausalPromptI8FastBc == kPagedKVPageSize);`
- `:84 static_assert(CausalPromptI8FastShape<8>::SmemBytes == 100352);`
- `:75 static_assert(Warps == 4 || Warps == 8);`

`CausalPromptI8FastShape<8>::SmemBytes == 100352` follows from the template, so nothing instantiates
the single-buffered shape today. (100352 != 84992 because the template charges K/Q/Ks at *bfloat16*
width even when the kernel keeps them as int8; the live footprint is smaller than the template says.)

## Two external findings that matter

1. **K and V do not need separate shared memory.** FlashAttention's Turing support
   (`ssiu/flash-attention-turing`, `Dao-AILab/flash-attention#3`, `#787`) and vLLM's prompt-attention
   work both read K only for the scores and V only for `p*v`, so V can overwrite K after the scores'
   barrier. That halves the KV part of a stage: 24576 -> 16384 B. The same trick appears in a sibling
   project's PR #1821 for a **TU102, int8, prompt-attention** kernel, where merging the K and V
   buffers took a block from 37.9 KB to 29.2 KB and thus from one CTA per SM to two — with bitwise
   identical output.
2. **`cp.async` does not exist on Turing.** `src/ops/common/memory.cuh:66-95` makes `cp_async_zfill`
   a synchronous copy under `NINFER_SM75`, and `:111-118` makes `cp_wait` a compiler barrier. So every
   "prefetch" in this tree is already just a copy, and all of the sm_75 kernels are implicitly
   single-buffered in effect. Whatever an earlier design assumed it was hiding with double buffering,
   it was not hiding it.

With K/V shared and single buffering the budget opens up:

| Warps | Br | bufs | smem | CTA/SM |
|---|---|---|---|---|
| 2 | 32 | 1 | 26112 | **2** |
| 4 | 64 | 1 | 34304 | 1 |
| 6 | 96 | 1 | 42496 | 1 |
| 8 | 128 | 1 | 50688 | 1 |
| 8 | 128 | 2 | 68608 | over |

Two CTAs per SM is reachable at `Warps=2, Br=32` — and occupancy is exactly what the P4 measurement
implicated: the chunked small-T kernel's 49,216 B admits one CTA of 6 warps, i.e. 9.4% occupancy,
and its tensor utilisation measured ~6%.

## What P4 still has to check

Only `k00 = 0` and `Bc = 64` matter to the loop shape; the constants are name-bound, so a sm_75
instantiation has to be added rather than a new kernel written. `prompt_i8_fast.cuh` has exactly one
includer (`prompt.cu`), so the rebuild after such a change is minutes, not hours.
