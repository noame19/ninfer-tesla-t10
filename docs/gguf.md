# GGUF block formats

NInfer serves GGUF releases that give every tensor its own ggml quantization type, such as
ISTA-DASLab's [GSQ-RCO models](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
without converting their weights to another format. The converter recipe `qwen3_8_27b_gguf` copies
each quantized tensor's blocks of a Qwen3.8-27B release into the artifact unchanged (see
[weight conversion](weight-conversion.md#a-mixed-precision-qwen38-27b-gguf)), and the runtime
multiplies them in place. The Qwen3.8-Flash-Next releases convert the same way with
`qwen3_8_flash_next_gguf`, their routed experts kept as expert-major banks
([Qwen3.8-Flash-Next](qwen3-8-flash-next.md)).

## Formats

Eighteen block types: the fifteen llama.cpp writes for dense models, `Q8_0`, `Q2_K`, `Q3_K`,
`Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_NL` and `IQ4_XS`, and `Q2_0`, `Q4_0` and `Q5_0`, which the Flash-Next releases use, stored as
the `gguf_*` formats of the `gguf_blocks_v1` layout
([tensor formats](maintainer/tensor-formats.md#35-gguf-block-formats),
[storage layouts](maintainer/storage-layouts.md#6-gguf_blocks_v1)). A projection may mix types:
the parts of a fused projection that share a type are multiplied together, the others one by one
against the same activation.

## Products

Every product quantizes its BF16 activation to ggml's `q8_1` numbers, one scale and one sum per 32
values, as llama.cpp does, and accumulates in FP32; the weights' represented values are exactly
those of `ggml-quants.c`.

- **Up to eight columns** (decode, speculative verification, small batches), the vector kernel
  (`src/ops/linear/gguf/ggml_bridge_vec.cuh`) decodes each 32-value slice of a row once into int8
  words and dots it with every column. A warp owns two rows; a `[gate; up]` pair of one type is one
  launch that writes `silu(gate) * up`.
- **More columns** (prompts) take llama.cpp's integer tensor-core kernel (`mul_mat_q`, vendored
  unmodified in `third_party/ggml-quants` under its MIT license) with its Ampere tile table.
  `IQ1_M` has no such kernel; its wide products dequantize to BF16 and run cuBLAS.
- **Token table** rows are dequantized exactly.

The scales of `IQ2_XXS`, `IQ2_XS`, `IQ2_S` and `IQ3_XXS` are applied in FP32 where llama.cpp's
vector kernels round an integer rescale, so a decode step's logits can differ from llama.cpp's in the
last bits; prompts use llama.cpp's own arithmetic.

## Converting any dense Qwen3.5/3.8 release

The block import is not tied to the 27B widths. `qwen35_dense_gguf` (the same recipe every published
27B GGUF artifact was converted with under its original name `qwen3_8_27b_gguf`) reads the tower's
geometry from the base checkpoint's resolved text config and then requires the GGUF's own header and
complete tensor set to agree with it, so one recipe serves every dense release and a file that is not
the tower its config describes is refused instead of guessed at:

| | hidden | layers | attention | GDN key/value heads | conv | intermediate |
|---|---:|---:|---|---|---:|---:|
| Qwen3.8-27B | 5120 | 64 | 24q/4kv | 16/48 | 10240 | 17408 |
| Qwen3.5-9B | 4096 | 32 | 16q/4kv | 16/32 | 8192 | 12288 |
| Qwen3.5-4B | 2560 | 32 | 16q/4kv | 16/32 | 8192 | 9216 |
| Qwen3.5-0.8B | 1024 | 24 | 8q/2kv | 16/16 | 6144 | 3584 |

```bash
python3 -m tools.convert \
  --model /path/to/base-config-only \
  --recipe qwen35_dense_gguf \
  --source gguf=RELEASE.gguf \
  --source vision=mmproj-RELEASE.gguf \
  --components text,vision,mtp \
  --name my-model \
  --out models/my-model.ninfer
```

`--model` supplies only `config.json` and the frontend resources, so the base release's own files are
enough. Three exporter conventions the geometry decides:

- **The MTP block** is the layer *after* the last one, so it is read from the file's own `nextn`
tensor rather than assumed to be `blk.64`, which is what lets a 33-block `-mtp` export (32 real
layers plus the head) import at `blk.32`. `--components ...,mtp` requires the block; a text-only
conversion of a file that carries one imports the text tower and leaves it in the file.
- **A tied tower** stores no `output.weight`: llama.cpp's converters drop it and the runtime reads
the token table. The recipe detects that from the config and aliases the two parameters, so the
blocks are stored once and the head keeps its own Use.
- **The GDN gates** (`ssm_alpha`, `ssm_beta`) are BF16 operands of the Engine. Releases in the wild
store them as BF16, F32, F16, Q8_0 or Q4_K; every stored form is decoded to the values it represents
before that single cast, so a gate never acquires a second quantization.

### Checking that nothing was lost

`tools.convert.gguf_fidelity` proves the artifact still holds the source's bytes, from the outside:
it reads the artifact's objects through the container reader and the GGUF's rows straight off the
mapped file, derives the geometry from the artifact's recorded config, and re-derives every row map
(identity, the head-interleaved `attn_q` split, the exporter's tiled value-head order) from that
geometry alone.

```bash
python3 -m tools.convert.gguf_fidelity --gguf RELEASE.gguf --artifact models/my-model.ninfer
```

Every row of every `gguf_*` object must match, including the permuted families, whose wrong ordering
would otherwise still load and still read fluently. Measured on the releases in this repository's
`local_model/model` tree:

| Artifact | Imported rows | Bytes identical to the source GGUF |
|---|---:|---:|
| Swift-1.5-Qwen3.8-27B-UD-Q3_K_XL | 4,460,544 | 13.10 GiB, 100.0000% |
| QwenPaw-Flash-9B-heretic-MTP-Q4_K_M | 1,969,152 | 5.76 GiB, 100.0000% |
| Ornith-1.5-9B-uncensored-Q4_K_M | 1,922,048 | 5.61 GiB, 100.0000% |
| Qwen3.5-4B-Q4_K_M-mtp | 1,415,168 | 2.80 GiB, 100.0000% |

What is *not* byte-identical is the vision tower and the small direct tensors, which take the
official Vision formats and BF16/FP32 exactly as the 27B artifacts do; `ssm_alpha`/`ssm_beta` are the
only text values that change, by one BF16 rounding.

### Which towers the Engine serves

Converting is general; **loading is not**. The Engine's dense path still carries the 27B's geometry
in several places, so of the towers above only a 27B artifact starts today. The 9B, 4B and 0.8B need
the engine work listed in [remaining sm_75 work](remaining-sm75-work.md#p0--dense-geometries-other-than-the-27b);
they convert now, and their artifacts are the input that work is validated against.

## Serving

A converted artifact starts like any other; MTP, DFlash2 and Vision work as with the official
artifact. The GSQ-RCO IQ3_S conversion is published as
[WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3):

```bash
./build/apps/ninfer-serve models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --model-id qwen3.8-27b --kv-dtype rk8v4 --gdn-state-fp16 --spec mtp --draft-tokens 3
```

## Measurements

GSQ-RCO IQ3_S (3.5 bits per weight, 10.95 GiB of weights) against the official Qwen3.8-27B NInfer
artifact (Q4/Q5 groups, 15.9 GiB of weights).

WikiText-2 test perplexity with the protocol of the GSQ-RCO model card (the test rows joined by
blank lines, disjoint 2,048-token windows, `ninfer-perplexity --disjoint`, BF16 KV):

| Artifact | Perplexity |
|---|---:|
| GSQ-RCO IQ3_S in NInfer | 7.071 |
| GSQ-RCO IQ3_S, model card | 7.07 |
| BF16 base model, model card | 7.05 |
| Official Qwen3.8-27B NInfer artifact | 7.286 |

The reasoning benchmarks of the [capability evaluation](../eval/README.md), with the protocol the
official artifact was scored with (EvalScope, temperature 1.0, one sampled run each, an RTX 5090):

| Benchmark | GSQ-RCO IQ3_S in NInfer | Official Qwen3.8-27B NInfer artifact |
|---|---:|---:|
| IFBench (prompt-level strict) | 80.33% | 77.67% |
| AIME 2025 | 100.00% | 96.67% |
| AIME 2026 | 100.00% | 96.67% |
| GPQA-Diamond | 88.38% | 87.37% |

The GSQ-RCO card reports 100 on AIME 2025 and 89.39 on GPQA-Diamond with its own protocol (BF16:
100 and 89.90). One run of GPQA-Diamond's 198 questions varies by about two points.

Speed against the official artifact on the same card in the same sitting, `rk8v4` KV, one request,
greedy, thinking off, with the reference client (`tools/bench/refbench.py`): an RTX 3090 at 420 W
and an RTX 4090 at 450 W with a 176,128-token window (the official DFlash2 row on the RTX 4090 at
167,936), an RTX 5090 at 450 W with 262,144. Short chat is the mean decode rate over five
512-token answers; GSQ-RCO IQ3_S / official:

| | RTX 3090 | RTX 4090 | RTX 5090 |
|---|---:|---:|---:|
| short chat, no speculation, tok/s | 59.9 / 40.3 | 70.6 / 55.0 | 107.5 / 88.1 |
| short chat, MTP, 3 drafts, tok/s | 108.7 / 82.7 | 146.4 / 109.3 | 221.4 / 178.1 |
| short chat, DFlash2, 5 drafts, tok/s | 115.6 / 109.2 | 169.4 / 141.3 | 226.5 / 222.5 |
| decode after 32K tokens, no speculation, tok/s | 50.4 / 38.4 | 66.0 / 52.2 | 100.3 / 83.2 |
| time to first token, 32K prompt, s | 20.5 / 22.2 | 9.3 / 12.8 | 8.6 / 9.3 |

The lead comes from reading 31% fewer weight bytes per token. Verification of several drafts gains
less than plain decode because the i-quants cost more arithmetic per byte than the official Q4/Q5
groups, and every extra column adds its own dot products; DFlash2's six-token verification is
where the two meet.

