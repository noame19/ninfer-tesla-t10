# NInfer · NVIDIA Tesla T10 (Turing sm_75)

An inference engine for the Qwen3.5 model family, ported to **Turing** — the Tesla T10
(TU102, 16 GB, compute capability 7.5). A single `ninfer-serve` binary serves the OpenAI,
Anthropic and llama.cpp APIs from one card.

This repository is a fork of [NInfer-all](https://github.com/iamwavecut/ninfer-all), which continues
[NInfer](https://github.com/Neroued/ninfer) through the NInfer-3090 line. The upstream engine targets
Ampere, Ada and Blackwell; everything here is the work of making it run, and then run *well*, on the
oldest architecture upstream's compatibility path still touches.

## What Turing is missing, and what the port does about it

Two hardware facts drive almost every change:

* **No `cp.async`** (added in Ampere). `src/ops/common/memory.cuh` stages copies through registers
  with `cp_commit`/`cp_wait` as no-ops; every staging loop already ends in `__syncthreads()`, which
  is what publishes the data.
* **No `m16n8k16` / `m16n8k32` tensor-core shapes.** Turing's f16 and int8 MMA shapes are
  `m16n8k8` and `m8n8k16`. `src/ops/common/mma.cuh` emulates `m16n8k16` as two `m16n8k8` and
  `m16n8k32` (s8) as four `m8n8k16`. `ldmatrix` exists on Turing, so fragment assembly is unchanged.

`CMakeLists.txt` admits `75`, runs the `NINFER_SM8X_COMPAT` path for it and defines `NINFER_SM75`.
FP8 A8 and NVFP4 W4A4 stay stubbed, as they are for `sm_86`. An `sm_75` build runs only on Turing.

## Measured on a Tesla T10

27B acceptance artifact (GGUF block formats, `rk4v4-e8` KV), 56 SMs, one request.

| Workload | Before the port's optimisations | Now |
|---|---:|---:|
| 11.2k-token prefill | 455.5 tok/s | **531 tok/s** |
| 47.6k-token prefill | 358.4 tok/s | **463 tok/s** |
| 98k-token prefill | 276.2 tok/s | **299 tok/s** |
| decode, 262k context | — | **17.8 tok/s** |
| perplexity (8,190 scored tokens) | 3.01334 | **3.00762** |

Perplexity is measured on both sides of every change and is the correctness gate; the table's
figure *improved*, which is the point — no optimisation here trades quality for speed.

## Optimisations that landed

Five changes survived measurement. Each one is a commit on the main line with its before/after in
the message.

**1. `rk4v4-e8` KV cache** — 4-bit keys and values with 8-bit exponential scales. On Turing this
raised the usable context from about 24k to about 98k tokens, and int8 KV had already given +31%
decode over bf16 at 32k. Decode sits at the memory wall and is closed as such.

**2. A hand-written T10 route profile** — `docs/device-profiles/t10-sm75.json` picks the kernel
route for every operation and width from what was measured on this card, instead of inheriting a
profile tuned on an RTX 3090.

**3. Honest attention tiers** (`db404f24`, `66b4bb61`, `054c56ef`) — the profile's attention tiers
named 2 CTAs/SM, but each one needs 49,216 B of shared memory (16,448 static + 32,768 dynamic)
against Turing's 65,536 B, so only one can ever be resident. The compiler was sinking registers for
a residency that could not happen. Retiring the unfillable `x2` tiers and naming every INT8-family
key so the profile can reach all five codings gave **+2.5% at 11.2k, +7.7% at 47.6k and +8.3% at
98k**, at bit-identical perplexity.

**4. Decode at the memory wall** — a decode step reads 15.29 GiB of weights once and cannot do
better; measured at 82% of the card's 394 GB/s. The remaining levers were the draft path and the
attention route, both taken below.

**5. MTP and DFlash2 speculative decoding** — `--spec mtp` measured **2.55× over plain decode** on
this card (52.7 against 20.7 tok/s with 4 drafts). `--spec dflash2` is implemented and works, but
measured *slower* than MTP here (39.8 against 44.6 tok/s at its best width), so MTP is the default.
The optimum draft width on a Tesla T10 is **2** (39.4 tok/s at 84.5% acceptance); wider drafts fall
off as acceptance drops.

### Tried, measured, and rejected

Recorded so the next person does not repeat them. Each was reverted; the numbers are in the commit
messages and `docs/sm75-prefill-bottleneck.md`.

| Attempt | Result |
|---|---|
| FP16 dequantize + cuBLAS prefill | −8%: staging every weight through FP16 costs 108 GB per chunk |
| Staging GGUF vector-kernel loads in shared memory | 20.5 → 19.3 tok/s; L1 sectors fell 35%, DRAM bytes did not move |
| Register-staged pipelined MMQ prefill | −1.7%, bit-identical perplexity: every kernel grew registers and `__syncthreads()` doubled |
| `__launch_bounds__(128,4)` on the GGUF vector kernel | +0.3%: occupancy is not the bottleneck |
| `__launch_bounds__(128,8)` on the same kernel | −67% at width 4, −79% at width 5: it spills |
| A narrower MMQ column tile for verify steps | J=16 already reads the weights once and is still 20% slower than the vector kernel |

## Building

Local Docker, not CI — a CI round costs about 1.7 GB of artifact downloads, and this build is
reproducible offline:

```bash
docker build -t ninfer-sm75-build .local-build
.local-build/build.sh scripts/build-native.sh configure
.local-build/build.sh scripts/build-native.sh target ninfer-serve ninfer-perplexity
```

The image is Debian 12 with the same versions CI uses (nvcc 13.4, CMake 3.31.6, g++-12). CUDA
headers and cublas are taken read-only from a CUDA 13 wheel already on the host, which avoids
downloading `libcublas-dev` (837 MB). See [`.local-build/README.md`](.local-build/README.md).

A cold build is about 3.5 hours; an incremental one is minutes.

## Serving

```bash
ninfer-serve MODEL.ninfer --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-dtype rk4v4-e8 \
  --prefill-chunk 1024 --spec mtp --draft-tokens 2
```

`--kv-dtype rk4v4-e8` is the all-round choice on a 16 GB card. `--draft-tokens 2` is the measured
optimum. The INT8-family KV formats route to the fast prompt-attention kernel automatically on
Turing, because there it is the only prompt kernel that fits in shared memory.

## Documentation

Upstream's guides still apply, with the port's additions indexed in
[`docs/README.md`](docs/README.md#sm_75-tesla-t10-port): the task list and every figure the port's
claims rest on, where prefill time actually goes, the T10 device profile, the local Docker
toolchain, and the measurement harness in [`tools/t10-perf/`](tools/t10-perf/README.md).
[`AGENTS.md`](AGENTS.md) records the branch, build and verification rules this line follows.

## Features

**Models and weight formats**

- **Qwen3.5 family.** Dense and MoE (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) with MTP,
  DFlash (35B-A3B) and DFlash2 (Qwen3.8-27B) drafters, Vision for images and video, and
  `groupwise-int`, FP8, NVFP4 and ternary weights. NVFP4 expert banks prefill through W4A4 and need
  a `120a` build.
- **GGUF block formats.** Releases that pick a ggml type per tensor, such as ISTA-DASLab's GSQ-RCO,
  convert without requantization and multiply their blocks in place. The 3.5-bit Qwen3.8-27B IQ3_S
  release holds 10.95 GiB of weights instead of 15.9, scores WikiText-2 perplexity 7.071 (the
  official artifact: 7.286) and decodes at 59.9 tok/s against the official artifact's 40.3 on an
  RTX 3090 ([GGUF block formats](docs/gguf.md)).
- **Ternary Bonsai 2 27B.** PrismML's [ternary Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
  runs from `t2_g128_fp16` weights: 2.125 bits per weight, imported without rounding, with the
  Hadamard rotations fused into the norms and gates and integer activations for its projections.
  The `bonsai2_27b_ternary` recipe adds ProCreations' MTP head and DFlash2 adapter and an exact
  proposal head.
- **Qwen3.8-Flash-Next.** The 125B-parameter MoE (512 experts, about 6B active) from the GSQ-RCO
  GGUF releases, with its experts on the GPUs, in pinned host memory with a GPU cache
  (`--expert-residency host`) or in the artifact's files read into a GPU cache
  (`--expert-residency disk`, under 1 GB of RAM). It serves up to eight requests with prefix reuse,
  structured output, images and video, and MTP drafting. On two RTX 5090s the Q2_0 release scores
  93.3% on AIME 2025 and 86.4% on GPQA-Diamond ([Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md)).

**Speculative decoding**

- MTP, DFlash and DFlash2 with 1 to 15 drafts; the artifact's proposal head drafts by default when it
  stores one. Under MTP and DFlash a sampled request keeps a draft only while it equals the model's
  own keyed draw, so every emitted token is the model's sample; DFlash2 verifies its proposal
  distribution by rejection sampling.
- `--adaptive-mtp` picks each round's width from measured draft survival and round cost.
- N-gram copy drafting beside the neural drafter (on with `--spec`) verifies up to 15 tokens copied
  from earlier prompt, tool-result or output text ([ngram copy proposals](docs/ngram.md)).

**Context and memory**

- Nine KV formats, from `bf16` to `rk2v4-e8` at 216 bytes per token and KV head; `rk8v4` is the
  all-round choice and `rk4v4` holds the native 262,144-token window of Qwen3.8-27B on a 24 GB card
  ([context and memory](docs/cli.md#context-and-memory)).
- Up to 1,048,576 tokens of context, four times the native window, with plain RoPE or YaRN
  (`--rope-yarn`).
- A context cache of checkpoints on the device, in pinned host memory and in an optional disk tier
  that survives restarts (`--disk-kv-path`, `--disk-kv-restore`), with endpoint and branch anchors
  that let an edited or regenerated turn resume where it left the conversation. The hybrid prefix
  cache (`--use-alt-prefix-caching`) shares content-addressed KV blocks across requests instead.
- Precision trades that buy context: `--embedding-q4`, `--lm-head-q6`, `--gdn-state-fp16`
  ([quality trades](docs/maintainer/quality-trade-experiments.md)).

**Serving**

- OpenAI Responses and Chat Completions, Anthropic Messages, llama.cpp's `/completion`,
  `/tokenize`, `/detokenize`, `/apply-template`, `/v1/rerank`, `/metrics`, `/slots` and `/props`,
  streaming, tool calls, structured output through xgrammar (`--structured-output`), token log
  probabilities and prompt grafts ([HTTP serving](docs/serving.md)).
- **Model suspend.** `--model-suspend` lets an idle server give its device memory back without
  exiting and take it again on the next request, retained conversations included. Ternary Bonsai 2
  27B on an RTX 3090: 7.9 GiB down to 0.3 GiB in 0.33 s, back in 1.1 s
  ([model suspend](docs/serving.md#model-suspend)).
- **Several models behind one server.** With `--models-dir` or a llama.cpp-style
  `--models-preset`, `ninfer-serve` is a router with llama.cpp's model API; with `--model-suspend` a
  model sleeps instead of unloading: two 27B models on one 24 GB RTX 3090 swap in 1.8 s instead of
  an 11.6 s cold load ([several models](docs/serving.md#several-models-router)).
- `GET /stats` and a [monitoring dashboard](tools/monitor/README.md), request logs with rotation,
  and an optional compiled-in WebUI.

**Kernels and devices**

- **Device route profiles.** Each card's measured profile picks the kernel route for every
  operation and width. Profiles are built in for the RTX 3090, 4090, 5090 and the three RTX PRO
  6000 editions; any other GPU is measured once at first start, and `ninfer-calibrate` re-measures
  ([device profiles](docs/device-profiles.md)).
- **Pipeline stages.** `--devices 0,1,...` splits the layers over up to eight GPUs on Linux, each
  stage owning its weights, KV and state ([pipeline stages](docs/maintainer/pipeline-stages.md)).
- FP8 A8 and NVFP4 W4A4 tensor-core routes on every `120a` build, an FP4 tensor-core prompt kernel
  for an NVFP4 KV cache, and programmatic dependent launches on Blackwell.

## Running

Download an artifact from the [table below](#artifacts) and serve it from the [Docker image](#docker)
or from a [build](#building). The server speaks the OpenAI and Anthropic APIs, and each card picks
up its device profile on its own.

```bash
hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local-dir models

# The image: `serve`, the artifact under /models, the flags. Listens on http://localhost:8080/v1.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 \
  --gdn-state-fp16 --spec dflash2 --draft-tokens 5

# A build: the same flags. Listens on 127.0.0.1:8080 unless --host and --port say otherwise.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

The recipes below are the configurations the [reference tables](docs/performance/reference-2026-09.md)
use. They are written for a build; in the container, replace `ninfer-serve models/` with the
`docker run ... serve /models/` line above.

<details>
<summary><b>Ternary Bonsai 2 27B</b>: DFlash2, MTP, the largest context, a disk tier</summary>

```bash
# Fastest single stream: DFlash2 with five drafts over the full 262,144-token window.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# MTP drafting through the proposal head.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3

# The largest context a 24 GB card holds: 958,464 tokens of rk4v4.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 958464 --kv-capacity 958464 --kv-dtype rk4v4 --gdn-state-fp16 --rope-yarn

# Adaptive MTP and a disk tier that keeps evicted conversations.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 5 --adaptive-mtp \
  --disk-kv-path /var/cache/ninfer --disk-kv-gib 64 --disk-kv-restore
```

- **Draft count.** Five drafts are the all-round choice. Seven are faster on short answers; after
  long documents three to seven win ([draft length](docs/performance/reference-2026-09.md#draft-length)).
- **Images.** `--vision --vision-residency overlay --vision-max-merged 12288` adds them. The encode
  borrows the drafter's memory, so the whole window still fits a 24 GB card.
- **Past 958,464 tokens.** The RTX 5090 and the PRO 6000 hold the 1,048,576-token maximum with
  `rk4v4`, DFlash2 or MTP included; the RTX 5090 holds 978,944 with `rk8v4`. Filled to 1,048,576
  tokens, the model misses the code at 90% (about 943K); up to about 880K it found every code on
  every card.

</details>

<details>
<summary><b>Qwen3.8-27B</b>: the official artifact and the GSQ-RCO IQ3_S release</summary>

```bash
# The official artifact on a 24 GB card: DFlash2 with five drafts over 245,760 tokens of rk4v4.
ninfer-serve models/qwen3_8_27b.ninfer --model-id qwen3.8-27b \
  --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# GSQ-RCO IQ3_S: MTP over 176,128 tokens of rk8v4.
ninfer-serve models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 176128 --kv-capacity 176128 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3
```

- **The official artifact with `rk8v4`.** The same speculation fits 167,936 tokens on an RTX 4090
  and 176,128 on an RTX 3090. An RTX 5090 takes the full 262,144 with either KV format.
- **IQ3_S.** It drafts with `--spec dflash2 --draft-tokens 5` as well, and `--vision` adds images.

</details>

<details>
<summary><b>Qwen3.8-Flash-Next</b>: experts on the GPUs, in host memory or on disk</summary>

```bash
hf download WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3 \
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer --local-dir models
hf download WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3 \
  Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer --local-dir models
M=models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer
T=models/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer

# Every expert on one RTX PRO 6000 (96 GB).
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768

# Every expert on two GPUs, one pipeline stage each (Linux): 24 GB cards for Q2_0, 32 GB for IQ3_S.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --devices 0,1

# One 24 GB GPU: the experts in pinned host memory, the most used of them cached on the GPU.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --expert-residency host

# One 24 GB GPU and little RAM: the experts stay in the file and stream into a GPU cache.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --expert-residency disk

# The container, experts in host memory.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all serve /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer \
  --ngram-table /models/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer \
  --model-id qwen3.8-flash-next --max-context 32768 --expert-residency host
```

- **Other releases.** IQ3_S and the Coder IQ1_M build take the same flags with their own file and
  the same table. The [tables above](#qwen38-flash-next) show the placements measured for each.
- **Memory.** Host experts pin 34 GB of RAM for Q2_0, 50 GB for IQ3_S and 25 GB for the Coder
  build. Disk experts need under 1 GB of RAM; the page cache does the rest. The GPU expert cache
  takes what is free after startup, or `--expert-cache-mib`.
- **The n-gram table.** Its rows are read from the file, 16 per token, unless
  `--ngram-residency ram` loads all 28.8 GB or `ram-hot` keeps the rows a profile ranks first in a
  RAM budget ([the n-gram rows](docs/qwen3-8-flash-next.md#the-n-gram-rows)). A model started
  without its table is refused; `--no-ngram-table` overrides that, an experimental mode with no
  practical use.
- **Drafting.** `--spec mtp --draft-tokens 3` drafts with the MTP block the published artifacts
  carry.
- **Images and video.** `--vision` adds the Vision tower (0.9 GB on the GPU).

</details>

<details>
<summary><b>Qwen3.6-35B-A3B NVFP4</b>: RTX 50 series and RTX PRO 6000 only</summary>

```bash
ninfer-serve models/Qwen3.6-35B-A3B-NVFP4-ninfer-v3.ninfer --model-id qwen3.6-35b-a3b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --vision
```

The NVFP4 experts prefill through W4A4 on Blackwell's FP4 tensor cores, so this needs a `120a`
build (the image has one); `sm_86` and `sm_89` builds refuse the file. On an RTX 5090 it starts in
17 s and leaves 7.4 GiB of the card free.

</details>

<details>
<summary><b>Qwen3.8-27B fine-tunes</b>: Huihui abliterated, HauhauCS Aggressive, MXFP8-CRACK</summary>

All three have the official Qwen3.8-27B identity and size, so the Qwen3.8-27B recipes above serve
them too; CRACK has no DFlash2 adapter. Their cards' configurations:

```bash
# Huihui abliterated on one RTX 3090: 198,400 tokens with MTP and Vision.
ninfer-serve models/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 \
  --vision --vision-residency overlay --vision-max-merged 12288

# HauhauCS Aggressive: DFlash2 with seven drafts.
ninfer-serve models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-ninfer-v3.ninfer \
  --model-id qwen3.8-27b --max-context 32768 --kv-capacity auto \
  --spec dflash2 --draft-tokens 7

# MXFP8-CRACK: MTP.
ninfer-serve models/Qwen3.8-27B-MXFP8-CRACK-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 32768 --kv-capacity auto --spec mtp --draft-tokens 3
```

</details>

<details>
<summary><b>A card without a built-in profile</b></summary>

```bash
ninfer-calibrate --print > my-gpu.json
```

The engine measures it by itself at first start. Running it by hand refreshes the profile after a
driver or clock change. See [device profiles](docs/device-profiles.md).

</details>

## Artifacts

The Bonsai, GSQ-RCO and Flash-Next artifacts use formats that only this engine reads.

| model | artifact | size | contents |
|---|---|---:|---|
| Ternary Bonsai 2 27B | [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 GiB | ternary weights, Vision, Bonsai-trained MTP head and DFlash2 adapter, exact proposal head |
| Qwen3.8-27B | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.03 GiB | the official `groupwise-int` (Q4/Q5) with MTP and DFlash2; the reference tables' artifact |
| Qwen3.8-27B GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 GiB | ISTA-DASLab's 3.5-bit GGUF blocks byte for byte, Q6_K MTP head, Vision, DFlash2, proposal head |
| Qwen3.8-27B, abliterated | [WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 GiB | the official `qwen3_8_27b` recipe with MTP, DFlash2 and a proposal head |
| Qwen3.8-27B Uncensored, HauhauCS Aggressive | [WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3) | 19.03 GiB | `groupwise-int` with the tune's MTP head, DFlash2 and a proposal head |
| Qwen3.8-27B MXFP8-CRACK | [WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3) | 16.96 GiB | `groupwise-int` with MTP and a proposal head; no DFlash2 |
| Qwen3.8-Flash-Next GSQ-RCO Q2_0 | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.49 GiB | ISTA-DASLab's 2.4-bit GGUF blocks byte for byte, Vision, shared-Q8_0 MTP; reads the n-gram table |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.50 GiB | the 3.5-bit release with Vision and shared-Q8_0 MTP |
| Qwen3.8-Flash-Next Coder GSQ-RCO IQ1_M | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 31.02 GiB | the expert-pruned coding build (256 experts per layer), Vision, shared-Q8_0 MTP |
| Qwen3.8-Flash-Next n-gram table | [WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.92 GiB | unchanged IQ4_NL rows plus an optional broad hot-row profile (`--ngram-table`) |
| Qwen3.6-35B-A3B NVFP4 | [WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 20.39 GiB | RedHatAI's NVFP4 experts code for code, Q8 projections, Vision, MTP, proposal head; `sm_120a` GPUs only |

The official NInfer artifacts (Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B, `groupwise-int` and
`nvfp4`) load too; the [documentation map](docs/README.md#model-artifacts) links them. Weight
conversion shows how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Docker

`ghcr.io/iamwavecut/ninfer-all:latest` is built from every `master` commit that passes CI, on CUDA
13.4 and Ubuntu 26.04. It carries two builds and starts the one that matches the GPU: `sm_86` for
the RTX 30 and RTX 40 series, `sm_120a` for the RTX 50 series and the RTX PRO 6000 Blackwell.

- **Host.** An NVIDIA driver of the CUDA 13 branch (580 or newer) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Tags.** `latest`, `sha-<commit>` and the `VERSION`; the image is 1.6 GB compressed.
- **Volumes.** `/models` holds artifacts, `/cache` the device profile measured on first start, and
  `/grafts/<model>/` optional [prompt grafts](docs/serving.md#prompt-grafts).
- **Checked** on an RTX 3090 and an RTX 5090 with driver 580.159.03.

The container's command chooses what runs:

| command | runs |
|---|---|
| `serve`, `ninfer`, `perplexity`, `calibrate` `[args]` | that binary with any artifact and flags; `serve` listens on `0.0.0.0:8080` unless given `--host` or `--port` |
| `run <model> [profile]` (the default: `run qwen38-27b`) | the [launcher profiles](docs/maintainer/launcher-profiles.md) of `scripts/run.sh` for the official `qwen38-27b` and `qwen36-35b-a3b`, with its `NINFER_*` overrides (`-e NINFER_SPEC=mtp`, `-e NINFER_CONTEXT=131072`, ...) |
| `download <model>` | `scripts/download-model.sh` into `/models`: `qwen38-27b`, `qwen36-27b` or `qwen36-35b-a3b` |

The official Qwen3.8-27B with the measured `tuned` profile, on http://localhost:8080/v1:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all run qwen38-27b
```

[compose.yaml](compose.yaml) wires the GPU, the port and the volumes for the same:
`docker compose run --rm ninfer download qwen38-27b`, then `docker compose up -d`.
`NINFER_IMAGE_ARCH=sm86|sm120a` overrides the GPU detection, and `docker build -t ninfer .` builds
the image from source (`--build-arg ARCHS=86` for one architecture).

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices. The Qwen3.8-Flash-Next artifacts carry the Qwen
Community License 1.0 of their model. Weights you convert yourself carry the licence of the release
they came from.
