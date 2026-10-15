# T10 (sm_75) measurement harness

The scripts used to measure and gate every change to this port. They exist because the acceptance
figures quoted in `docs/remaining-sm75-work.md`, `docs/sm75-prefill-bottleneck.md` and
`docs/device-profiles/README.md` all have to be reproducible with one command.

## The two-sided gate

Every route change in this port is accepted or rejected on **speed and quality together**:

```bash
./measure.sh <tag> <serve-binary> [<perplexity-binary>]
```

* **Speed** — prefill tok/s at 11.2k and 47.6k tokens on the T10.
* **Quality** — perplexity on the acceptance artifact. A change that only moves data (tile splits,
  warp counts, launch shapes) must return **exactly 3.01334** over 8,190 scored tokens. A change that
  alters accumulation order or precision must return a *stated and justified* delta instead; the
  documented band for this engine's precision trades is `docs/device-profiles/README.md`.

## Scripts

| Script | Purpose |
|---|---|
| `env.sh` | Shared paths and the `free_gpu` helper. Sourced, not run. |
| `measure.sh` | The gate above: 11.2k + 47.6k + PPL. **Start here.** |
| `ab.sh` | One prefill measurement; prints the server's own `req#1 done` line. |
| `ppl.sh` | The quality half alone. |
| `nsys_pf.sh` | Per-kernel attribution for one prefill. |
| `dram.sh` | GPU performance counters (needs sudo): DRAM vs SM issue vs tensor utilisation. |
| `ab2.py` | The prefill driver. Posts a 14-token sentence repeated `reps` times. |
| `ppl_probe.txt` | The perplexity corpus (8,190 scored tokens at `--context 4096 --disjoint`). |
| `waitfor.sh` | Block until a log shows a pattern, a job dies, or a timeout expires. Replaces `sleep N`. |

## Conventions

* **Prompt length comes from `reps`**, not from a tokenizer: `ab2.py`'s sentence is 14 tokens, so
  `reps=800` is ~11.2k, `reps=3400` is ~47.6k, `reps=7000` is ~98k.
* **`nsys`, never `ncu`, for `mul_mat_q`.** Kernel replay on that kernel hard-froze this host twice.
  `nsys profile --trace=cuda` is enough for attribution, and `dram.sh` gives the counter evidence.
* **Kill by PID, never `pkill -f ninfer-serve`.** The agent shell's own command line contains that
  string, so a pattern kill takes the shell down with it (exit 143). `nvidia-smi
  --query-compute-apps=pid` is the reliable source.
* **Outputs go to `$OUT_DIR`** (default `${TMPDIR:-/tmp}/ninfer-t10-perf`), never beside the binary
  or in the repository.
* **Use `waitfor.sh`, not a fixed `sleep`.** A fixed sleep always costs the full duration and loses
  the result when the job has already exited.

## Overridable environment

| Variable | Default |
|---|---|
| `NINFER_T10_MODEL` | the 27B acceptance artifact (see `env.sh`) |
| `NINFER_T10_PROFILE` | `docs/device-profiles/t10-sm75.json` |
| `NINFER_T10_MAXCTX` | `20480` for the profiling scripts |
| `NINFER_T10_OUT` | `${TMPDIR:-/tmp}/ninfer-t10-perf` |
| `NINFER_T10_PORT` | `18099` |
| `NINFER_CUDA_LIB` | the CUDA wheel's `cu13/lib` the local build links against |

## Baseline figures

Taken on the Tesla T10 with the acceptance artifact, `rk4v4-e8` KV, 56 SMs:

| Workload | tok/s | Notes |
|---|---|---|
| 11.2k prefill | ~467 | TTFT ~24 s |
| 47.6k prefill | ~386 | TTFT ~2 m 3 s |
| 98k prefill | ~299 | TTFT ~5 m 28 s |
| PPL (`--context 4096 --disjoint`) | 3.01334 | 8,190 scored tokens, bit-exact gate |

Re-measure rather than trusting these numbers if the card, driver, or acceptance artifact changed.

## Outputs

`ppl.sh` and the profiling scripts leave their own working files where the engine puts them
(`profiles/`, next to the current directory). They are transient and ignored by
`.gitignore`'s `profiles/` rule; delete them freely.
