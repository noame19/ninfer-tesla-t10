# One-off probes

The measurements quoted in `docs/` were taken with these. They are kept because they encode how a
number was obtained — a re-run reproduces the figure — not because they are part of the harness.
Everything reusable lives one directory up.

| Probe | What it measured |
|---|---|
| `sweep_draft.sh` | decode tok/s against `--draft-tokens` 2..6; the source of "2 is optimal on a T10" |
| `sweep_spec.sh` | the same sweep across `--spec mtp` / `--spec dflash2` |
| `width_sweep.sh` | decode tok/s against verify width, used to derive the 39.0 ms + 10.8 ms/column cost model |
| `equalwidth.sh` | MTP against DFlash2 at *equal* verify width, which is the only fair comparison of the two |
| `spec_probe.py` | one decode run of a fixed code prompt, printing tok/s and acceptance |
| `accept.sh` | acceptance counters only, no timing |
| `vram_cal2.sh` | the VRAM ladder: 14,012 MiB at 16k through 15,412 at 96k, and the KVMem points |
| `vram_user.sh` | VRAM under the user's own KVMem budget pair (53248/30720) |

They assume the environment `../env.sh` exports and take the server binary as an argument; see
`../README.md`.
