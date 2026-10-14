# Device route profiles (checked-in)

`t10-sm75.json` is a hand-written route profile for the NVIDIA Tesla T10 (Turing, sm_75). Install it
where the engine reads profiles, or point `NINFER_DEVICE_PROFILES` at it:

```bash
mkdir -p ~/.cache/ninfer && cp docs/device-profiles/t10-sm75.json ~/.cache/ninfer/device-profiles.json
```

Two settings, both measured on the card:

* `attn_i8_small/h24/rk4v4-e8/w1..w8` — the attention launch tier per implementation window, so the
  chunked small-T route uses the tier this card measures best instead of the compiled default.
* `gdn_two_stage/h48 on` — the two-stage GDN prefill route; an unprofiled card uses the
  WY/state-passing pipeline.

Measured on the T10, 47,600-token prefill, `rk4v4-e8`, 56 SMs: prefill 339.8 -> 354.3 tok/s (+4.3%),
decode unchanged (17.5 -> 17.7). Isolated at that shape: tiers alone +2.0%, tiers + two-stage GDN
+4.3%, `attn_pv_f16` +0.0% (so it is left off — it is the only member that lowers precision and it
buys nothing here). Quality through `ninfer-perplexity --text` (8,190 scored tokens): PPL 3.01132 ->
3.01334 (+0.067%), inside the band this engine's documented precision trades occupy
(`--lm-head-q6` +0.01%, `--embedding-q4` +0.30%).

`calibration.route_catalog` is this build's `calibration_route_catalog_digest()`; a profile whose
digest differs is refused with a line naming the correct one, so a stale copy degrades to "no
profile" rather than to a wrong route. Re-derive it after any change to the route catalog.

Do **not** add `attn_parallel_tiles`: with it on, a request fails with `cudaErrorIllegalAddress` on
this port (measured), so that route is not usable on sm_75.

`docs/device-profiles.md` describes the mechanism and the built-in profiles.
