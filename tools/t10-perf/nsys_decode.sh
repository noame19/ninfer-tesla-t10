#!/bin/bash
# nsys_decode.sh <binary> <n_predict> <tag> [extra serve args...]
#
# Kernel attribution for one decode run: a tiny prompt, then <n_predict> generated tokens, with
# whatever speculative flags the caller passes. The prefill is made negligible on purpose so the
# summary is the per-token cost.
#
# The point of measuring decode separately: `req#1 done` reports prefill and decode tok/s as two
# numbers, and a change that helps one need not help the other. On this port the draft head is a
# candidate for decode only.
#
# Set NINFER_T10_PROMPT to change the prompt (default: a 40-token request for a long answer).
set -u
. "$(dirname "$0")/env.sh"
BIN=$1; NP=$2; TAG=$3; shift 3
LOG="$OUT_DIR/nsysdec_$TAG.log"
REP="$OUT_DIR/nsysdec_$TAG"
MAXCTX="${NINFER_T10_MAXCTX:-8192}"

[ -x "$BIN" ] || { echo "$TAG NOBINARY $BIN" >&2; exit 1; }
free_gpu || exit 1

nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  --output "$REP" \
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" NINFER_DEVICE_PROFILES="$PROFILE" \
      "$BIN" "$MODEL" --model-id q8 --max-concurrency 1 \
      --max-context "$MAXCTX" --kv-capacity "$MAXCTX" --kv-dtype rk4v4-e8 \
      --host 127.0.0.1 --port "$PORT" "$@" > "$LOG" 2>&1 &
NSYS=$!

for _ in $(seq 1 150); do grep -qE "listening on|FATAL" "$LOG" && break; sleep 2; done
if ! grep -q "listening on" "$LOG"; then
  echo "$TAG STARTFAIL"; grep -iE "FATAL|error" "$LOG" | head -3; kill $NSYS 2>/dev/null; exit 1
fi

python3 - "$PORT" "$NP" <<'PY'
import json, sys, urllib.request
port, npred = sys.argv[1], int(sys.argv[2])
body = json.dumps({
    "prompt": "Count upward from 1 and keep going. Reply with the numbers only.",
    "n_predict": npred, "temperature": 0, "cache_prompt": False,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as r:  # noqa: S310 (localhost only)
    json.load(r)
PY
grep -E "req#1 done" "$LOG" | tail -1 | sed "s/.*INFO  //;s/^/$TAG  /"

SRVPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
[ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
wait $NSYS 2>/dev/null

echo "--- report: $REP.nsys-rep ---"
nsys stats --report cuda_gpu_kern_sum --format table --force-export=true "$REP.nsys-rep" 2>/dev/null | head -40
free_gpu
