#!/bin/bash
# nsys_pf.sh <binary> <reps> <tag> [extra serve args...]
#
# Kernel attribution for one prefill. Uses nsys, never ncu: kernel replay on mul_mat_q
# hard-froze this host twice, and `nsys profile` is enough to separate MMQ from attention from GDN.
#
# Set NINFER_T10_MAXCTX to attribute a longer prefill (the KV pool is allocated at startup, so it
# must be at least as large as the prefill being measured).
#
# Writes $OUT_DIR/nsys_<tag>.nsys-rep and prints the per-kernel summary.
set -u
. "$(dirname "$0")/env.sh"
BIN=$1; REPS=$2; TAG=$3; shift 3
LOG="$OUT_DIR/nsyspf_$TAG.log"
REP="$OUT_DIR/nsys_$TAG"
MAXCTX="${NINFER_T10_MAXCTX:-20480}"

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

python3 "$AB_DRIVER" "$PORT" 16 "$REPS" >/dev/null 2>&1
grep -E "req#1 done" "$LOG" | tail -1 | sed "s/.*INFO  //;s/^/$TAG  /"

# Stop the server by the PID the driver reports, not with `pkill -f "$BIN"`: the pattern also
# matches this script's own command line and the agent shell that launched it.
SRVPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
[ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
wait $NSYS 2>/dev/null

echo "--- report: $REP.nsys-rep ---"
nsys stats --report cuda_gpu_kern_sum --format table --force-export=true "$REP.nsys-rep" 2>/dev/null | head -30
free_gpu
