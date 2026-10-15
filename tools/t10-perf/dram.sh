#!/bin/bash
# dram.sh <binary> <reps> <tag> [extra serve args...]
#
# GPU-metrics profile of one prefill: samples the card's performance counters so a traffic-bound
# kernel (high DRAM utilisation) is told apart from a stall-bound one (low utilisation). This is how
# the T10 prefill was established to be stall-bound rather than bandwidth-bound.
#
# Needs the performance counters, so it runs under sudo. Reads on sudo are query-only.
# Writes $OUT_DIR/dram_<tag>.nsys-rep — the raw samples are the GPU_METRICS table of the .sqlite
# beside it (metricId 22 = DRAM read, 23 = write, 9 = tensor, 8 = SM issue, 17/20 = warps).
set -u
. "$(dirname "$0")/env.sh"
BIN=$1; REPS=$2; TAG=$3; shift 3
LOG="$OUT_DIR/dram_$TAG.log"
REP="$OUT_DIR/dram_$TAG"
MAXCTX="${NINFER_T10_MAXCTX:-20480}"

[ -x "$BIN" ] || { echo "$TAG NOBINARY $BIN" >&2; exit 1; }
free_gpu || exit 1

sudo -n env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" NINFER_DEVICE_PROFILES="$PROFILE" \
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  --gpu-metrics-devices=0 --gpu-metrics-set=tu10x --gpu-metrics-frequency=2000 \
  --output "$REP" \
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

SRVPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
[ -n "$SRVPID" ] && sudo -n kill "$SRVPID" 2>/dev/null
wait $NSYS 2>/dev/null

echo "--- report: $REP.nsys-rep (samples in $REP.sqlite) ---"
nsys stats --report gpu_metrics_device --format csv "$REP.nsys-rep" 2>/dev/null | head -2
free_gpu
