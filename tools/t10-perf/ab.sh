#!/bin/bash
# ab.sh <binary> <reps> <n_predict> <tag> [extra serve args...]
#
# One prefill measurement. The prompt is ab2.py's sentence at 14 tokens per rep, so reps=800 is
# ~11.2k and reps=3400 is ~47.6k. Prints the server's own `req#1 done` line, which carries the
# prefill rate and the time to first token.
#
# Everything lands under $OUT_DIR (see env.sh) -- nothing is written beside the binary or the repo.
set -u
. "$(dirname "$0")/env.sh"
BIN=$1; REPS=$2; NP=$3; TAG=$4; shift 4
LOG="$OUT_DIR/ab_$TAG.log"

[ -x "$BIN" ] || { echo "$TAG NOBINARY $BIN" >&2; exit 1; }
free_gpu || exit 1

NINFER_DEVICE_PROFILES="$PROFILE" "$BIN" "$MODEL" --model-id q8 --max-concurrency 1 \
  --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 \
  --host 127.0.0.1 --port "$PORT" "$@" > "$LOG" 2>&1 &
SRV=$!

# The server announces itself once the weights are resident; loading 15.3 GiB takes ~40 s.
for _ in $(seq 1 150); do grep -qE "listening on|FATAL" "$LOG" && break; sleep 2; done
if ! grep -q "listening on" "$LOG"; then
  echo "$TAG STARTFAIL"
  grep -iE "FATAL|error" "$LOG" | head -3
  kill $SRV 2>/dev/null; exit 1
fi
grep -iE "device profile|calibrat" "$LOG" | head -2

python3 "$AB_DRIVER" "$PORT" "$NP" "$REPS" >/dev/null 2>&1
grep -E "req#1 done" "$LOG" | tail -1 | sed "s/.*INFO  //;s/^/$TAG  /"

kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2
