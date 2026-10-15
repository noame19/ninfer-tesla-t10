#!/bin/bash
# ppl.sh <binary> <tag> [extra perplexity args...]
#
# The quality gate. Baseline for the acceptance artifact:
#   3.01132 with the device profile off, 3.01334 with the shipped T10 profile on,
#   8,190 scored tokens (--text, --context 4096, --disjoint).
#
# A route change that only moves data (tile splits, warp counts, launch shapes) must return exactly
# 3.01334. Anything else is a numerics bug, not a trade.
set -u
. "$(dirname "$0")/env.sh"
BIN=$1; TAG=$2; shift 2
OUT="$OUT_DIR/ppl_$TAG.out"

[ -x "$BIN" ] || { echo "$TAG NOBINARY $BIN" >&2; exit 1; }
free_gpu || exit 1

{
  echo "##### $TAG  ($BIN)"
  NINFER_DEVICE_PROFILES="$PROFILE" "$BIN" "$MODEL" --text "$PPL_TEXT" --context 4096 --disjoint \
    --kv-dtype rk4v4-e8 "$@" 2>&1 | grep -iE "perplexity|scored|ppl|tokens|ERROR|FATAL" | tail -12
} | tee "$OUT"
free_gpu
