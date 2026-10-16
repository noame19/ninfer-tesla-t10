#!/bin/bash
# kvmem_serve.sh [--small|--long] -- start ninfer-serve with KVMem on, for the recall probe.
#
#   --small  32k context,  8k window  (~90 s per probe, for iteration)
#   --long  242k context, 53k window  (the user's own settings)
#
# Flags are the user's, except the port, the binary and the log path.
set -u
. "$(dirname "$0")/../env.sh"
SMALL=0
[ "${1:-}" = "--small" ] && SMALL=1
if [ "$SMALL" = 1 ]; then
  MAX_CONTEXT=32768;  KV_BUDGET=8192;  GEN_RESERVE=2048;  HOST_MIB=2048; TAG=small
else
  MAX_CONTEXT=262144; KV_BUDGET=53248; GEN_RESERVE=30720; HOST_MIB=5120; TAG=long
fi
LOG="$OUT_DIR/kvmem-$TAG.log"
echo "serving $TAG: max-context $MAX_CONTEXT kvmem-budget $KV_BUDGET -> $LOG"
exec env NINFER_DEVICE_PROFILES="$PROFILE" "$KVMEM_BIN" "$MODEL" --model-id q8 \
  --kv-dtype rk4v4-e8 --max-context "$MAX_CONTEXT" --kvmem-budget "$KV_BUDGET" \
  --kvmem-gen-reserve "$GEN_RESERVE" --kvmem-host-mib "$HOST_MIB" --kvmem-sessions 1 \
  --kvmem-verify-transfers --prefill-chunk 1024 \
  --host 127.0.0.1 --port "$KVMEM_PORT" \
  --log-stats-interval-ms 5000 > "$LOG" 2>&1
