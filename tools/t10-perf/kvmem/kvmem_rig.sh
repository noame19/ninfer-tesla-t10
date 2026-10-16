#!/bin/bash
# kvmem_rig.sh -- start the small KVMem rig and run a multi-turn recall probe against it.
#
# The small rig exists so KVMem behaviour can be tested in ~3 minutes instead of the ~25 minutes a
# 200k-token turn 1 costs on this card:
#
#   max-context 32768  window (B) 8192   reserve (R) 2048   host 2048 MiB
#
# Turn 1 fills 26k tokens, so ~18k of it must be evicted; a probe that only the retrieval path can
# bring back is therefore a real test, at a 26k-token prompt instead of a 242k-token one.
#
#   kvmem_rig.sh start              # (re)start the small server, wait until it listens
#   kvmem_rig.sh run <turn1> [args] # run kvmem_multiturn.py against the running server
#   kvmem_rig.sh all   <turn1>      # start, then run
#
# It kills the process holding the GPU only when told to start; it never uses `pkill -f ninfer-serve`
# (that pattern also matches the invoking shell and kills it).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../env.sh"

stop_server() {
  local pid
  pid="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1)"
  if [ -n "${pid:-}" ]; then kill "$pid" 2>/dev/null && echo "stopped pid $pid"; sleep 4; fi
}

start_server() {
  stop_server
  echo "starting the small rig (32768 context, 8192 window) -> $OUT_DIR/kvmem-small.log"
  : > "$OUT_DIR/kvmem-small.log"   # truncate, else the wait below matches the PREVIOUS run's line
  "$HERE/kvmem_serve.sh" --small > "$OUT_DIR/kvmem-small.serve.log" 2>&1 &
  local waited=0
  while [ "$waited" -lt 180 ]; do
    grep -q 'listening on' "$OUT_DIR/kvmem-small.log" 2>/dev/null && { echo "listening after ${waited}s"; sleep 3; return 0; }
    sleep 5; waited=$((waited + 5))
  done
  echo "server did not start in 180s; last lines:"; tail -5 "$OUT_DIR/kvmem-small.log"; return 1
}

case "${1:-all}" in
  start) start_server ;;
  run)   shift; exec python3 "$HERE/kvmem_multiturn.py" "$@" ;;
  all)   shift; start_server || exit 1
         exec python3 "$HERE/kvmem_multiturn.py" "$@" ;;
  *)     echo "usage: $0 [start|run|all] [turn1-tokens] [more args]" >&2; exit 2 ;;
esac
