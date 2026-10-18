#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/home/agent/comfyui/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
BIN=/home/agent/work-rom/ninfer-t10-perf/build-sm75/apps/ninfer-serve
ART=/mnt/disk238g/ninfer_model/Qwen3.8-27B/Qwen3.8-27B-Q2Lynn-Q8MTP-DFlash2Q4.ninfer
OUT=/home/agent/work-rom/ninfer-t10-perf/.perf-out
PORT=18093

run_arm() {
  local tag="$1"; shift
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done
  sleep 4
  "$BIN" "$ART" --host 127.0.0.1 --port $PORT --api-key sk-cno239cnl20sb41mfg5f \
    --max-context 4096 --kv-capacity 4096 --kv-dtype bf16 --prefill-chunk 1024 \
    --no-cuda-graph --host-kv-mib 1024 "$@" > "$OUT/spec_$tag.log" 2>&1 &
  local spid=$!
  for i in $(seq 1 40); do grep -q "engine ready\|FATAL" "$OUT/spec_$tag.log" && break; sleep 3; done
  if grep -q FATAL "$OUT/spec_$tag.log"; then
    echo "$tag: FATAL $(grep FATAL "$OUT/spec_$tag.log" | head -1)"; return
  fi
  echo "=== $tag ==="
  python3 "$OUT/spec_probe.py" $PORT 128 2>&1 | grep -E "DECODE|Error" || true
  kill $spid 2>/dev/null
  sleep 3
}

sweep() {
  run_arm mtp2    --spec mtp --draft-tokens 2
  run_arm mtp3    --spec mtp --draft-tokens 3
  run_arm df2_3   --spec dflash2 --draft-tokens 3
  run_arm df2_4   --spec dflash2 --draft-tokens 4
  run_arm df2_5   --spec dflash2 --draft-tokens 5
  run_arm df2_6   --spec dflash2 --draft-tokens 6
}
sweep
echo ALLDONE
