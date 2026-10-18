#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/home/agent/comfyui/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
BIN=/home/agent/work-rom/ninfer-t10-perf/build-sm75/apps/ninfer-serve
ART=/mnt/disk238g/ninfer/out/Qwen3.8-27B-Q2Lynn-Q8MTP-DFlash2Q4.ninfer
OUT=/home/agent/work-rom/ninfer-t10-perf/.perf-out
PORT=18093
run() {
  local tag="$1"; shift
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done
  sleep 3
  "$BIN" "$ART" --host 127.0.0.1 --port $PORT --api-key sk-cno239cnl20sb41mfg5f \
    --max-context 2048 --kv-capacity 2048 --kv-dtype bf16 --prefill-chunk 1024 \
    --host-kv-mib 512 --no-cuda-graph "$@" > "$OUT/eq_$tag.log" 2>&1 &
  local spid=$!
  for i in $(seq 1 40); do grep -q "engine ready\|FATAL" "$OUT/eq_$tag.log" && break; sleep 3; done
  if grep -q FATAL "$OUT/eq_$tag.log"; then echo "$tag FATAL"; return; fi
  python3 "$OUT/spec_probe.py" $PORT 512 2>&1 | grep DECODE | sed "s/^/$tag /"
  kill $spid 2>/dev/null; sleep 3
}
run mtp4   --spec mtp --draft-tokens 4
run mtp4b  --spec mtp --draft-tokens 4
run df2_4  --spec dflash2 --draft-tokens 4
run df2_4b --spec dflash2 --draft-tokens 4
echo ALLDONE
