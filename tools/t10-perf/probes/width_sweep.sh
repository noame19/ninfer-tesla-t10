#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/home/agent/comfyui/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
BIN=/home/agent/work-rom/ninfer-t10-perf/build-sm75/apps/ninfer-serve
ART=/mnt/disk238g/ninfer_model/Qwen3.8-27B/Qwen3.8-27B-Q2Lynn-Q8MTP-DFlash2Q4.ninfer
OUT=/home/agent/work-rom/ninfer-t10-perf/.perf-out
PORT=18093
for n in 1 2 3 4 5 6 7 8; do
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done
  sleep 3
  "$BIN" "$ART" --host 127.0.0.1 --port $PORT --api-key sk-cno239cnl20sb41mfg5f \
    --max-context 2048 --kv-capacity 2048 --kv-dtype bf16 --prefill-chunk 1024 \
    --host-kv-mib 512 --spec mtp --draft-tokens $n --no-cuda-graph > "$OUT/w$n.log" 2>&1 &
  spid=$!
  for i in $(seq 1 40); do grep -q "engine ready\|FATAL" "$OUT/w$n.log" && break; sleep 3; done
  grep -q FATAL "$OUT/w$n.log" && { echo "mtp$n FATAL"; continue; }
  python3 "$OUT/spec_probe.py" $PORT 512 2>&1 | grep -E "DECODE" | sed "s/^/mtp$n /"
  kill $spid 2>/dev/null; sleep 3
done
echo ALLDONE
