#!/bin/bash
cd /home/agent/work-rom/ninfer-t10-perf
export CUBLAS_LIB=/home/agent/comfyui/.venv/lib/python3.11/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$CUBLAS_LIB:$LD_LIBRARY_PATH
BIN=build-sm75/apps/ninfer-serve
MODEL=/home/agent/local_model/model/Qwen3.8-27B/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-Q8MTP.ninfer
run() { local tag=$1; shift
  "$BIN" "$MODEL" --host 127.0.0.1 --port 18091 --api-key k "$@" > .perf-out/u_$tag.log 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do grep -qE "engine ready|listening|FATAL" .perf-out/u_$tag.log && break; sleep 1; done
  local used; used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid," | cut -d, -f2)
  echo "$tag ${used:-NONE}"
  grep -E "FATAL" .perf-out/u_$tag.log | head -2
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}
COMMON="--max-context 262144 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --kvmem-host-mib 5120 --kvmem-sessions 1 --max-concurrency 1 --vision --vision-residency cpu"
run user_base  --kvmem-budget 53248 --kvmem-gen-reserve 30720 $COMMON
run user_900   --kvmem-budget 57344 --kvmem-gen-reserve 32768 $COMMON
echo "=== ALLDONE ==="
