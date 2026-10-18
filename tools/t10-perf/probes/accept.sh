#!/bin/bash
# P1 acceptance: (1) 242k real scenario on /completion, (2) chat control, (3) PPL with KVMem OFF.
cd /home/agent/work-rom/ninfer-t10-perf
. tools/t10-perf/env.sh
BIN=$PWD/build-sm75/apps/ninfer-serve
PPL=$PWD/build-sm75/apps/ninfer-perplexity
kill_srv() { local p; p="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|head -1)"; [ -n "${p:-}" ] && kill "$p" 2>/dev/null && sleep 4; }

echo "############ 1) 242k, user's exact flags, /completion ############"
kill_srv; : > .perf-out/acc-long.log
NINFER_DEVICE_PROFILES="$PROFILE" "$BIN" "$MODEL" --model-id q8 --kv-dtype rk4v4-e8 \
  --max-context 262144 --kvmem-budget 53248 --kvmem-gen-reserve 30720 --kvmem-host-mib 5120 \
  --kvmem-sessions 1 --prefill-chunk 1024 --host 127.0.0.1 --port 18083 > .perf-out/acc-long.log 2>&1 &
for _ in $(seq 1 120); do grep -q 'listening on' .perf-out/acc-long.log && break; sleep 5; done
python3 tools/t10-perf/kvmem/kvmem_recall.py --port 18083 --budget 242000 --novel-first \
  --n-predict 96 2>&1 | grep -E 'SCORE|HIT|MISS|WALL'

echo
echo "############ 2) 242k, chat endpoint (control) ############"
kill_srv; : > .perf-out/acc-long2.log
NINFER_DEVICE_PROFILES="$PROFILE" "$BIN" "$MODEL" --model-id q8 --kv-dtype rk4v4-e8 \
  --max-context 262144 --kvmem-budget 53248 --kvmem-gen-reserve 30720 --kvmem-host-mib 5120 \
  --kvmem-sessions 1 --prefill-chunk 1024 --host 127.0.0.1 --port 18083 > .perf-out/acc-long2.log 2>&1 &
for _ in $(seq 1 120); do grep -q 'listening on' .perf-out/acc-long2.log && break; sleep 5; done
python3 tools/t10-perf/kvmem/kvmem_recall.py --port 18083 --budget 242000 --novel-first \
  --endpoint chat --n-predict 96 2>&1 | grep -E 'SCORE|HIT|MISS|WALL'
kill_srv

echo
echo "############ 3) PPL with KVMem OFF (must be bit-exact 3.01334) ############"
NINFER_T10_OUT=$PWD/.perf-out tools/t10-perf/ppl.sh "$PPL" p1-ppl 2>&1 | tail -4
echo ALLDONE
