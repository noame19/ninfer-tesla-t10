#!/bin/bash
# bench_models.sh <binary> <label> [models...]
#
# One baseline row per artifact: load, 11.2k prefill, decode, a deterministic sample, and PPL where
# the corpus is meaningful. Every model is measured with the same binary in one run so the numbers
# are comparable with each other, and the samples are recorded so a future build can be checked for
# a silent quality change.
#
# The sampling settings come from the operator's llama.cpp preset at ~/local_model/llama.ini, so a
# figure here can be read against what that runtime produces for the same model: thinking stays on
# (`reasoning = auto`), temp 0.95 / top-p 0.95 / top-k 20 / min-p 0, and the MTP towers draft two
# tokens per round (`spec-type = draft-mtp`, `spec-draft-n-max = 2`). The preset's repeat-penalty 1.1
# has no counterpart here: NInfer refuses any value but the neutral 1.0.
#
# The sample is taken twice per model: once as the operator runs it, and once at temperature 0 so
# the text is reproducible. Only the second is a regression reference -- the first is there to show
# what the model actually answers under the preset.
set -u
. "$(dirname "$0")/env.sh"
BIN=$1; LABEL=$2; shift 2

PPL_BIN="${NINFER_T10_PPL_BIN:-$(dirname "$BIN")/ninfer-perplexity}"
PPL_TEXT="${NINFER_T10_PPL_TEXT:-$T10_PERF_DIR/ppl_probe.txt}"

# The preset. Fields NInfer does not have are dropped; the rest map one to one.
TEMP=${NINFER_T10_TEMP:-0.95}
TOP_P=${NINFER_T10_TOP_P:-0.95}
TOP_K=${NINFER_T10_TOP_K:-20}
MIN_P=${NINFER_T10_MIN_P:-0.0}
DRAFT=${NINFER_T10_DRAFT_TOKENS:-2}

if [ "$#" -eq 0 ]; then
  set -- \
    "$MODELS/Qwen3.8-27B/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q3_K_XL.ninfer" \
    "$MODELS/QwenPaw-Flash-9B/QwenPaw-Flash-9B-heretic-MTP-Q6_K.ninfer" \
    "$MODELS/Ornith-1.5-9B/Ornith-1.5-9B-MTP-Q6_K.ninfer" \
    "$MODELS/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M-MTP.ninfer" \
    "$MODELS/Qwen3.5-0.8B/Qwen3.5-0.8B.ninfer"
fi

[ -x "$BIN" ] || { echo "$LABEL NOBINARY $BIN" >&2; exit 1; }

# Start a server for one artifact and report whether it came up. Every model needs its own process
# because the weights are bound at startup.
start_server() {
  local model=$1 tag=$2
  shift 2
  local log="$OUT_DIR/bench_$tag.log"
  free_gpu || return 1
  NINFER_DEVICE_PROFILES="$PROFILE" "$BIN" "$model" --model-id "$tag" --max-concurrency 1 \
    --max-context 16384 --kv-capacity 16384 --kv-dtype rk4v4-e8 \
    --host 127.0.0.1 --port "$PORT" "$@" > "$log" 2>&1 &
  echo $! > "$OUT_DIR/bench_$tag.pid"
  for _ in $(seq 1 180); do
    grep -qE "listening on|FATAL|failed" "$log" && break
    sleep 2
  done
  if ! grep -q "listening on" "$log"; then
    echo "  STARTFAIL"
    grep -iE "FATAL|failed|error" "$log" | head -3 | sed 's/^/  ! /'
    return 1
  fi
  return 0
}

stop_server() {
  local tag=$1 pid
  pid=$(cat "$OUT_DIR/bench_$tag.pid" 2>/dev/null)
  [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null
  [ -n "${pid:-}" ] && wait "$pid" 2>/dev/null
  sleep 2
}

for model in "$@"; do
  name=$(basename "$model" .ninfer)
  tag="$LABEL-${name:0:24}"
  echo "===== $name"
  echo "  artifact $(du -m --apparent-size "$model" | cut -f1) MiB"

  # Does this artifact carry an MTP component? Only then can speculative decoding be selected.
  has_mtp=$(python3 - "$model" <<'PY'
import sys
sys.path.insert(0, '.')
from tools.artifact.reader import Artifact
try:
    print("yes" if "mtp" in Artifact(sys.argv[1]).directory.components else "no")
except Exception:
    print("no")
PY
)
  spec=()
  [ "$has_mtp" = yes ] && spec=(--spec mtp --draft-tokens "$DRAFT")
  echo "  mtp=$has_mtp draft=${DRAFT}"

  if ! start_server "$model" "$tag" "${spec[@]}"; then
    echo "  RESULT start=fail"
    continue
  fi

  python3 "$AB_DRIVER" "$PORT" 16 800 >/dev/null 2>&1
  grep -E "req#1 done" "$OUT_DIR/bench_$tag.log" | tail -1 |
    sed 's/.*INFO  //;s/^/  PREFILL /'

  python3 "$T10_PERF_DIR/ab3.py" "$PORT" 128 64 2>/dev/null

  # The operator's own settings: thinking on, the preset's sampling. Not reproducible, recorded so
  # the answers can be eyeballed for a load-path regression.
  python3 "$T10_PERF_DIR/sample.py" "$PORT" 1024 \
    --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" 2>/dev/null |
    sed 's/^/  PRESET/'

  # Temperature 0 with a thinking budget, which is the reproducible reference.
  python3 "$T10_PERF_DIR/sample.py" "$PORT" 1024 --temp 0 --top-k 1 2>/dev/null |
    sed 's/^/  GREEDY/'

  stop_server "$tag"

  if [ -x "$PPL_BIN" ]; then
    NINFER_DEVICE_PROFILES="$PROFILE" "$PPL_BIN" "$model" --text "$PPL_TEXT" --context 4096 \
      --disjoint --kv-dtype rk4v4-e8 2>&1 |
      grep -iE "^ *custom |ERROR|FATAL" | tail -1 | sed 's/^/  PPL  /'
  fi
  free_gpu
done
echo "ALLDONE $LABEL"
