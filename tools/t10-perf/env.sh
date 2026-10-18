# Shared environment for the T10 (sm_75) measurement harness.
#
# Source this from the other scripts in this directory. Every path is either derived from the
# repository root or overridable by an environment variable, so the harness works from a clone in
# any location and never hard-codes one person's home directory.
#
# The one exception is MODEL: the acceptance artifact is a local file outside the repository and
# has no repository-relative location. Override NINFER_T10_MODEL to measure something else.

T10_PERF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$T10_PERF_DIR/../.." && pwd)"

# The acceptance artifact. Every speed and quality figure quoted in docs/ was taken on this file.
# Converted artifacts all live under one directory, one subdirectory per tower.
MODELS="${NINFER_T10_MODELS:-/mnt/disk238g/ninfer_model}"
MODEL="${NINFER_T10_MODEL:-$MODELS/Qwen3.8-27B/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-Q8MTP.ninfer}"

# The hand-written route profile for this card. Editing it is a zero-compile experiment.
PROFILE="${NINFER_T10_PROFILE:-$REPO_ROOT/docs/device-profiles/t10-sm75.json}"

# The perplexity corpus. 8,190 scored tokens at --context 4096 --disjoint.
PPL_TEXT="${NINFER_T10_PPL_TEXT:-$T10_PERF_DIR/ppl_probe.txt}"

# The prefill driver: posts a 14-token-per-rep sentence, so reps=800 is ~11.2k and 3400 is ~47.6k.
AB_DRIVER="${NINFER_T10_AB_DRIVER:-$T10_PERF_DIR/ab2.py}"

# Logs, profile captures and the generated filler land here and nowhere else. Transient by
# construction, and defined before the corpus paths that default into it.
OUT_DIR="${NINFER_T10_OUT:-${TMPDIR:-/tmp}/ninfer-t10-perf}"
mkdir -p "$OUT_DIR"

# The long-context recall probe runs against the installed binary: KVMem is part of the main line.
KVMEM_BIN="${NINFER_T10_KVMEM_BIN:-$HOME/.local/bin/ninfer-serve}"
KVMEM_PORT="${NINFER_T10_KVMEM_PORT:-18083}"

# The long-context probe corpus. The filler is generated from the repository's own documentation
# (make_corpus.py) so it is never committed; the narrative half is a published novel and is kept
# outside the tree on purpose. Point these anywhere else to override.
RECALL_FILLER="${NINFER_T10_RECALL_FILLER:-$OUT_DIR/filler.txt}"
RECALL_NOVEL="${NINFER_T10_RECALL_NOVEL:-$HOME/work-rom/corpus/novel.txt}"

PORT="${NINFER_T10_PORT:-18099}"

# The locally built binary links libcublas/libcublasLt from the CUDA wheel this host already has
# (see .local-build/Dockerfile). Runtime needs the same directory on the loader path.
CUBLAS_LIB="${NINFER_CUDA_LIB:-/home/agent/comfyui/.venv/lib/python3.11/site-packages/nvidia/cu13/lib}"
export LD_LIBRARY_PATH="$CUBLAS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Wait until the device is ours to use. A leftover serve process of ours is killed; anything else
# is LEFT ALONE -- the card may be someone else's -- but a small resident process does not block a
# measurement. What matters is that no ninfer process is running and that enough memory is free.
# The floor is generous on purpose: the acceptance artifact needs ~13.2 GiB of the 16 GiB.
free_gpu() {
  local floor_mib=${NINFER_T10_FREE_MIB:-14000}
  for _ in $(seq 1 45); do
    local busy=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      if grep -qa ninfer "/proc/$p/cmdline" 2>/dev/null; then
        kill "$p" 2>/dev/null
        busy=1
      fi
    done
    if [ "$busy" = 0 ]; then
      local free_mib
      free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free_mib" -ge "$floor_mib" ] && return 0
    fi
    sleep 2
  done
  echo "free_gpu: still busy: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')" >&2
  echo "free_gpu: free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader), need ${floor_mib} MiB" >&2
  return 1
}

# Kill by PID from nvidia-smi, never by `pkill -f ninfer-serve`: the agent's own shell command line
# contains that string, so a pattern kill takes the shell down with it (exit 143).
