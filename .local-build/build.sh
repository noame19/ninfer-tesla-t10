#!/usr/bin/env bash
# Run a command in the local sm_75 build container. Everything the build needs is in the image;
# this script supplies the two host directories it cannot carry: the source tree and the CUDA 13
# cublas that is borrowed instead of downloaded.
#
#   .local-build/build.sh scripts/build-native.sh configure
#   .local-build/build.sh scripts/build-native.sh target ninfer-serve ninfer-perplexity
#   .local-build/build.sh bash -lc 'nvcc --version && cmake --version'
#
# Overridable: NINFER_NVIDIA_PREFIX, NINFER_CCACHE_DIR, NINFER_JOBS, NINFER_BENCHMARKS,
# NINFER_BUILD_IMAGE, CCACHE_MAXSIZE.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${NINFER_BUILD_IMAGE:-ninfer-sm75-build}"

# The CUDA 13 runtime+headers the host already has, so libcublas-dev (837 MB) is not downloaded.
# Point NINFER_NVIDIA_PREFIX at any prefix holding cu13/include and cu13/lib.
nvidia_prefix="${NINFER_NVIDIA_PREFIX:-/home/agent/comfyui/.venv/lib/python3.11/site-packages/nvidia}"
ccache_dir="${NINFER_CCACHE_DIR:-$repo/.ccache}"

if [ ! -d "$nvidia_prefix/cu13/include" ]; then
  printf 'no cublas headers at %s/cu13/include -- set NINFER_NVIDIA_PREFIX\n' "$nvidia_prefix" >&2
  exit 2
fi
mkdir -p "$ccache_dir"

exec docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  -v "$repo:/src" \
  -v "$ccache_dir:/ccache" \
  -v "$nvidia_prefix:/opt/nvidia:ro" \
  -e HOME=/tmp \
  -e NINFER_CUDA_ARCH="${NINFER_CUDA_ARCH:-75}" \
  -e NINFER_BUILD_DIR="${NINFER_BUILD_DIR:-/src/build-sm75}" \
  -e NINFER_TESTS="${NINFER_TESTS:-OFF}" \
  -e NINFER_MULTICALL="${NINFER_MULTICALL:-OFF}" \
  -e NINFER_BENCHMARKS="${NINFER_BENCHMARKS:-OFF}" \
  -e NINFER_JOBS="${NINFER_JOBS:-$(nproc)}" \
  -e CCACHE_DIR=/ccache \
  -e CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-3G}" \
  -e CCACHE_COMPILERCHECK=content \
  "$image" "$@"
