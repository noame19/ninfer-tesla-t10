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
# The build tree and ccache live outside the repository by default (NINFER_BUILD_ROOT, see below),
# reached through the symlinks build-sm75/ and .ccache/. A bind mount does not follow a symlink, so
# whatever they point at has to be mounted too. When they point inside $repo this is a no-op.
build_root="${NINFER_BUILD_ROOT:-/home/agent/work-rom/ninfer-build}"

# The image has no git, so cmake/GenerateBuildId.cmake cannot derive the id there and every build
# stamps '+unknown' -- a binary that cannot say which commit it came from. Ask the host instead.
#
# The id is the exact tag when HEAD is one, otherwise the short commit hash. Not `git describe
# --tags`: this fork's newest tag is v0.6.0-rtx3090 while VERSION says 0.12.0, so describe would
# print 'v0.6.0-rtx3090-1744-g<hash>' and the log line would contradict itself. VERSION already
# says which release this is; the id only has to say which commit.
if [ -n "${NINFER_BUILD_ID:-}" ]; then
  build_id="$NINFER_BUILD_ID"
elif git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
  build_id="$(git -C "$repo" describe --tags --exact-match 2>/dev/null ||
              git -C "$repo" rev-parse --short HEAD 2>/dev/null ||
              printf 'unknown')"
  # Same scope as NINFER_PRODUCT_ROOTS in cmake/GenerateBuildId.cmake: only files that are compiled
  # into the binaries can make the id dirty.
  if [ -n "$(git -C "$repo" status --porcelain --untracked-files=normal -- \
                       src include apps cmake third_party CMakeLists.txt CMakePresets.json)" ]; then
    build_id="${build_id}-dirty"
  fi
else
  build_id="unknown"
fi

if [ ! -d "$nvidia_prefix/cu13/include" ]; then
  printf 'no cublas headers at %s/cu13/include -- set NINFER_NVIDIA_PREFIX\n' "$nvidia_prefix" >&2
  exit 2
fi
mkdir -p "$ccache_dir"

exec docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  -v "$repo:/src" \
  -v "$ccache_dir:/ccache" \
  -v "$build_root:$build_root" \
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
  -e NINFER_BUILD_ID="$build_id" \
  "$image" "$@"
