# .local-build — building sm_75 on the host instead of in CI

`../.github/workflows/build-sm75.yml` builds on a GitHub runner and hands back ~1.7 GB of
artifacts per iteration. This directory does the same build locally, for about 0.35 GB of
one-time downloads and no traffic afterwards.

## Why it is not just the workflow

The workflow installs `cuda-toolkit-13-4`: 3.5 GB downloaded, 7.1 GB installed. This tree reads
only a slice of it, so the image installs `cuda-nvcc-13-4` (nvcc, cicc, libnvvm, ptxas, cudart
headers, cccl's cub/thrust, `cuda.h`) plus `cuda-nvtx-13-4` (`nvtx3/nvToolsExt.h`) — 94 MB — and
borrows cublas from the CUDA 13 wheel already present on the host, which avoids `libcublas-dev`'s
837 MB. Everything else (Debian 12 base, gcc-12, CMake 3.31.6 static, the apt list, the configure
flags) is the workflow's.

The host must therefore have a CUDA 13 wheel on disk. `build.sh` defaults to the one in the
comfyui venv and takes `NINFER_NVIDIA_PREFIX` for any other prefix holding `cu13/include` and
`cu13/lib`.

## Use

```
docker build -t ninfer-sm75-build .local-build
.local-build/build.sh scripts/build-native.sh configure
.local-build/build.sh scripts/build-native.sh target ninfer-serve ninfer-perplexity
```

`NINFER_JOBS` (default: every core), `NINFER_BENCHMARKS` (default OFF — the benchmark rig doubles
the build), `NINFER_CUDA_ARCH`, `CCACHE_MAXSIZE`, `NINFER_BUILD_IMAGE`, `NINFER_BUILD_ROOT` and
`NINFER_CCACHE_DIR` are overridable from the environment.
The build tree and ccache live in `/home/agent/work-rom/ninfer-build/` (`build-sm75/` and `ccache/`)
and are reached through the `build-sm75` and `.ccache` symlinks in the repository root. Both are
gitignored. They were briefly on `/mnt/disk238g` because the root filesystem was full; they are
back on the root filesystem now that it has room, because a build across two mounts is slower and
loses the ccache's hardlink advantage. `NINFER_BUILD_ROOT` moves them again.

The container runs as the invoking user, so the build tree, the ccache and the binaries come out
owned by that user rather than by root.

`ninfer-serve` links to 1.3 GB (843 MB of it is the `.nv_fatbin` section holding every kernel for
every architecture in the build), so installing it is a slow copy rather than an atomic rename if
`~/.local/bin` is on another filesystem. Run it in the background, or the shell's own timeout will
cut the copy short and leave a partial binary behind:

```
setsid nohup bash -c 'cp -f build-sm75/apps/ninfer-serve ~/.local/bin/ninfer-serve' \
  < /dev/null > /tmp/install.log 2>&1 &
```

## `toolchain-smoke.cu`

One translation unit that includes every third-party CUDA header this tree uses — cub, `cuda/atomic`,
`cuda_fp4.h`, `cuda_fp8.h`, `cuda_bf16.h`, `cublas_v2.h`, `cublasLt.h`, `nvtx3/nvToolsExt.h`,
`cuda_pipeline.h` — and links cublas. It is the check that the borrowed cublas is wired correctly:

```
.local-build/build.sh bash -c 'cd /src/.local-build && nvcc -std=c++17 -arch=sm_75 toolchain-smoke.cu -o /tmp/smoke -lcublas'
```

The `libcublas.so.13` / `libcublasLt.so.13` symlinks in the image matter here: the linker resolves
`-lcublas` through `libcublas.so`, but that library records a `DT_NEEDED` on `libcublasLt.so.13` and
the linker looks the soname up by name.
