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

`NINFER_JOBS` (default 4), `NINFER_BENCHMARKS` (default OFF — the benchmark rig doubles the build),
`NINFER_CUDA_ARCH`, `CCACHE_MAXSIZE` and `NINFER_BUILD_IMAGE` are overridable from the environment.
The build tree is `build-sm75/` in the repository root and ccache lives in `.ccache/`; both are
gitignored.

The container runs as the invoking user, so the build tree, the ccache and the binaries come out
owned by that user rather than by root.

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
