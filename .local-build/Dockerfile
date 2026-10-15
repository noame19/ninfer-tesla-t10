# syntax=docker/dockerfile:1
#
# Local sm_75 build environment for the Tesla T10, for building on the host instead of in CI.
#
# The recipe is the one .github/workflows/build-sm75.yml uses, with one deliberate difference:
# the workflow installs cuda-toolkit-13-4 (3.5 GB of downloads, most of it never read by this
# tree), while this image installs only the slice a compile actually needs --
#
#   cuda-nvcc-13-4   nvcc, cicc, libnvvm, ptxas, cudart headers, cccl's cub/thrust, cuda.h
#   cuda-nvtx-13-4   nvtx3/nvToolsExt.h
#
# and borrows cublas from the CUDA 13 wheel already installed on the host (see build.sh), which
# avoids libcublas-dev's 837 MB. Together that is ~0.35 GB instead of 3.5 GB.
#
# Build:
#   docker build -t ninfer-sm75-build .local-build
#
# Run through .local-build/build.sh, which supplies the source and the cublas prefix.

FROM debian:12

ENV DEBIAN_FRONTEND=noninteractive

# The workflow's apt list, minus git (the source is bind-mounted, not cloned) and gnupg (the CUDA
# keyring is unpacked with dpkg, which does not check signatures).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl ccache ninja-build pkg-config file rsync \
      gcc-12 g++-12 \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
      libcurl4-openssl-dev zlib1g-dev python3 python3-jinja2 \
 && rm -rf /var/lib/apt/lists/*

# CMake: Debian 12 ships 3.25 and the tree needs 3.28. This is the same static build the workflow
# uses, so configure sees the same CMake and the same FindCUDAToolkit.
ARG CMAKE_VERSION=3.31.6
RUN curl -fsSLo /tmp/cmake.tgz \
      "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
 && tar -xzf /tmp/cmake.tgz -C /opt \
 && rm -f /tmp/cmake.tgz
ENV PATH=/opt/cmake-3.31.6-linux-x86_64/bin:$PATH

# CUDA 13.4, only the slice a compile needs.
ARG CUDA_PKG_VERSION=13-4
RUN curl -fsSLo /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb \
 && rm -f /tmp/cuda-keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "cuda-nvcc-${CUDA_PKG_VERSION}" "cuda-nvtx-${CUDA_PKG_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

# /usr/local/cuda is a symlink normally created by cuda-toolkit-13-4, which is not installed here;
# cuda-nvcc-13-4 creates it itself, but check rather than assume, and prove nvcc runs.
RUN set -eux; \
    [ -e /usr/local/cuda ] || ln -s /usr/local/cuda-13.4 /usr/local/cuda; \
    ls -ld /usr/local/cuda; \
    /usr/local/cuda/bin/nvcc --version

ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12

# cublas, from the CUDA 13 wheel that build.sh mounts at /opt/nvidia. These symlinks dangle while
# the image is built and resolve when the container runs.
#
# Both names are needed. The linker resolves -lcublas through libcublas.so, but the wheel's
# libcublas.so.13 records a DT_NEEDED on libcublasLt.so.13 and the linker looks that up by soname,
# so the versioned names have to exist in the search directory too.
RUN set -eux; \
    cuda=/usr/local/cuda; \
    libdir="$cuda/lib64"; \
    [ -d "$libdir" ] || libdir="$cuda/targets/x86_64-linux/lib"; \
    for h in cublas_v2.h cublasLt.h cublas_api.h cublas.h; do \
      ln -sf "/opt/nvidia/cu13/include/$h" "$cuda/include/$h"; \
    done; \
    for l in libcublas libcublasLt; do \
      ln -sf "/opt/nvidia/cu13/lib/$l.so.13" "$libdir/$l.so"; \
      ln -sf "/opt/nvidia/cu13/lib/$l.so.13" "$libdir/$l.so.13"; \
    done

WORKDIR /src
