#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cuda/atomic>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_pipeline.h>
int main(){ cublasHandle_t h=nullptr; cublasCreate(&h); cublasDestroy(h); return 0; }
