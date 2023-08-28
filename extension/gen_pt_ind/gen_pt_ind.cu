#include "gen_pt_ind.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <stdio.h>

#define THREAD_PER_BLOCK 256


__global__ void gen_pt_ind_kernel(torch::PackedTensorAccessor32<float, 2> cat_pt_ind,
                           torch::PackedTensorAccessor32<float, 2> cat_xyz,
                           torch::PackedTensorAccessor32<float, 1> min_bound,
                           torch::PackedTensorAccessor32<float, 1> intervals)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= cat_xyz.size(0))
    {
        return;
    }

    cat_pt_ind[n][0] = cat_xyz[n][0];
    cat_pt_ind[n][1] = (cat_xyz[n][1] - min_bound[0]) / intervals[0];
    cat_pt_ind[n][2] = (cat_xyz[n][2] - min_bound[1]) / intervals[1];
    cat_pt_ind[n][3] = (cat_xyz[n][3] - min_bound[2]) / intervals[2];
}


void gen_pt_ind_launch(torch::Tensor cat_pt_ind,
                       torch::Tensor cat_xyz,
                       torch::Tensor min_bound,
                       torch::Tensor intervals)
{
    const at::cuda::OptionalCUDAGuard device_guard(cat_pt_ind.device());

    // 配置block和thread
    dim3 thread(THREAD_PER_BLOCK, 1, 1);
    int temp = cat_pt_ind.size(0) / THREAD_PER_BLOCK + ((cat_pt_ind.size(0) % THREAD_PER_BLOCK) > 0);
    dim3 block(temp, 1, 1);

    // 启动kernel
    gen_pt_ind_kernel<<<block, thread>>> (cat_pt_ind.packed_accessor32<float, 2>(),
                                   cat_xyz.packed_accessor32<float, 2>(),
                                   min_bound.packed_accessor32<float, 1>(),
                                   intervals.packed_accessor32<float, 1>());
}