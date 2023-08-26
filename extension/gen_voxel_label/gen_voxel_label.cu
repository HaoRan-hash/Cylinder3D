#include "gen_voxel_label.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <stdio.h>

#define THREAD_PER_BLOCK 256


__global__ void gen_voxel_label_kernel(torch::PackedTensorAccessor32<int, 5> voxel_labels,
                                       torch::PackedTensorAccessor32<int64_t, 2> pt_lab,
                                       torch::PackedTensorAccessor32<int64_t, 2> pt_ind)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= pt_lab.size(0))
    {
        return;
    }

    int64_t b = pt_ind[n][0];
    int64_t h = pt_ind[n][1];
    int64_t w = pt_ind[n][2];
    int64_t l = pt_ind[n][3];
    int64_t c = pt_lab[n][0];

    atomicAdd(&voxel_labels[b][h][w][l][c], 1);
}


void gen_voxel_label_launch(torch::Tensor voxel_labels,
                            torch::Tensor pt_lab,
                            torch::Tensor pt_ind)
{
    const at::cuda::OptionalCUDAGuard device_guard(voxel_labels.device());

    // 配置block和thread
    dim3 thread(THREAD_PER_BLOCK, 1, 1);
    int temp = pt_lab.size(0) / THREAD_PER_BLOCK + ((pt_lab.size(0) % THREAD_PER_BLOCK) > 0);
    dim3 block(temp, 1, 1);

    // 启动kernel
    gen_voxel_label_kernel<<<block, thread>>> (voxel_labels.packed_accessor32<int, 5>(),
                                               pt_lab.packed_accessor32<int64_t, 2>(),
                                               pt_ind.packed_accessor32<int64_t, 2>());
}