#include <torch/extension.h>


void gen_voxel_label_launch(torch::Tensor voxel_labels,
                            torch::Tensor pt_lab,
                            torch::Tensor pt_ind);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gen_voxel_label_cuda", &gen_voxel_label_launch);
}