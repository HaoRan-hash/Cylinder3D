#include <torch/extension.h>


void gen_pt_ind_launch(torch::Tensor cat_pt_ind,
                       torch::Tensor cat_xyz,
                       torch::Tensor min_bound,
                       torch::Tensor intervals);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gen_pt_ind_cuda", &gen_pt_ind_launch);
}