from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='gen_pt_ind',
    ext_modules=[CUDAExtension('gen_pt_ind', 
                               ['gen_pt_ind.cu'])],
    cmdclass={'build_ext': BuildExtension}
)