from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='gen_voxel_label',
    ext_modules=[CUDAExtension('gen_voxel_label', 
                               ['gen_voxel_label.cu'])],
    cmdclass={'build_ext': BuildExtension}
)