
import torch, os
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.realpath(__file__))
torch_dir = os.path.dirname(torch.__file__)
conda_include_dir = '/'.join(torch_dir.split('/')[:-4]) + '/include'

extra = {'cxx': ['-std=c++11', '-fopenmp'], 'nvcc': ['-std=c++11', '-Xcompiler', '-fopenmp']}

setup(
    name='IntrinsicCost',
    packages=['Cost'],
    ext_modules=[
      CUDAExtension('Cost',
        ['Cost/cuda.cu', 'Cost/cost_compute_cuda.cpp','Cost/pybind.cpp'],
        include_dirs=[conda_include_dir, this_dir+'/Cost/'],
        extra_compile_args=extra)],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
