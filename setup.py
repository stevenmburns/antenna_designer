from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "pysim_accelerators",
        ["src/antenna_designer/pysim_accelerators.cpp"],
        extra_compile_args=["-fopenmp", "-g"],
        extra_link_args=["-fopenmp", "-lpthread"],
    ),
]

setup(
    name='antenna_designer',
    ext_modules=ext_modules,
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
