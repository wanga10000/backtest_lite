from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize([
        "c_utils/c_backtest_utils.pyx",
        "c_utils/c_backtest_core.pyx",
        "c_utils/c_backtest_mea_core.pyx",
    ], compiler_directives={
        'language_level': 3,
    }),
    include_dirs=[np.get_include()]
)