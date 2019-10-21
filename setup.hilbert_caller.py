from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    "HilbertCode_caller.pyx",                 # our Cython source
    language="c++",             # generate C++ code
    )
)