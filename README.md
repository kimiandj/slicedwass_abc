### Sliced-Wasserstein Approximate Bayesian Computation

This repository contains the implementation of Sliced-Wasserstein Approximate Bayesian Computation (SW-ABC). We apply it to approximate the scaling factor of the covariance matrix of a multivariate Gaussian distribution, and compare the SW-ABC performance against other ABC approaches.

Requirements: pyabc, cython, scipy.

Before running the code, you need to compile the C files with:
`python setup.hilbert_caller.py build_ext --inplace`
`python setup.swapsweep_caller.py build_ext --inplace`