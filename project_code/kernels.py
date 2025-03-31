# kernels.py

import numpy as np

def linear_kernel(x, z):
    return np.dot(x, z)

def polynomial_kernel(x, z, degree=3, coef0=1.0):
    return (np.dot(x, z) + coef0)**degree

def rbf_kernel(x, z, gamma=1.0):
    # gamma = 1 / (2 sigma^2)
    diff = x - z
    return np.exp(-gamma * np.dot(diff, diff))

def laplacian_kernel(x, z, gamma=1.0):
    diff = x - z
    return np.exp(-gamma * np.sum(np.abs(diff)))

# For convenience, you can define a wrapper to produce a kernel_func with certain parameters:
def make_polynomial_kernel(degree=3, coef0=1.0):
    def kernel(x, z):
        return polynomial_kernel(x, z, degree=degree, coef0=coef0)
    return kernel

def make_rbf_kernel(gamma=1.0):
    def kernel(x, z):
        return rbf_kernel(x, z, gamma=gamma)
    return kernel

def make_laplacian_kernel(gamma=1.0):
    def kernel(x, z):
        return laplacian_kernel(x, z, gamma=gamma)
    return kernel
