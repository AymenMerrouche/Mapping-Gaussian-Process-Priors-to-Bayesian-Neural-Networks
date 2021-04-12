import math
import torch
import gpytorch
import numpy as np


class GaussianKernel(gpytorch.kernels.Kernel):
    def __init__(self, sigma_squared = 2, **kwargs):
        super().__init__(**kwargs)
        self.sigma_squared = sigma_squared
    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2,square_dist=True, **params)
        return torch.exp(-diff/self.sigma_squared)
class PeriodicKernel(gpytorch.kernels.Kernel):
    def __init__(self, sigma_squared = 1, **kwargs):
        super().__init__(**kwargs)
        self.sigma_squared = sigma_squared
    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = torch.sqrt(self.covar_dist(x1, x2,square_dist=True, **params))
        return torch.exp(-2 * ((torch.sin(diff*np.pi)**2) / self.sigma_squared ))
    
class RqKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2,square_dist=True, **params)
        return torch.pow(1 + (diff/6), -3)
    
class GaussianPeriodicKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2,square_dist=True, **params)
        diff1 = torch.sqrt(diff)
        diff1 = torch.sin(diff*np.pi)**2
        per = torch.exp(-2 * (diff1/25))
        rbf = torch.exp(-diff/2)
        return per*rbf
    
all_kernels = {
    "Linear Kernel" :  gpytorch.kernels.LinearKernel(),
    "Gaussian Kernel 2" : GaussianKernel(),
    "Gaussian Kernel 50" : GaussianKernel(sigma_squared = 50),
    "Periodic Kernel 1" : PeriodicKernel(),
    "Periodic Kernel 25" : PeriodicKernel(sigma_squared = 25),
    "RQ Kernel" : RqKernel(),
    "Gaussian Periodic Kernel" : GaussianPeriodicKernel()
}