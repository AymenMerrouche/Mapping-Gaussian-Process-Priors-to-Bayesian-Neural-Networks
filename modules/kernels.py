import math
import torch
import gpytorch


class GaussianKernel(gpytorch.kernels.Kernel):
    def __init__(self, sigma_squared = 2, **kwargs):
        super().__init__(**kwargs)
        self.sigma_squared = sigma_squared
    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2,square_dist=True, **params)
        return torch.exp(-diff/self.sigma_squared)