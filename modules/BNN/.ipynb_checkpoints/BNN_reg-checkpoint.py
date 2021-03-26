import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from .activations import *
import math

# sigma = log(1 + exp(rho))
# find rho for sigma = 1
# (our prior is a diagonal gaussian)
rho = torch.log(torch.exp(torch.tensor(1))-1).item()

@variational_estimator
class BayesianRegressor(nn.Module):
    """Two Layers BNN for regression with custom activation functions"""
    def __init__(self, input_dim, output_dim, hidden=32, activation=torch.sin):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, hidden, posterior_rho_init = rho)
        self.blinear2 = BayesianLinear(hidden, output_dim, posterior_rho_init = rho)
        self.activation = activation
        
    def forward(self, x):
        x_ = self.activation(self.blinear1(x))
        return self.blinear2(x_)
    
    def sample(self, lower_bound=0, upper_bound=10, n_sampling_points = 1000 ,n_samples = 3, ax = None):
        """
        Sample BNN functions (f(x) ~ p_bnn(f(x)))
        :param lower_bound: lower bound for evaluation points.
        :param upper_bound: upper bound for evaluation points.
        :param n_sampling_points: number of evaluation points.
        :param n_samples: number of samples.
        :param ax: matplotlib axis, if not None then plot samples.
        """
        sampling_points = torch.linspace(lower_bound, upper_bound, n_sampling_points).unsqueeze(-1)
        with torch.no_grad():
            s = [self.forward(sampling_points).squeeze(-1) for _ in range(n_samples)]
        if ax is not None :
            for i in range(n_samples):
                ax.plot(sampling_points.detach().cpu(),s[i].detach().cpu())
        else:
            return s