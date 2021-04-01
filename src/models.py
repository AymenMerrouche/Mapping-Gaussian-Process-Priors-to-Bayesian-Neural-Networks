import torch
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch import nn


class RegularMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_out),
        )

    def forward(self, x, num_samples=1):
        if num_samples == 1:
            # return a single prediction
            return self.model(x)


@variational_estimator
class BayesianMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h):
        super().__init__()
        self.blinear1 = BayesianLinear(dim_in, dim_h, prior_sigma_1=1.)
        self.blinear2 = BayesianLinear(dim_h, dim_out, prior_sigma_1=1.)

    def forward(self, x, num_samples=1):
        if num_samples == 1:
            # return a single prediction
            x = torch.relu(self.blinear1(x))
            return self.blinear2(x)

        else:
            # make num_samples predictions with different weights
            predictions = [self(x) for _ in range(num_samples)]

            predictions = torch.stack(predictions)  # (num_samples, *)
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            return mean, std
