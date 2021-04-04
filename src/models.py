import torch

# from blitz.modules import BayesianLinear
from blitz.modules import PriorWeightDistribution, TrainableRandomDistribution
from torch import nn
from torch.distributions import Normal, Categorical, MixtureSameFamily
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        prior_sigma: float,
    ):
        super().__init__()
        # Parameters theta=[mu,rho] of variational posterior q(w|theta): for weights...
        # Variational weight parameters and sample
        self.mu = nn.Parameter(torch.zeros(dim_out, dim_in))
        self.rho = nn.Parameter(
            torch.log(torch.exp(torch.ones(dim_out, dim_in) * prior_sigma) - 1)
        )
        # ... and biases
        self.mu_bias = nn.Parameter(torch.zeros(dim_out))
        self.rho_bias = nn.Parameter(
            torch.log(torch.exp(torch.ones(dim_out) * prior_sigma) - 1)
        )

        # Prior
        self.prior = Normal(0, prior_sigma)
        # Gaussian mixture prior (like Bayes By Backprop paper)
        # mix = Categorical(torch.Tensor([prior_pi, 1 - prior_pi]))
        # comp = Normal(torch.zeros(2), torch.Tensor([prior_sigma_1, prior_sigma_2]))
        # self.prior = MixtureSameFamily(mix, comp)

        self.log_variational_posterior = 0.0
        self.log_prior = 0.0

    def forward(self, x):
        # Sample the weights and forward it
        # perform all operations in the forward rather than in __init__ (including log[1+exp(rho)])
        variational_posterior = Normal(self.mu, torch.log1p(torch.exp(self.rho)))
        variational_posterior_bias = Normal(
            self.mu_bias, torch.log1p(torch.exp(self.rho_bias))
        )
        w = variational_posterior.rsample()
        b = variational_posterior_bias.rsample()

        # Get the log prob
        self.log_variational_posterior = (variational_posterior.log_prob(w)).sum() + (
            variational_posterior_bias.log_prob(b)
        ).sum()
        self.log_prior = (self.prior.log_prob(w) + self.prior.log_prob(b)).sum()
        return F.linear(x, w, b)


class VariationalEstimator(nn.Module):
    def sample_elbo(
        self, inputs, labels, criterion, sample_nbr: int, complexity_cost_weight: float
    ):
        kl_div = 0
        log_likelihood = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            log_likelihood -= criterion(outputs, labels) / sample_nbr
            kl_div += self.kl_div() * complexity_cost_weight / sample_nbr
        loss = kl_div - log_likelihood
        return loss, kl_div, log_likelihood

    def kl_div(self):
        r"""
        Computes empirical estimate of
            KL[q(w|mu,rho) || P(w)] = KL[variational posterior || prior]

        .. math::
            \sum_i \log q(w|\mu,\rho) - \log P(w)
        where w is sampled from q

        Returns:

        """
        kl_divergence = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl_divergence += module.log_variational_posterior - module.log_prior
        return kl_divergence


class RBF(nn.Module):
    def forward(self, x):
        return torch.exp(-(x ** 2))


class BayesianMLP(VariationalEstimator):
    def __init__(self, dim_in, dim_out, dim_h, prior_sigma):
        super().__init__()
        self.model = nn.Sequential(
            BayesianLinear(
                dim_in,
                dim_h,
                prior_sigma=prior_sigma,
            ),
            RBF(),
            BayesianLinear(
                dim_h,
                dim_h,
                prior_sigma=prior_sigma,
            ),
            RBF(),
            BayesianLinear(
                dim_h,
                dim_h,
                prior_sigma=prior_sigma,
            ),
            RBF(),
            BayesianLinear(
                dim_h,
                dim_out,
                prior_sigma=prior_sigma,
            ),
        )

    def forward(self, x, num_samples=1):
        if num_samples == 1:
            # return a single prediction
            return self.model(x)

        else:
            # make num_samples predictions with different weights
            predictions = [self(x) for _ in range(num_samples)]

            predictions = torch.stack(predictions)  # (num_samples, *)
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            return mean, std


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
