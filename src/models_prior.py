import torch
from blitz.modules import TrainableRandomDistribution, PriorWeightDistribution

from torch import nn
from torch.distributions import Normal, Categorical, MixtureSameFamily
import torch.nn.functional as F
import numpy as np


class RBF(nn.Module):
    def forward(self, x):
        return torch.exp(-(x ** 2))
class Linear(nn.Module):
    def forward(self, x):
        return x 
class Sinus(nn.Module):
    def forward(self, x):
        return torch.sin(x) 


class BayesianLinear(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        prior_sigma: float,
    ):
        """
        Assume a gaussian prior w_i ~ Normal(0, prior_sigma), with w_i iid
        Args:
            dim_in:
            dim_out:
            prior_sigma: std
        """
        super().__init__()
        # Parameters theta=[mu,rho] of variational posterior q(w|theta): for weights...
        # Variational weight parameters and sample
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.prior_sigma = prior_sigma
        self.mu = nn.Parameter(torch.zeros(dim_out, dim_in))
        self.rho = nn.Parameter(
            torch.ones(dim_out, dim_in) * (-np.log(np.exp(1)-1))
        )
        # ... and biases
        self.mu_bias = nn.Parameter(torch.zeros(dim_out))
        self.rho_bias = nn.Parameter(
            torch.ones(dim_out) * (-np.log(np.exp(1)-1))
        )

        # Prior
        self.prior_weights = Normal(0, prior_sigma)
        self.prior_bias = Normal(0, prior_sigma)
        # Gaussian mixture prior (like Bayes By Backprop paper)
        # implementation below is un-tested, for this prior, try BayesianLinearBlitz
        # mix = Categorical(torch.Tensor([prior_pi, 1 - prior_pi]))
        # comp = Normal(torch.zeros(2), torch.Tensor([prior_sigma_1, prior_sigma_2]))
        # self.prior = MixtureSameFamily(mix, comp)

        # todo: add prior that can be optimized to match GP

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
        self.log_prior = self.prior_weights.log_prob(w).sum() + self.prior_bias.log_prob(b).sum()
        return F.linear(x, w, b)
    def copy_otpimized_prior(self):
        """prior optimisation is done on the first variational parameters
           after this step, these parameters are copied to the prior and reinitialised
        """
        # set the prior
        self.prior_weights = Normal(self.mu.data, torch.log1p(torch.exp(self.rho.data)))
        self.prior_bias = Normal(self.mu_bias.data, torch.log1p(torch.exp(self.rho_bias.data)))
        # reinitialize variational parameters
        # for weights
        self.mu = nn.Parameter(torch.zeros(self.dim_out, self.dim_in))
        self.rho = nn.Parameter(
            torch.ones(self.dim_out, self.dim_in) * (-np.log(np.exp(1)-1))
        )
        # ... and biases
        self.mu_bias = nn.Parameter(torch.zeros(self.dim_out))
        self.rho_bias = nn.Parameter(
            torch.ones(self.dim_out) * (-np.log(np.exp(1)-1))
        )
        self.log_variational_posterior = 0.0
        self.log_prior = 0.0


class VariationalEstimator(nn.Module):
    def sample_elbo(
        self,
        inputs,
        labels,
        n_samples: int,
        complexity_cost_weight: float,
        model_noise_var: float,
    ):
        r"""
        Assuming:
            - prior on the weights P(w)
            - variational posterior q(w|theta)
            - model P(y|x,w)
        compute an estimate of the ELBO loss by sampling `n_samples` weights w from q(w|theta):
        .. math::
            elbo loss = \sum_{i=1}^n_samples \log q(w_i|\theta) - \log P(w_i) - \log P(y|x,w_i)
        Args:
            inputs: (batch_size, dim_x)
            labels: (batch_size, dim_y)
            n_samples: number of BNN weight samples to draw for the ELBO estimate
            complexity_cost_weight: term in front of the KL
            model_noise_var: sigma_y**2, assuming that P(y|x,w) = Normal(f_w(x), sigma_y)
        Returns:
            estimate of the elbo loss (to minimize)
        """
        kl_div = 0
        log_likelihood = 0
        for _ in range(n_samples):
            outputs = self(inputs)
            kl_div += self.kl_div() * complexity_cost_weight
            # log P(y|x,w), assuming a gaussian distribution (regression tasks)
            # todo: for classification tasks, use cross entropy
            log_likelihood -= ((outputs - labels) ** 2 / model_noise_var).sum()

        kl_div /= n_samples
        log_likelihood /= n_samples
        loss = kl_div - log_likelihood
        return loss, kl_div, log_likelihood

    def kl_div(self):
        r"""
        Computes empirical estimate of
            KL[q(w|mu,rho) || P(w)] = KL[variational posterior || prior]
        .. math::
            \sum_i \log q(w_i|\mu,\rho) - \log P(w_i)
        where w_i are sampled from q
        Returns:
        """
        kl_divergence = 0
        for module in self.modules():
            if hasattr(module, "log_variational_posterior"):
                kl_divergence += module.log_variational_posterior - module.log_prior
        return kl_divergence


class BayesianMLP(VariationalEstimator):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_h: int,
        activation: str,
        prior_type: str,
        prior_sigma: float = None,
        prior_sigma_1: float = None,
        prior_sigma_2: float = None,
        prior_pi: float = None,
        posterior_rho_var: float = None,
    ):
        super().__init__()
        if activation == "rbf":
            activation_fct = RBF()
        elif activation == "linear":
            activation_fct = Linear()
        elif activation == "softplus":
            activation_fct = nn.Softplus()
        elif activation == "sinus":
            activation_fct = Sinus()
        elif activation == "relu":
            activation_fct = nn.ReLU()
        elif activation == "tanh":
            activation_fct = nn.Tanh()
        else:
            raise ValueError

        if prior_type == "normal":
            self.model = nn.Sequential(
                BayesianLinear(
                    dim_in,
                    dim_h,
                    prior_sigma=prior_sigma,
                ),
                activation_fct,
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
    def copy_optimised_priors(self):
        self.model[0].copy_otpimized_prior()
        self.model[2].copy_otpimized_prior()
        
        


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