import torch
import numpy as np


def get_toy_data(num_samples=100, sigma=0.05):
    """
    Create a noisy dataset of samples f(x) = sin(2 pi x) + x
    With
        x in [0,1] for training samples
        x in [-1, 2] for ground truth samples

    Args:
        num_samples:
        sigma:

    Returns:
        x_train, y_train, x_val, y_val, x_all, y_all
    """

    def f(t):
        return np.sin(2 * np.pi * t) + t

    x_all = np.linspace(-1, 2, 150).reshape((-1, 1))
    y_all = f(x_all)

    # training
    x_train = np.random.uniform(0, 1, num_samples).reshape((-1, 1))
    noise = np.random.normal(0, sigma, len(x_train)).reshape((-1, 1))
    y_train = f(x_train) + noise
    # validation
    x_val = np.random.uniform(0, 1, 100).reshape((-1, 1))
    noise_val = np.random.normal(0, sigma, len(x_val)).reshape((-1, 1))
    y_val = f(x_val) + noise_val

    x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
    x_val, y_val = torch.tensor(x_val).float(), torch.tensor(y_val).float()
    x_all, y_all = torch.tensor(x_all).float(), torch.tensor(y_all).float()

    return x_train, y_train, x_val, y_val, x_all, y_all