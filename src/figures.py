from pathlib import Path

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from torch.nn.functional import mse_loss

from models import BayesianMLP

sns.set_style("white")
n = 9
bnn_col = ["deep sky blue", "bright sky blue"]
gpp_bnn_col = ["red", "salmon"]
gp_col = ["green", "light green"]
colors = {"bnn": bnn_col, "gpp": gpp_bnn_col, "gp": gp_col}
sample_col = {"bnn": "bright sky blue", "gpp": "watermelon", "gp": "light lime"}
pal_col = {
    "bnn": sns.light_palette("#3498db", n_colors=n),  # nice blue
    "gpp": sns.light_palette("#e74c3c", n_colors=n),  # nice red
    "gp": sns.light_palette("#2ecc71", n_colors=n),
}  # nice green eh not so nice

project_dir = "../figures/"


def plot_deciles(
    x_all, y_all_pred, y_all_ground_truth=None, x_train=None, y_train=None, mode="bnn", title=None
):
    """
    Takes numpy arrays of 1D data, with optionally ground truth data and training samples.
    Args:
        x_all: (n,)
        y_all_pred: (n, N_samples): BNN predictions from the N_samples models
        y_all_ground_truth: (n,), optional
        x_train: (n_train), optional
        y_train: (n_train), optional
        mode: different color codes to use
        title: title of figure if saved
    Returns:
        figure
    """
    color = colors[mode]

    mean = np.mean(y_all_pred, axis=1)
    std = np.std(y_all_pred, axis=1)

    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)

    # Get critical values for the deciles
    lvls = 0.1 * np.linspace(1, 9, 9)
    alphas = 1 - 0.5 * lvls
    zs = norm.ppf(alphas)

    if x_train is not None:
        # plot data
        ax.plot(x_train, y_train, "ko", ms=4, label="Training data")
        # plot ground truth
        ax.plot(x_all, y_all_ground_truth, "k", lw=1, label="Ground truth")

    # plot samples
    ax.plot(x_all, y_all_pred[:, :5], sns.xkcd_rgb[sample_col[mode]], lw=1)

    # plot mean of samples
    ax.plot(x_all, mean, sns.xkcd_rgb[color[0]], lw=1, label="Prediction mean")

    # plot the deciles
    pal = pal_col[mode]
    for z, col in zip(zs, pal):
        ax.fill_between(x_all, mean - z * std, mean + z * std, color=col)
    ax.tick_params(labelleft="off", labelbottom="off")
    # ax.set_ylim([-2, 3])
    # ax.set_xlim([-8, 8])

    plt.legend()
    if title is not None:
        plt.savefig(project_dir + title + "_deciles.png", bbox_inches="tight")
    return fig


def plot_prior(model, x_all, n_samples=50, writer=None):
    # plot results with all ground truth
    y_all_pred = (
        torch.stack(
            [model(x_all, sample_from_prior=True).view(-1) for _ in range(n_samples)],
            dim=1,
        )
        .detach()
        .numpy()
    )
    fig = plot_deciles(
        x_all.view(-1).detach().numpy(),
        y_all_pred,
    )
    if writer is not None:
        writer.add_figure("prior", fig, 0)
    else:
        plt.plot()


def plot_posterior(model, x_all, y_all, x_train, y_train, n_samples=50):
    y_pred_train, _ = model(x_train, num_samples=30)
    mse_train = mse_loss(y_pred_train, y_train).mean().item()
    print(f"MSE (train - mean of 30 BNN samples): {mse_train}")

    # plot results with all ground truth
    y_all_pred = (
        torch.stack([model(x_all).view(-1) for _ in range(n_samples)], dim=1)
        .detach()
        .numpy()
    )
    fig = plot_deciles(
        x_all.view(-1).detach().numpy(),
        y_all_pred,
        y_all.view(-1).detach().numpy(),
        x_train.view(-1).detach().numpy(),
        y_train.view(-1).detach().numpy(),
    )
    plt.plot()

def plot_prior_no_deciles(sampling_points, predictions, title):
    """
    Plots prior without deciles
    """
    fig = plt.figure(facecolor="white", figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(sampling_points, predictions[:, :5], sns.xkcd_rgb[sample_col["gpp"]], lw=1)
    plt.savefig(project_dir + title + "-prior.png", bbox_inches="tight")
    return fig

def plot_priors(sampling_points, gpp, bnnp, bnnop, title):
    """
    Plot priors of three different models
    """
    to_plot = {"gp":gpp, "gpp":bnnop,"bnn":bnnp}
    x_all = sampling_points
    fig = plt.figure(facecolor="white", figsize=(10, 5))
    for i,mode in enumerate(to_plot):
        y_all_pred = to_plot[mode]
        
        color = colors[mode]

        mean = np.mean(y_all_pred, axis=1)
        std = np.std(y_all_pred, axis=1)

        ax = fig.add_subplot(310+(i+1))
        ax.set_title(mode)

        # Get critical values for the deciles
        lvls = 0.1 * np.linspace(1, 9, 9)
        alphas = 1 - 0.5 * lvls
        zs = norm.ppf(alphas)

        # plot samples
        ax.plot(x_all, y_all_pred[:, :5], sns.xkcd_rgb[sample_col[mode]], lw=1)
        # plot mean of samples
        ax.plot(x_all, mean, sns.xkcd_rgb[color[0]], lw=1, label="Prediction mean")

        # plot the deciles
        pal = pal_col[mode]
        for z, col in zip(zs, pal):
            ax.fill_between(x_all, mean - z * std, mean + z * std, color=col)
        ax.tick_params(labelleft="off", labelbottom="off")
        # ax.set_ylim([-2, 3])
        # ax.set_xlim([-8, 8])

        plt.legend()
        plt.savefig(project_dir + title + "-prior.png", bbox_inches="tight")
        plt.suptitle(title)
    fig.tight_layout()
    return fig
