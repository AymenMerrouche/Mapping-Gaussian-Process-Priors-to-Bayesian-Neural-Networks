from pathlib import Path

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    x_all, y_all_pred, y_all_ground_truth=None, x_train=None, y_train=None, mode="bnn", title=""
):
    """
    Takes numpy arrays of 1D data, with optionally ground truth data and training samples.

    Args:
        x_all: (n,)
        y_all_pred: (n, N_samples): BNN predictions from the N_samples models
        y_all_ground_truth: (n,), optional
        x_train: (n_train), optional
        y_train: (n_train), optional

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
    plt.savefig(project_dir + title + "-deciles.pdf", bbox_inches="tight")
    return fig

def plot_prior(sampling_points, predictions, title):
    fig = plt.figure(facecolor="white", figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(sampling_points, predictions[:, :5], sns.xkcd_rgb[sample_col["gpp"]], lw=1)
    plt.savefig(project_dir + title + "-prior.pdf", bbox_inches="tight")
    return fig

