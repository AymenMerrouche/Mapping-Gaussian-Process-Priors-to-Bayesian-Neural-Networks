from datetime import datetime
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import get_toy_data
from figures import plot_deciles
from models import BayesianMLP, VariationalEstimator


def train(
    model: VariationalEstimator,
    optimizer: torch.optim.Optimizer,
    dataloader_train,
    n_epochs: int,
    log_dir: str,
    evaluate_func,
    evaluate_data,
):
    writer = SummaryWriter(str(log_dir))
    global_step = 0
    for epoch in range(n_epochs):
        loss_history = []
        kl_div_history = []
        log_p_history = []
        M = len(dataloader_train)
        # M = len(x_train)
        for i, (x, y) in enumerate(dataloader_train):
            optimizer.zero_grad()
            loss, kl_div, log_p = model.sample_elbo(
                inputs=x,
                labels=y,
                criterion=torch.nn.MSELoss(),
                sample_nbr=3,
                complexity_cost_weight=1 / M,
                # complexity_cost_weight=(2 ** M - i) / (2 ** M - 1),
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_history.append(loss.item())
            kl_div_history.append(kl_div.item())
            log_p_history.append(log_p.item())

        with torch.no_grad():
            # todo: use weight means instead of samples?
            val_metric = evaluate_func(
                model,
                evaluate_data,
                writer,
                loss_history,
                kl_div_history,
                log_p_history,
                epoch,
            )
    return val_metric


# --- 1D regression
def eval_1d_regression(
    model: BayesianMLP,
    data,
    writer,
    loss_history: List[float],
    kl_div_history: List[float],
    log_p_history: List[float],
    epoch: int,
):
    writer.add_scalar("loss", np.mean(loss_history), epoch)
    writer.add_scalar("kl_div", np.mean(kl_div_history), epoch)
    writer.add_scalar("log_p", np.mean(log_p_history), epoch)

    x_train, y_train, x_val, y_val, x_all, y_all = data
    y_pred_train, _ = model(x_train, num_samples=30)
    mse_train = F.mse_loss(y_pred_train, y_train).mean().item()
    writer.add_scalar("mse_train", mse_train, epoch)
    y_pred_val, _ = model(x_val, num_samples=30)
    mse_val = F.mse_loss(y_pred_val, y_val).mean().item()
    writer.add_scalar("mse_val", mse_val, epoch)

    if epoch % 30 == 0:
        n_samples = 50
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
        writer.add_figure("regression", fig, epoch)

    return mse_val


def train_1d_regression():
    dim_h = 20
    prior_sigma = 1.0

    log_dir = (
        project_dir
        / f"runs/individual/{datetime.now().strftime('%Y%m%d_%H%M%S')}-dim_h_{dim_h}"
        f"-sigma_{prior_sigma:.2f}"
    )

    x_train, y_train, x_val, y_val, x_all, y_all = get_toy_data()
    dataloader_train = DataLoader(
        TensorDataset(x_train, y_train), batch_size=16, shuffle=True
    )

    model = BayesianMLP(
        dim_in=1,
        dim_out=1,
        dim_h=dim_h,
        prior_sigma=prior_sigma,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    return train(
        model,
        optimizer,
        dataloader_train,
        n_epochs=1000,
        log_dir=log_dir,
        evaluate_func=eval_1d_regression,
        evaluate_data=(x_train, y_train, x_val, y_val, x_all, y_all),
    )


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    train_1d_regression()
