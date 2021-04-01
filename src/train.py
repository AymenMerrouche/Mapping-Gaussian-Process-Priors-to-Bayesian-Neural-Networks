from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from src.datasets import get_toy_data
from src.models import BayesianMLP, RegularMLP
from src.utils import plot_results, evaluate_regression


def main(log_dir):
    writer = SummaryWriter(str(log_dir))

    x_train, y_train, x_test, y_test = get_toy_data()
    dataloader_train = DataLoader(
        TensorDataset(x_train, y_train), batch_size=32, shuffle=True
    )

    model = BayesianMLP(dim_in=1, dim_out=1, dim_h=512)
    # model = RegularMLP(dim_in=1, dim_out=1, dim_h=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    global_step = 0
    for epoch in range(1000):
        loss_history = []
        # M = len(dataloader_train)
        M = len(x_train)
        for i, (x, y) in enumerate(dataloader_train):
            optimizer.zero_grad()
            if isinstance(model, BayesianMLP):
                loss = model.sample_elbo(
                    inputs=x,
                    labels=y,
                    criterion=torch.nn.MSELoss(),
                    sample_nbr=3,
                    complexity_cost_weight=1 / M,
                    # complexity_cost_weight=(2 ** M - i) / (2 ** M - 1),
                )
            else:
                loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_history.append(loss.item())

        writer.add_scalar("loss", np.mean(loss_history), epoch)
        if isinstance(model, BayesianMLP):
            y_pred_train, _ = model(x_train, num_samples=10)
        else:
            y_pred_train = model(x_train)
        writer.add_scalar(
            "mse_train",
            F.mse_loss(y_pred_train, y_train).mean().item(),
            epoch,
        )

        if epoch % 50 == 0 and isinstance(model, BayesianMLP):
            # plot results with all ground truth
            y_pred, std_pred = model(x_test, num_samples=10)
            fig = plot_results(x_train, y_train, x_test, y_test, y_pred, std_pred)
            writer.add_figure("regression", fig, epoch)


if __name__ == "__main__":
    time_tag = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    log_dir = f"../runs/{time_tag}"
    main(log_dir)
