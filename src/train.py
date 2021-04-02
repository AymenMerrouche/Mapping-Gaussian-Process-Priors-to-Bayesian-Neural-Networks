import argparse
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from datasets import get_toy_data
from models import BayesianMLP, RegularMLP
from utils import plot_results, evaluate_regression
import optuna


def log_metrics(writer, model, data, loss_history: List[float], epoch: int):
    x_train, y_train, x_val, y_val, x_all, y_all = data
    writer.add_scalar("loss", np.mean(loss_history), epoch)
    y_pred_train, _ = model(x_train, num_samples=10)
    mse_train = F.mse_loss(y_pred_train, y_train).mean().item()
    writer.add_scalar("mse_train", mse_train, epoch)
    y_pred_val, _ = model(x_val, num_samples=10)
    mse_val = F.mse_loss(y_pred_val, y_val).mean().item()
    writer.add_scalar("mse_val", mse_val, epoch)

    if epoch % 500 == 0:
        # plot results with all ground truth
        y_pred, std_pred = model(x_all, num_samples=10)
        fig = plot_results(x_train, y_train, x_all, y_all, y_pred, std_pred)
        writer.add_figure("regression", fig, epoch)

    return mse_val


def objective(trial: optuna.Trial):
    dim_h = 512
    prior_sigma_1 = trial.suggest_float("prior_sigma_1", low=1e-3, high=10.0, log=True)
    prior_sigma_2 = trial.suggest_float("prior_sigma_2", low=1e-3, high=10.0, log=True)
    prior_pi = trial.suggest_float("prior_pi", low=0.0, high=1.0)

    log_dir = (
        project_dir
        / f"runs/{trial.study.study_name}/trial_{trial.number}-dim_h_{dim_h}-"
        f"s1_{prior_sigma_1:.2f}-s2_{prior_sigma_2:.2f}-pi_{prior_pi:.2f}"
    )
    writer = SummaryWriter(str(log_dir))

    x_train, y_train, x_val, y_val, x_all, y_all = get_toy_data()
    dataloader_train = DataLoader(
        TensorDataset(x_train, y_train), batch_size=16, shuffle=True
    )

    model = BayesianMLP(
        dim_in=1,
        dim_out=1,
        dim_h=dim_h,
        prior_sigma_1=prior_sigma_1,
        prior_sigma_2=prior_sigma_2,
        prior_pi=prior_pi,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    global_step = 0
    for epoch in range(1000):
        loss_history = []
        M = len(dataloader_train)
        # M = len(x_train)
        for i, (x, y) in enumerate(dataloader_train):
            optimizer.zero_grad()
            loss = model.sample_elbo(
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

        mse_val = log_metrics(
            writer,
            model,
            (x_train, y_train, x_val, y_val, x_all, y_all),
            loss_history,
            epoch,
        )
        trial.report(mse_val, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return mse_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, help="resume existing study", default="")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]

    if args.study == "":
        # create study
        study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(direction="minimize", study_name=study_name)
        # save source files
        os.makedirs(project_dir / f"runs/{study_name}/", exist_ok=True)
        for f in glob.iglob(str(project_dir) + "/src/*.py"):
            shutil.copy2(f, project_dir / f"runs/{study_name}/")
    else:
        # resume study
        study = joblib.load(project_dir / f"runs/{args.study}/study.pkl")
        study_name = args.study

    study.optimize(objective, n_trials=100)
    joblib.dump(study, project_dir / f"runs/{study_name}/study.pkl")