import argparse
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import optuna
import torch
from torch.utils.data import TensorDataset, DataLoader

from datasets import get_toy_data
from models import BayesianMLP
from train_inference import train, eval_1d_regression


def objective(trial: optuna.Trial):
    dim_h = 20
    activation = "rbf"
    sigma_model = 0.1
    num_samples = 70
    sigma_prior = trial.suggest_float("prior_sigma", low=1e-3, high=10.0, log=True)
    # M = trial.suggest_int("M", low=1, high=200, log=True)
    M = 70
    model_noise_var = trial.suggest_float(
        "model_noise_var", low=0.01, high=1.0, log=True
    )

    log_dir = (
        project_dir
        / f"runs/{trial.study.study_name}/trial_{trial.number}-dim_h_{dim_h}-act_{activation}-"
        f"sigma_{sigma_prior:.2f}"
    )

    x_train, y_train, x_val, y_val, x_all, y_all = get_toy_data(
        num_samples=num_samples, sigma=sigma_model
    )
    dataloader_train = DataLoader(
        TensorDataset(x_train, y_train), batch_size=num_samples, shuffle=True
    )

    model = BayesianMLP(
        dim_in=1,
        dim_out=1,
        dim_h=dim_h,
        prior_sigma=sigma_prior,
        activation=activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    return train(
        model,
        optimizer,
        dataloader_train,
        n_epochs=150,
        log_dir=log_dir,
        evaluate_func=eval_1d_regression,
        evaluate_data=(x_train, y_train, x_val, y_val, x_all, y_all, 50),
        model_noise_var=model_noise_var,
        M=M,
    )


def make_hyper_param_study():
    if args.study == "":
        # create study
        study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            # sampler=optuna.samplers.RandomSampler(),
        )
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


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, help="resume existing study", default="")
    args = parser.parse_args()
    make_hyper_param_study()