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
    dim_h = 512
    prior_sigma_1 = trial.suggest_float("prior_sigma_1", low=1e-3, high=10.0, log=True)
    prior_sigma_2 = trial.suggest_float("prior_sigma_2", low=1e-3, high=10.0, log=True)
    prior_pi = trial.suggest_float("prior_pi", low=0.0, high=1.0)

    log_dir = (
        project_dir
        / f"runs/{trial.study.study_name}/trial_{trial.number}-dim_h_{dim_h}-"
        f"s1_{prior_sigma_1:.2f}-s2_{prior_sigma_2:.2f}-pi_{prior_pi:.2f}"
    )

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
    return train(
        model,
        optimizer,
        dataloader_train,
        n_epochs=500,
        log_dir=log_dir,
        evaluate_func=eval_1d_regression,
        evaluate_data=(x_train, y_train, x_val, y_val, x_all, y_all),
    )


def make_hyper_param_study():
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


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, help="resume existing study", default="")
    args = parser.parse_args()
