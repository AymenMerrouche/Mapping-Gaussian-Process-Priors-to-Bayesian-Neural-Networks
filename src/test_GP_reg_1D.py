import math
import datetime
import yaml
from matplotlib import pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

from sklearn.metrics import mean_squared_error


import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

from models import *
from datasets import *
from figures import *
from kernels import *

import warnings
warnings.filterwarnings("ignore")


class ExactGP_Regression(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGP_Regression, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def sample_gpp(model ,sampling_points, T):
    model.eval()
    likelihood.eval()
    # sample functions
    preds = model(sampling_points)
    y = torch.stack([preds.sample() for i in range(T)])
    return y

def fit(model, likelihood, optimizer, criterion ,train_x, train_y, val_x, val_y, epochs, writer=None):
    """Full training loop"""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    def train_epoch():
        """
        Returns:
            The epoch loss
        """
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(model.train_inputs[0])
        # Calc loss and backprop gradients
        loss = -criterion(output, model.train_targets)
        loss.backward()
        optimizer.step()
        return loss.item(), model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()
    
    def get_metrics(y_true, y_pred):
        """
        Returns MSE score
        """
        return mean_squared_error(y_true, y_pred)
    
    def evaluate_epoch(x, y, role='Val'):
        """
        Args:
            loader (torch.utils.data.DataLoader): either the train of validation loader
            role (str): either 'Val' or 'Train'
        Returns:
            Tuple containing mean loss and accuracy
        """
        model.eval()
        likelihood.eval()
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Test points are regularly spaced along [0,1]
            preds = likelihood(model(x)).loc

        return get_metrics(y, preds)



    for epoch in range(0, epochs):
        loss, lengthscale, noise = train_epoch()
        mse_train = evaluate_epoch(x_train, y_train, 'Train')
        mse_test =  evaluate_epoch(x_val, y_val, 'Val')
        

        print(f"Epoch {epoch}/{epochs}, Train Loss: {loss:.4f}, Train Noise: {noise:.4f}, Train Lengthscale: {lengthscale:.4f}")
        print(f"Epoch {epoch}/{epochs}, Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")

        if writer:
            writer.add_scalars("Train Loss",loss , epoch)
            writer.add_scalars("Train Noise",noise , epoch)
            writer.add_scalars("Train Lengthscale",lengthscale , epoch)
            writer.add_scalars("Mse", {"Train": mse_train, "Test" : mse_test}, epoch)
    print("\nFinished.")
    
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load args

    # data paths args
    with open('../configs/gp_reg_1D.yaml', 'r') as stream:
        gp_reg_args  = yaml.load(stream,Loader=yaml.Loader)
    if gp_reg_args["experiment"] == True : 
        for func in function_dict:
            # load the data
            x_train, y_train, x_val, y_val, x_all, y_all = get_1D_data_from_func(150, sigma=0.05, f = function_dict[func])
            x_train, y_train, x_val, y_val = x_train.squeeze(-1).to(device), y_train.squeeze(-1).to(device), x_val.squeeze(-1).to(device), y_val.squeeze(-1).to(device)

            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGP_Regression(x_train, y_train, likelihood, kernel = gpytorch.kernels.RBFKernel()).to(device)



           # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Logging + Experiment

            ignore_keys = {'no_tensorboard'}
            # get hyperparameters with values in a dict
            hparams = {**gp_reg_args}
            # generate a name for the experiment
            expe_name = '_'.join([f"{key}={val}" for key, val in hparams.items()])
            print("Experimenting with : \n \t"+expe_name)
            # Tensorboard summary writer
            if gp_reg_args['no_tensorboard']:
                writer = None
            else:
                writer = SummaryWriter("runs/runs"+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+expe_name)
            # start the experiment
            fit(model, likelihood, optimizer, mll, x_train, y_train, x_val, y_val, gp_reg_args['epochs'], writer=writer)

            if not gp_reg_args['no_tensorboard']:
                writer.close()
                
            # create and save figure
            x_all = x_all.squeeze(-1)
            y_all_pred = sample_gpp(model ,x_all, 5).cpu().numpy().T
            x_train, y_train, x_val, y_val, x_all, y_all = x_train.cpu().numpy(), y_train.cpu().numpy(), x_val.cpu().numpy(),\
            y_val.cpu().numpy(), x_all.cpu().numpy(), y_all.cpu().numpy()
            title = "regression results on "+func
            plot_deciles(x_all, y_all_pred, y_all, x_train, y_train, mode="gp", title=title)
    if gp_reg_args["sample"] == True :
        # get some sampling points
        sampling_points = torch.linspace(-10, 10, 1000)
        # data that will not be used
        # ficticious data
        train_x = torch.linspace(0, 1, 100)
        train_y = torch.sin(train_x * (100 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
        for kernel in all_kernels:
            # initialize likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # initialise model
            model = ExactGP_Regression(train_x, train_y, likelihood, kernel = all_kernels[kernel])
            model.eval()
            likelihood.eval()
            # None to train inputs to sample from prior
            model.train_inputs = None
            # get samples
            # sample functions
            preds = model(sampling_points)
            y = torch.stack([preds.sample() for i in range(5)]).cpu().numpy().T
            # draw from priors
            # title of the plot
            title = "prior with kernel = "+kernel
            plot_prior(sampling_points, y, title=title)
        
        