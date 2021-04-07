import math
import datetime
import yaml
from matplotlib import pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

from models import *

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_classif_data(num_samples=100, dimension = 10, num_classes=2, class_sep=0.5, test_size=0.33):
    """
    Generate a random n-class classification problem.

    Args:
        num_samples:
        dimension:
        num_classes:
        class_sep:
        test_size:

    Returns:
        x_train, y_train, x_val, y_val, x_all, y_all
    """
    # make the isotropic gaussian blobs
    x_all, y_all = make_classification(n_samples=num_samples, n_features=dimension, n_repeated=0, class_sep=class_sep, n_redundant=0, random_state=42)

    # trai test split
    x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=test_size, random_state=42)

    return torch.tensor(x_train).float(), torch.tensor(y_train), torch.tensor(x_val).float(), \
    torch.tensor(y_val), torch.tensor(x_all).float(), torch.tensor(y_all)

def fit(model, likelihood, optimizer, criterion ,train_loader, val_loader, epochs, writer=None):
    """Full training loop"""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    min_loss = float('inf')
    iteration = 1
    best_acc = float('-inf')
    def train_epoch():
        """
        Returns:
            The epoch loss
        """
        nonlocal iteration
        epoch_loss = 0.
        epoch_lengthscale = 0.
        epoch_noise = 0.
        model.train()
        likelihood.train()
        for (i, batch) in enumerate(train_loader):
            data, labels = batch
            data, labels = data.to(device), labels.to(device).long()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_train)
            # Calc loss and backprop gradients
            loss = -criterion(output, labels).sum()
            if writer:
                writer.add_scalar('Iteration_loss', loss.item(), iteration)
                writer.add_scalar('LengthScale', model.covar_module.base_kernel.lengthscale.mean().item(), iteration)
                writer.add_scalar('Noise', model.likelihood.second_noise_covar.noise.mean().item(), iteration)
            loss.backward()
            optimizer.step()
            iteration += 1
            del data
            del labels
            torch.cuda.empty_cache()
        epoch_loss /= len(train_loader)
        epoch_lengthscale /= len(train_loader)
        epoch_noise /= len(train_loader)
        return epoch_loss, epoch_lengthscale, epoch_noise
    
    def get_metrics(y_true, logits):
        """
        Returns accuracy, auc score and classification report (precsision, recall and f1 score)
        """
        y_pred = logits.max(0)[1]
        return accuracy_score(y_true, y_pred), classification_report(y_true, y_pred),
    
    def evaluate_epoch(loader, role='Val'):
        """
        Args:
            loader (torch.utils.data.DataLoader): either the train of validation loader
            role (str): either 'Val' or 'Train'
        Returns:
            Tuple containing mean loss and accuracy
        """
        model.eval()
        likelihood.eval()
    
        correct = 0
        mean_loss = 0.
        lebels_stacked = []
        probs_stacked = []
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            for batch in loader:
                # get the data
                data, labels = batch
                data, labels = data.to(device), labels.to(device).long()
                # get the output distribution
                pred_dist = model(data)
                # get logits (means)
                pred_means = pred_dist.loc
                # compute loss
                loss = -criterion(pred_dist, labels).sum()
                mean_loss += loss.item()
                
                probs_stacked.append(pred_means)
                lebels_stacked.append(labels)
                
        targets = torch.cat(lebels_stacked, 0)
        probs = torch.cat(probs_stacked, 0)

        return mean_loss/len(loader), get_metrics(targets, probs)


    for epoch in range(0, epochs):
        train_epoch()
        loss_train, (acc_train, _) = evaluate_epoch(train_loader, 'Train')
        loss_test, (acc_test, clf_report_test) =  evaluate_epoch(val_loader, 'Val')
        

        print(f"Epoch {epoch}/{epochs}, Train Loss: {loss_train:.4e}, Test Loss: {loss_test:.4f}")
        print(f"Epoch {epoch}/{epochs}, Train Accuracy: {acc_train*100:.2f}%, Test Accuracy: {acc_test*100:.2f}%")
        print("Classification Report on Val Set : ")
        print(clf_report_test)
        if writer:
            writer.add_scalars("Loss", {"Train": loss_train, "Test" : loss_test}, epoch)
            writer.add_scalars("Accuracy", {"Train": acc_train*100, "Test" : acc_test*100}, epoch)
        if acc_test > best_acc:
            best_acc = acc_test
    print("\nFinished.")
    print(f"With accuracy: {best_acc}")
    
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load args

    # data paths args
    with open('../configs/gp_classif.yaml', 'r') as stream:
        gp_classif_args  = yaml.load(stream,Loader=yaml.Loader)

    # load the data
    x_train, y_train, x_val, y_val,_,_ = get_classif_data(num_samples = gp_classif_args["num_examples"], dimension=gp_classif_args["dim"], num_classes=gp_classif_args["num_classes"])
    x_train, y_train, x_val, y_val = x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device)
    # get dataloaders
    train_loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=x_train.shape[0],shuffle=True)
    test_loader = DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=x_val.shape[0],shuffle=False)


    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = DirichletClassificationLikelihood(y_train.long(), learn_additional_noise=True).to(device)
    model = DirichletGPModel(x_train, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes).to(device)

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
    hparams = {**gp_classif_args}
    # generate a name for the experiment
    expe_name = '_'.join([f"{key}={val}" for key, val in hparams.items()])
    print("Experimenting with : \n \t"+expe_name)
    # Tensorboard summary writer
    if gp_classif_args['no_tensorboard']:
        writer = None
    else:
        writer = SummaryWriter("runs/runs"+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+expe_name)
    # start the experiment
    fit(model, likelihood, optimizer, mll, train_loader, test_loader, gp_classif_args['epochs'], writer=writer)

    if not gp_classif_args['no_tensorboard']:
        writer.close()