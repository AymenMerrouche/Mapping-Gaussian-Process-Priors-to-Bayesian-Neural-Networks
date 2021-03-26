from torch.utils.data import Dataset
import torch


class UniformDistributionDataset(Dataset):
    """
    Sample n_samples samples from unifom distribution in an interval [lower_bound, upper_bound].
    """

    def __init__(self, lower_bound=-10, upper_bound=-10, n_samples=1000):
        
        self.n_samples = n_samples
        self.data = torch.FloatTensor(n_samples, 1).uniform_(lower_bound, upper_bound)
    def __len__(self):
        return self.n_samples

    def __getitem__(self, ix):
        return self.data[ix], self.data[ix]