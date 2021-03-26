from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np



def get_data_loaders(dataset, batch_size_train = 128, batch_size_validation = 128, no_batching_for_validation = False ,test_proportion = 0.33):
    """
    Function that returns train and validation dataloaders.
    :param dataset: torch.utils.data.Dataset object.
    :param batch_size_train: batch size for the train loader.
    :param batch_size_validation: batch size for the test loader.
    :param no_batching_for_validation: if true then batch size for validation loader equals the length of the validation set (only one batch),
    otherwise use batch_size_validation.
    :param test_proportion: proportion of examples to use in the validation set.
    """
    
        
    # test train split
    len_validation = np.round(test_proportion*len(dataset))
    len_train = len(dataset)-len_validation
    dataset_train, dataset_validation = torch.utils.data.random_split(dataset, [int(len_train), int(len_validation)])
    if no_batching_for_validation :
        batch_size_validation = int(len_validation)
    # dataloaders
    # step 2: define the dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train,shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation,shuffle=False)
    return dataloader_train, dataloader_validation