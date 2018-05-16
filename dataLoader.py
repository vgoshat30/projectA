"""Converting data from MATLAB .mat file to torch tensors

    Searches for 'shlezingerMat.mat' file, extracts the variables 'trainX'
    'trainS' 'dataX' 'dataS' variables and returning two classes, containing the
    train and test data.

    Returns
    -------
    ShlezDatasetTrain
        A class containing the train data

    ShlezDatasetTest
        A class containing the test data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np


class ShlezDatasetTrain(Dataset):
    """ Data class for the training data set (X and S pairs) """

    def __init__(self):
        # Loading .mat file
        shlezMat = sio.loadmat('shlezingerMat.mat')
        # Getting train data variables from the .mat file:
        Xdata = shlezMat['trainX']
        Sdata = shlezMat['trainS']

        # Converting numpy arrays to pytorch tensors:
        self.X_data = torch.from_numpy(Xdata)
        self.S_data = torch.from_numpy(Sdata)

        # Number of X, S couples:
        self.len = Sdata.shape[0]

    def __getitem__(self, index):
        return self.X_data[index], self.S_data[index]

    def __len__(self):
        return self.len


class ShlezDatasetTest(Dataset):
    """ Data class for the testing data set (X and S pairs) """

    def __init__(self):
        # Loading .mat file
        shlezMat = sio.loadmat('shlezingerMat.mat')
        # Getting test data variables from the .mat file:
        Xdata = shlezMat['dataX']
        Sdata = shlezMat['dataS']

        # Converting numpy arrays to pytorch tensors:
        self.X_data = torch.from_numpy(Xdata)
        self.S_data = torch.from_numpy(Sdata)

        # Number of X, S couples:
        self.len = Sdata.shape[0]

    def __getitem__(self, index):
        return self.X_data[index], self.S_data[index]

    def __len__(self):
        return self.len
