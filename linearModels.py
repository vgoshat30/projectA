"""Creating Simple NN Class

    Creating a class for the NN

    Returns
    -------
    SignQuantizerNet
        A class containing the feed forward function: forward(self, x) and sign quantization
    UniformQuantizerNet
        A class containing the feed forward function: forward(self, x) and uniform quantization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np

import LearningQuantizer
from projectConstants import *


class SignQuantizerNet(nn.Module):

    def __init__(self):
        super(SignQuantizerNet, self).__init__()
        self.l1 = nn.Linear(240, 520)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l2 = nn.Linear(520, 80)
        self.l3 = nn.Linear(80, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 80)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = torch.sign(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class UniformQuantizerNet(nn.Module):

    def __init__(self, codebook):
        super(UniformQuantizerNet, self).__init__()
        self.l1 = nn.Linear(240, 520)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l2 = nn.Linear(520, 80)
        self.l3 = nn.Linear(80, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 80)
        self.q1 = LearningQuantizer.QuantizationFunction.apply
        self.codebook = codebook

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x, self.codebook)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class AnalogProcessNet(nn.Module):
    '''A NN performind the analog pre-processing before the quantizers

        Learns to minimize the distance between the resulting channel
        coefficients ("S Hat") and the originial S. Meant to be trained
        separately from the NN after the quantization

        Args
        ----
            s_len
                Size of features vector (channel coefficients)
    '''

    def __init__(self):
        super(AnalogProcessNet, self).__init__()
        self.l1 = nn.Linear(240, 520)
        self.l2 = nn.Linear(520, 80)
        self.l3 = nn.Linear(80, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 80)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class DigitalProcessNet(nn.Module):
    '''A NN performing the digital processing before the quantizers

        Meant to be trained separately from the NN before the quantization

        Args
        ----
            s_len
                Size of features vector (channel coefficients)
    '''

    def __init__(self):
        super(DigitalProcessNet, self).__init__()
        self.l1 = nn.Linear(80, 340)
        self.l2 = nn.Linear(340, 100)
        self.l3 = nn.Linear(100, 80)
        self.l4 = nn.Linear(80, 300)
        self.l5 = nn.Linear(300, 80)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)
