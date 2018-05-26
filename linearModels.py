"""Creating Simple NN Class

    Creating a class for the NN

    Returns
    -------
    SignQuantizerNet
        A class containing the feed forward function: forward(self, x) and
        sign quantization
    UniformQuantizerNet
        A class containing the feed forward function: forward(self, x) and
        uniform quantization
    SOMQuantizerNet
        A class containing the feed forward function: forward(self, x) and
        learning SOM quantization
    AnalogProcessNet + DigitalProcessNet
        Two classes responsible for the quantization process using two distinct
        NN for the analog and the digital processes

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np

import UniformQuantizer
import LearningQuantizer
from projectConstants import *


class SignQuantizerNet(nn.Module):

    def __init__(self):
        super(SignQuantizerNet, self).__init__()
        self.l1 = nn.Linear(INPUT_DIMENSION, 520)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l2 = nn.Linear(520, OUTPUT_DIMENSION)
        self.l3 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, OUTPUT_DIMENSION)
        self.q1 = LearningQuantizer.signActivation.apply

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class UniformQuantizerNet(nn.Module):

    def __init__(self, codebook):
        super(UniformQuantizerNet, self).__init__()
        self.l1 = nn.Linear(INPUT_DIMENSION, 520)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l2 = nn.Linear(520, OUTPUT_DIMENSION)
        self.l3 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, OUTPUT_DIMENSION)
        self.q1 = LearningQuantizer.QuantizationFunction.apply
        self.codebook = codebook

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x, self.codebook)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)

class SOMQuantizerNet(nn.Module):

    def __init__(self, codebook):
        super(SOMQuantizerNet, self).__init__()
        self.l1 = nn.Linear(INPUT_DIMENSION, 520)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l2 = nn.Linear(520, OUTPUT_DIMENSION)
        self.l3 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, OUTPUT_DIMENSION)
        self.q1 = LearningQuantizer.QuantizationFunction.apply
        self.codebook = codebook
        # self.testCodebook = Variable(self.testCodebook.float())
        # self.testCodebook = torch.from_numpy(np.asarray(codebook))
        self.testCodebook = codebook

    def forward(self, x):
        # update the learning codebook
        x_numpy = x.data.numpy()
        newCodebook = list(self.testCodebook)
        for ii in range(0, x.size(0)):
            for jj in range(0, x.size(1)):
                itrVal, quantized_idx = UniformQuantizer.get_optimal_word(
                    x_numpy[ii, jj], tuple(newCodebook))
                # update winner codeword
                newCodebook[quantized_idx] = newCodebook[quantized_idx] + CODEBOOK_LR*(x_numpy[ii, jj] - itrVal)
        self.testCodebook = tuple(newCodebook)

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
        self.l1 = nn.Linear(INPUT_DIMENSION, 520)
        self.l2 = nn.Linear(520, 80)
        self.l3 = nn.Linear(80, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, OUTPUT_DIMENSION)

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
        self.l1 = nn.Linear(OUTPUT_DIMENSION, 340)
        self.l2 = nn.Linear(340, 100)
        self.l3 = nn.Linear(100, 80)
        self.l4 = nn.Linear(80, 300)
        self.l5 = nn.Linear(300, OUTPUT_DIMENSION)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)
