"""Creating RNN Classes

    Creating a class for RNN

    Returns
    -------
    SignQuantizerNetRNN

        A class containing the feed forward function: forward(self, x) with RNN
        layer and sign quantization

    SignQuantizerNetLSTM

        A class containing the feed forward function: forward(self, x) with LSTM
        layer and sign quantization
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


class SignQuantizerNetRNN(nn.Module):

    def __init__(self):
        super(SignQuantizerNetRNN, self).__init__()

        # One cell RNN input_dim (240) -> output_dim (80).sequance: 1
        self.rnn = nn.RNN(input_size=INPUT_DIMENSION, hidden_size=OUTPUT_DIMENSION)

        # hidden : (num_layers * num_directions, batch, hidden_size)
        # whether batch_first=True or False
        self.hidden = Variable(torch.randn(1, BATCH_SIZE, OUTPUT_DIMENSION))
        self.l1 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l2 = nn.Linear(240, OUTPUT_DIMENSION)    # See Hardware-Limited Task-Based
        # Quantization Proposion 3. for the
        # choice of output features
        self.l3 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, OUTPUT_DIMENSION)
        self.q1 = LearningQuantizer.signActivation.apply

    def forward(self, x):
        x = x.view(1, BATCH_SIZE, INPUT_DIMENSION)
        x, self.hidden = self.rnn(x, self.hidden)
        x = x.view(BATCH_SIZE, OUTPUT_DIMENSION)
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class SignQuantizerNetLSTM(nn.Module):

    def __init__(self):
        super(SignQuantizerNetLSTM, self).__init__()

        # LSTM input_dim (240) -> output_dim (80).sequance: 1
        self.lstm = nn.LSTM(input_size=INPUT_DIMENSION, hidden_size=OUTPUT_DIMENSION)

        # hidden : (num_layers * num_directions, batch, hidden_size)
        # whether batch_first=True or False
        self.hidden = Variable(torch.randn(1, BATCH_SIZE, OUTPUT_DIMENSION))
        # cell : (num_layers * num_directions, batch, hidden_size)
        self.cell = Variable(torch.randn(1, BATCH_SIZE, OUTPUT_DIMENSION))
        self.l1 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l2 = nn.Linear(240, OUTPUT_DIMENSION)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l3 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, OUTPUT_DIMENSION)
        self.q1 = LearningQuantizer.signActivation.apply

    def forward(self, x):
        x = x.view(1, BATCH_SIZE, INPUT_DIMENSION)
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        x = x.view(BATCH_SIZE, OUTPUT_DIMENSION)
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)
