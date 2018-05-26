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

from projectConstants import *


class SignQuantizerNetRNN(nn.Module):

    def __init__(self):
        super(SignQuantizerNetRNN, self).__init__()

        # One cell RNN input_dim (240) -> output_dim (80).sequance: 1
        self.rnn = nn.RNN(input_size=240, hidden_size=80)

        # hidden : (num_layers * num_directions, batch, hidden_size)
        # whether batch_first=True or False
        self.hidden = Variable(torch.randn(1, BATCH_SIZE, 80))
        self.l1 = nn.Linear(80, 240)
        self.l2 = nn.Linear(240, 80)    # See Hardware-Limited Task-Based
        # Quantization Proposion 3. for the
        # choice of output features
        self.l3 = nn.Linear(80, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 80)

    def forward(self, x):
        """
            The feed forward
            x rank (seq_len = 1, batch = BATCH_SIZE, input_size = 240)
        """
        x = x.view(1, BATCH_SIZE, 240)
        x, self.hidden = self.rnn(x, self.hidden)
        x = x.view(BATCH_SIZE, 80)
        x = self.l1(x)
        x = self.l2(x)
        x = torch.sign(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class SignQuantizerNetLSTM(nn.Module):

    def __init__(self):
        super(SignQuantizerNetLSTM, self).__init__()

        # LSTM input_dim (240) -> output_dim (80).sequance: 1
        self.lstm = nn.LSTM(input_size=240, hidden_size=80)

        # hidden : (num_layers * num_directions, batch, hidden_size)
        # whether batch_first=True or False
        self.hidden = Variable(torch.randn(1, BATCH_SIZE, 80))
        # cell : (num_layers * num_directions, batch, hidden_size)
        self.cell = Variable(torch.randn(1, BATCH_SIZE, 80))
        self.l1 = nn.Linear(80, 240)
        self.l2 = nn.Linear(240, 80)    # See Hardware-Limited Task-Based
        # Quantization Proposion 3. for the
        # choice of output features
        self.l3 = nn.Linear(80, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 80)

    def forward(self, x):
        """
            The feed forward
            x rank (seq_len = 1, batch = BATCH_SIZE, input_size = 240)
        """
        x = x.view(1, BATCH_SIZE, 240)
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        x = x.view(BATCH_SIZE, 80)
        x = self.l1(x)
        x = self.l2(x)
        x = torch.sign(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)
