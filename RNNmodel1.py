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

        # hidden : (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
        self.hidden = Variable(torch.randn(1, BATCH_SIZE, 80))
        self.l1 = nn.Linear(80, 240)
        self.l2 = nn.Linear(240, 80)    # See Hardware-Limited Task-Based
        # Quantization Proposion 3. for the
        # choice of output features
        self.l3 = nn.Linear(80, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 80)

    def forward(self, x):   # x rank (seq_len = 1, batch = BATCH_SIZE, input_size = 240)
        x = x.view(1, BATCH_SIZE, 240)
        x, self.hidden = self.rnn(x, self.hidden)
        x = self.l1(x)
        x = self.l2(x)
        x = torch.sign(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


'''
def train(epoch):
    RNN_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.float()), Variable(target.float())

        optimizer_RNN.zero_grad()
        data_RNN_model = data.view(1, BATCH_SIZE, 240)
        output_RNN_model = RNN_model(data_RNN_model)
        loss_RNN_model = criterion(output_RNN_model.view(-1, 1), target.view(-1, 1))
        loss_RNN_model.backward()
        optimizer_RNN.step()

        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tRNN Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_RNN_model))


def test():
    RNN_model.eval()
    test_RNN_loss = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.float()), Variable(target.float())

        data_RNN_model = data.view(1, BATCH_SIZE, 240)
        output_RNN_model = RNN_model(data_RNN_model)
        # sum up batch RNN loss
        test_RNN_loss += criterion(output_RNN_model.view(-1, 1), target.view(-1, 1))

    test_RNN_loss /= (len(test_loader.dataset)/BATCH_SIZE)

    print('\nTest set: Average RNN loss: {:.4f}\n'.format(test_RNN_loss))
'''
