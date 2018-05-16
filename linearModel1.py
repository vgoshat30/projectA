import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np

from projectConstants import *


class SignQuantizerNet(nn.Module):

    def __init__(self):
        super(SignQuantizerNet, self).__init__()
        self.l1 = nn.Linear(240, 520)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the choice of output features
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


'''
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.float()), Variable(target.float())

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 1), target.view(-1, 1))
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLinear Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)), loss)


def test():
    model.eval()
    test_loss = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))

    test_loss /= (len(test_loader.dataset)/BATCH_SIZE)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))
'''
