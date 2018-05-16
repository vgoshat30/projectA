"""Main file of projectA

    Trains and tests 1 bit quantization network and a RNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np

from dataLoader import *
import linearModel1
import RNNmodel1
from projectConstants import *
from UniformQuantizer import generate_codebook

def train(epoch, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.float()), Variable(target.float())

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 1), target.view(-1, 1))
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLinear Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(model):
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


datasetTrainLoader = ShlezDatasetTrain()
train_loader = DataLoader(dataset=datasetTrainLoader, batch_size=BATCH_SIZE, shuffle=True)
datasetTestLoader = ShlezDatasetTest()
test_loader = DataLoader(dataset=datasetTestLoader, batch_size=BATCH_SIZE, shuffle=True)
Quantization_codebook = generate_codebook(datasetTrainLoader.X_data_variance, M)


model = linearModel1.SignQuantizerNet()
RNN_model = RNNmodel1.SignQuantizerNetRNN()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer_RNN = optim.SGD(RNN_model.parameters(), lr=0.01, momentum=0.5)

print('\n\nTRAINING...')
for epoch in range(0, EPOCHS):
    print('Training Linear model:')
    train(epoch, model, optimizer)
    print('Training RNN model:')
    train(epoch, RNN_model, optimizer_RNN)

print('\n\nTESTING...')
print('Testing Linear model:')
test(model)
print('Testing RNN model:')
test(RNN_model)
