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
from UniformQuantizer import *


def train(epoch, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = Variable(data.float()), Variable(target.float())

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 1), target.view(-1, 1))
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLinear Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss))


def test(model):
    model.eval()
    test_loss = 0

    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))

    test_loss /= (len(test_loader.dataset)/BATCH_SIZE)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))


# Get the class containing the train data from dataLoader.py
trainData = ShlezDatasetTrain()
# define training dataloader
trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# Do the same for the test data
testData = ShlezDatasetTest()
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=True)

# Generate uniform code book using the variance of the tain data
Quantization_codebook = codebook_uniform(trainData.X_var, M)


model_lin1 = linearModel1.SignQuantizerNet()
model_lin2 = linearModel1.UniformQuantizerNet(Quantization_codebook)
model_RNN1 = RNNmodel1.SignQuantizerNetRNN()

criterion = nn.MSELoss()
optimizer_lin1 = optim.SGD(model_lin1.parameters(), lr=0.01, momentum=0.5)
optimizer_lin2 = optim.SGD(model_lin2.parameters(), lr=0.01, momentum=0.5)
optimizer_RNN1 = optim.SGD(model_RNN1.parameters(), lr=0.01, momentum=0.5)

print('\n\nTRAINING...')
for epoch in range(0, EPOCHS):
    print('Training Linear model:')
    train(epoch, model_lin1, optimizer_lin1)
    print('Training Linear model:')
    train(epoch, model_lin2, optimizer_lin2)
    print('Training RNN model:')
    train(epoch, model_RNN1, optimizer_RNN1)

print('\n\nTESTING...')
print('Testing Linear model:')
test(model_lin1)
print('Testing Linear model:')
test(model_lin2)
print('Testing RNN model:')
test(model_RNN1)
