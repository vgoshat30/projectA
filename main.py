"""Main file of projectA

    Trains and tests:
    -- Sign quantization Linear network
    -- Uniform quantization Linear network
    -- Sign qunatization Linear + RNN network
    -- Sign qunatization Linear + LSTM network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np

from dataLoader import *
import linearModels
import RNNmodels
from projectConstants import *
import UniformQuantizer


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


def trainAnalogDigital(epoch, modelAnalog, modelDigital,
                       optimizerAnalog, optimizerDigital, codebook):
    modelAnalog.train()
    modelDigital.train()

    # Train analog NN
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = Variable(data.float()), Variable(target.float())

        optimizerAnalog.zero_grad()
        output = modelAnalog(data)
        lossAnalog = criterion(output.view(-1, 1), target.view(-1, 1))
        lossAnalog.backward(retain_graph=True)
        optimizerAnalog.step()

        if batch_idx % 10 == 0:
            print('Analog: Epoch: {} [{}/{} ({:.0f}%)]\tLinear Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), lossAnalog))

    # Train digital NN
    for batch_idx, (data, target) in enumerate(trainLoader):

        target = Variable(target.float())

        analogProcessData = modelAnalog(data.float())
        quantizedData = Variable(justQuantize(analogProcessData, codebook))

        optimizerDigital.zero_grad()
        output = modelDigital(quantizedData)
        lossDigital = criterion(output.view(-1, 1), target.view(-1, 1))
        lossDigital.backward(retain_graph=True)
        optimizerDigital.step()

        if batch_idx % 10 == 0:
            print('Digital: Epoch: {} [{}/{} ({:.0f}%)]\tLinear Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), lossDigital))


def trainWithQuantizer(epoch, model, optimizer, codebook):
    model.train()
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = Variable(data.float()), Variable(target.float())

        optimizer.zero_grad()
        output = model(data, codebook)
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
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def testWithQuantizer(model, codebook):
    model.eval()
    test_loss = 0

    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data, codebook)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))

    test_loss /= (len(test_loader.dataset)/BATCH_SIZE)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def justQuantize(input, codebook):
    input_data = input.data
    input_numpy = input_data.numpy()
    qunatized_input = torch.zeros(input.size())
    for ii in range(0, input_data.size(0)):
        for jj in range(0, input_data.size(1)):
            qunatized_input[ii][jj] = UniformQuantizer.get_optimal_word(input_numpy[ii, jj],
                                                       codebook)
    return qunatized_input


# Get the class containing the train data from dataLoader.py
trainData = ShlezDatasetTrain()
# define training dataloader
trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# Do the same for the test data
testData = ShlezDatasetTest()
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=True)

# Generate uniform code book using the variance of the tain data
Quantization_codebook = UniformQuantizer.codebook_uniform(trainData.X_var, M)

# model_lin1: Basic linear network with sign activation as the quantization
model_lin1 = linearModels.SignQuantizerNet()
# model_lin2: Basic linear network with uniform quantization instead of sign
model_lin2 = linearModels.UniformQuantizerNet(Quantization_codebook)
# model_lin3: Basic linear network which learns to prepare the analog signal
# data for quantization
model_lin3 = linearModels.AnalogProcessNet()
# model_lin4: Basic linear network which learns to perform the digital
# processing after the quantization and results the channel coefficients
model_lin4 = linearModels.DigitalProcessNet()
# model_RNN1: Basic linear network with sign activation and pre-quantization
# RNN layer
model_RNN1 = RNNmodels.SignQuantizerNetRNN()
# model_RNN2: Basic linear network with sign activation and pre-quantization
# LSTM layer
model_RNN2 = RNNmodels.SignQuantizerNetLSTM()


criterion = nn.MSELoss()
optimizer_lin1 = optim.SGD(model_lin1.parameters(), lr=0.01, momentum=0.5)
optimizer_lin2 = optim.SGD(model_lin2.parameters(), lr=0.01, momentum=0.5)
optimizer_lin3 = optim.SGD(model_lin3.parameters(), lr=0.01, momentum=0.5)
optimizer_lin4 = optim.SGD(model_lin4.parameters(), lr=0.01, momentum=0.5)
optimizer_RNN1 = optim.SGD(model_RNN1.parameters(), lr=0.01, momentum=0.5)
optimizer_RNN2 = optim.SGD(model_RNN2.parameters(), lr=0.01, momentum=0.5)


print('\n\nTRAINING...')
for epoch in range(0, EPOCHS):
    print('Training Linear sign quantization model:')
    trainAnalogDigital(epoch, model_lin3, model_lin4, optimizer_lin3,
                       optimizer_lin4, Quantization_codebook)
    print('Training Linear sign quantization model:')
    train(epoch, model_lin1, optimizer_lin1)
    print('Training Linear uniform quantization model:')
    train(epoch, model_lin2, optimizer_lin2)
    print('Training RNN sign quantization model:')
    train(epoch, model_RNN1, optimizer_RNN1)
    print('Training LSTM sign quantization model:')
    train(epoch, model_RNN1, optimizer_RNN1)


print('Quantization Rate: {:.3f}'.format(QUANTIZATION_RATE))
print('\n\nTESTING...')
print('Testing Linear sign quantization model:')
test(model_lin1)
print('Testing Linear uniform quantization model:')
test(model_lin2, Quantization_codebook)
print('Testing RNN sign quantization model:')
test(model_RNN1)
print('Testing LSTM sign quantization model:')
test(model_RNN2)
