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
import testLogger


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

    modelAnalog.eval()
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


def test(model):
    model.eval()
    test_loss = 0

    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))

    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)

    return test_loss.detach().numpy()


def justQuantize(input, codebook):
    input_data = input.data
    input_numpy = input_data.numpy()
    qunatized_input = torch.zeros(input.size())
    for ii in range(0, input_data.size(0)):
        for jj in range(0, input_data.size(1)):
            qunatized_input[ii][jj], __ = UniformQuantizer.get_optimal_word(
                input_numpy[ii, jj], codebook)
    return qunatized_input


# Get the class containing the train data from dataLoader.py
trainData = ShlezDatasetTrain()
# define training dataloader
trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# Do the same for the test data
testData = ShlezDatasetTest()
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=True)

# Generate uniform codebooks
X_codebook = UniformQuantizer.codebook_uniform(trainData.X_var, M)
S_codebook = UniformQuantizer.codebook_uniform(trainData.S_var, M)
# model_lin1: Basic linear network with sign activation as the quantization
model_lin1 = linearModels.SignQuantizerNet()
# model_lin2: Basic linear network with uniform quantization instead of sign
model_lin2 = linearModels.UniformQuantizerNet(S_codebook)
# model_lin3: Basic linear network which learns to prepare the analog signal
# data for quantization
model_lin3 = linearModels.AnalogProcessNet()
# model_lin4: Basic linear network which learns to perform the digital
# processing after the quantization and results the channel coefficients
model_lin4 = linearModels.DigitalProcessNet()
# model_lin5: Basic linear network with SOM learning quantization instead of sign
model_lin5 = linearModels.SOMQuantizerNet(S_codebook)
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
optimizer_lin5 = optim.SGD(model_lin5.parameters(), lr=0.01, momentum=0.5)
optimizer_RNN1 = optim.SGD(model_RNN1.parameters(), lr=0.01, momentum=0.5)
optimizer_RNN2 = optim.SGD(model_RNN2.parameters(), lr=0.01, momentum=0.5)


# responsible for the learning rate decay
def lambda_lin2(epoch): return 0.8 ** epoch


scheduler_lin2 = optim.lr_scheduler.LambdaLR(optimizer_lin2, lr_lambda=lambda_lin2)


def lambda_lin3(epoch): return 0.1 ** epoch


scheduler_lin3 = optim.lr_scheduler.LambdaLR(optimizer_lin3, lr_lambda=lambda_lin3)


def lambda_lin4(epoch): return 0.1 ** epoch


scheduler_lin4 = optim.lr_scheduler.LambdaLR(optimizer_lin4, lr_lambda=lambda_lin4)

whichModelToTrain = 2

print('\n\nTRAINING...')
for epoch in range(0, EPOCHS):
    print('\nTraining Linear sign quantization model:')
    train(epoch, model_lin1, optimizer_lin1)
    # print('\nTraining Linear uniform codebook quantization model:')
    # train(epoch, model_lin2, optimizer_lin2)
    # print('\nTraining Linear SOM learning codebook quantization model:')
    # train(epoch, model_lin5, optimizer_lin5)
    # print('\nTraining analog - sign quantization- digital model:')
    # trainAnalogDigital(epoch, model_lin3, model_lin4, optimizer_lin3,
    #                    optimizer_lin4, S_codebook)
    # print('\nTraining RNN sign quantization model:')
    # train(epoch, model_RNN1, optimizer_RNN1)
    # print('\nTraining LSTM sign quantization model:')
    # train(epoch, model_RNN2, optimizer_RNN2)
    # step the learning rate decay
    # scheduler_lin2.step()
    # scheduler_lin3.step()
    # scheduler_lin4.step()


print('\n\n==============\nTESTING...\n==============\n\n')
print('\nTesting Linear sign quantization model:\n')
test(model_lin1)
model_lin1_loss = test(model_lin1)
print('\nRate: {:.4f}\nAverage loss: {:.4f}'.format(QUANTIZATION_RATE,
                                                    model_lin1_loss))
testLogger.logResult(QUANTIZATION_RATE, model_lin1_loss,
                     algorithm='Linear sign')
# print('=======================================================================')
# print('\nTesting Linear uniform quantization model:')
# model_lin2_loss = test(model_lin2)
# print('\nTest average loss: {:.4f}\n'.format(model_lin2_loss))
# print('=======================================================================')
# print('\nTesting Linear SOM quantization model:')
# model_SOM = linearModels.UniformQuantizerNet(model_lin5.testCodebook)
# test(model_SOM)
# print('=======================================================================')
# print('\nTesting RNN sign quantization model:')
# test(model_RNN1)
# print('=======================================================================')
# print('\nTesting LSTM sign quantization model:')
# test(model_RNN2)
