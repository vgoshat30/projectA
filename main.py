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
from datetime import datetime

from dataLoader import *
import linearModels
import RNNmodels
from projectConstants import *
import UniformQuantizer
import Logger as log
import UserInterface as UI


def train(modelname, epoch, model, optimizer):
    for corrEpoch in range(0, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = Variable(data.float()), Variable(target.float())

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, 1), target.view(-1, 1))
            loss.backward(retain_graph=True)
            optimizer.step()

            if batch_idx % 10 == 0:
                UI.trainIteration(modelname, corrEpoch, batch_idx, data,
                                  trainLoader, loss)


def trainAnalogDigital(modelname, epoch, modelAnalog, modelDigital,
                       optimizerAnalog, optimizerDigital, codebook):
    for corrEpoch in range(0, epoch):
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
                UI.trainIteration('Analog', corrEpoch, batch_idx, data,
                                  trainLoader, lossAnalog)

    modelAnalog.eval()
    # Train digital NN
    for corrEpoch in range(0, epoch):
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
                UI.trainIteration('Digital', corrEpoch, batch_idx, data,
                                  trainLoader, lossDigital)


def trainSOM(modelname, epoch, initCodebook):
    print('Init Codebook:\n{0}'.format(initCodebook))
    newCodebook = list(initCodebook)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = Variable(data.float()), Variable(target.float())
        target_numpy = target.data.numpy()
        for ii in range(0, target.size(0)):
            for jj in range(0, target.size(1)):
                itrVal, quantized_idx = UniformQuantizer.get_optimal_word(
                    target_numpy[ii, jj], tuple(newCodebook))
                # update winner codeword
                newCodebook[quantized_idx] = newCodebook[quantized_idx] + \
                    CODEBOOK_LR*(target_numpy[ii, jj] - itrVal)
    testCodebook = tuple(newCodebook)
    print('SOM Codebook:\n{0}'.format(testCodebook))
    SOMtestModel = linearModels.UniformQuantizerNet(testCodebook)
    SOMtestOptim = optim.SGD(SOMtestModel.parameters(), lr=0.01, momentum=0.5)
    train(modelname, epoch, SOMtestModel, SOMtestOptim)
    return SOMtestModel


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
# model_linSignQunat: Basic linear network with sign activation as the quantization
model_linSignQunat = linearModels.SignQuantizerNet()
# model_linUniformQunat: Basic linear network with uniform quantization instead of sign
model_linUniformQunat = linearModels.UniformQuantizerNet(S_codebook)
# model_linAnalogSign: Basic linear network which learns to prepare the analog signal
# data for quantization
model_linAnalogSign = linearModels.AnalogProcessNet()
# model_linDigitalSign: Basic linear network which learns to perform the digital
# processing after the quantization and results the channel coefficients
model_linDigitalSign = linearModels.DigitalProcessNet()


# model_rnnSignQuant: Basic linear network with sign activation and pre-quantization
# RNN layer
model_rnnSignQuant = RNNmodels.SignQuantizerNetRNN()
# model_lstmSignQuant: Basic linear network with sign activation and pre-quantization
# LSTM layer
model_lstmSignQuant = RNNmodels.SignQuantizerNetLSTM()


criterion = nn.MSELoss()
optimizer_linSignQunat = optim.SGD(model_linSignQunat.parameters(), lr=0.01, momentum=0.5)
optimizer_linUniformQunat = optim.SGD(model_linUniformQunat.parameters(), lr=0.01, momentum=0.5)
optimizer_linAnalogSign = optim.SGD(model_linAnalogSign.parameters(), lr=0.01, momentum=0.5)
optimizer_linDigitalSign = optim.SGD(model_linDigitalSign.parameters(), lr=0.01, momentum=0.5)
optimizer_rnnSignQuant = optim.SGD(model_rnnSignQuant.parameters(), lr=0.01, momentum=0.5)
optimizer_lstmSignQuant = optim.SGD(model_lstmSignQuant.parameters(), lr=0.01, momentum=0.5)


# responsible for the learning rate decay
def lambda_linUniformQunat(epoch): return 0.8 ** epoch


scheduler_linUniformQunat = optim.lr_scheduler.LambdaLR(
    optimizer_linUniformQunat, lr_lambda=lambda_linUniformQunat)


def lambda_linAnalogSign(epoch): return 0.1 ** epoch


scheduler_linAnalogSign = optim.lr_scheduler.LambdaLR(
    optimizer_linAnalogSign, lr_lambda=lambda_linAnalogSign)


def lambda_linDigitalSign(epoch): return 0.1 ** epoch


scheduler_linDigitalSign = optim.lr_scheduler.LambdaLR(
    optimizer_linDigitalSign, lr_lambda=lambda_linDigitalSign)


########################################################################
###               Training and testing all networks                  ###
########################################################################

# ------------------------------
# ---       Training         ---
# ------------------------------

UI.trainHeding()

if 'Linear sign quantization' in modelsToActivate:
    model_linSignQunat_runtime = datetime.now()
    modelname = 'Linear sign quantization'
    UI.trainMessage(modelname)
    train(modelname, EPOCHS_linSignQunat, model_linSignQunat, optimizer_linSignQunat)
    model_linSignQunat_runtime = datetime.now() - model_linSignQunat_runtime


if 'Linear uniform codebook' in modelsToActivate:
    model_linUniformQunat_runtime = datetime.now()
    modelname = 'Linear uniform codebook'
    UI.trainMessage(modelname)
    train(modelname, EPOCHS_linUniformQunat, model_linUniformQunat, optimizer_linUniformQunat)
    # step the learning rate decay
    scheduler_linUniformQunat.step()
    model_linUniformQunat_runtime = datetime.now() - model_linUniformQunat_runtime

if 'Linear SOM learning codebook' in modelsToActivate:
    model_linSOMQuant_runtime = datetime.now()
    modelname = 'Linear SOM learning codebook'
    UI.trainMessage(modelname)
    model_SOM = trainSOM(modelname, EPOCHS_linSOMQuant, S_codebook)
    model_linSOMQuant_runtime = datetime.now() - model_linSOMQuant_runtime

if 'Analog sign quantization' in modelsToActivate:
    modelname = 'Analog sign quantization'
    UI.trainMessage(modelname)
    trainAnalogDigital(modelname, EPOCHS_ADSignQuant, model_linAnalogSign, model_linDigitalSign,
                       optimizer_linAnalogSign, optimizer_linDigitalSign, S_codebook)
    # step the learning rate decay
    scheduler_linDigitalSign.step()

if 'RNN sign quantization' in modelsToActivate:
    modelname = 'RNN sign quantization'
    UI.trainMessage(modelname)
    train(modelname, EPOCHS_rnnSignQuant, model_rnnSignQuant, optimizer_rnnSignQuant)

if 'LSTM sign quantization' in modelsToActivate:
    modelname = 'LSTM sign quantization'
    UI.trainMessage(modelname)
    train(modelname, EPOCHS_lstmSignQuant, model_lstmSignQuant, optimizer_lstmSignQuant)

# ------------------------------
# ---        Testing         ---
# ------------------------------

UI.testHeding()
if 'Linear sign quantization' in modelsToActivate:
    UI.testMessage('Linear sign quantization')
    model_linSignQunat_loss = test(model_linSignQunat)
    UI.testResults(QUANTIZATION_RATE, model_linSignQunat_loss)
    log.logResult(QUANTIZATION_RATE, model_linSignQunat_loss,
                  algorithm='Linear sign quantization',
                  runtime=model_linSignQunat_runtime)

if 'Linear uniform codebook' in modelsToActivate:
    UI.testMessage('Linear uniform codebook')
    model_linUniformQunat_loss = test(model_linUniformQunat)
    log.logResult(QUANTIZATION_RATE, model_linUniformQunat_loss,
                  algorithm='Linear sign quantization',
                  runtime=model_linUniformQunat_runtime)


if 'Linear SOM learning codebook' in modelsToActivate:
    UI.testMessage('Linear SOM learning codebook')
    model_linSOMQuant_loss = test(model_SOM)
    log.logResult(QUANTIZATION_RATE, model_linSOMQuant_loss,
                  algorithm='Linear SOM learning codebook',
                  runtime=model_linSOMQuant_runtime)


if 'RNN sign quantization' in modelsToActivate:
    UI.testMessage('RNN sign quantization')
    test(model_rnnSignQuant)

if 'LSTM sign quantization' in modelsToActivate:
    UI.testMessage('LSTM sign quantization')
    test(model_lstmSignQuant)
