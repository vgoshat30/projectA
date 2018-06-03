"""Main file of projectA

    Trains and tests:
    -- Sign quantization Linear network
    -- Uniform quantization Linear network
    -- Sign qunatization Linear + RNN network
    -- Sign qunatization Linear + LSTM network
    -- Tanh learning quantization Linear networks
    -- Strached tanh learning quantization Linear networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
import math
from datetime import datetime

from dataLoader import *
import linearModels
import RNNmodels
from projectConstants import *
import UniformQuantizer
import Logger as log
import userInterface as UI
import TanhToStep


def train(modelname, epoch, model, optimizer, scheduler=None):
    model.train()
    for corrEpoch in range(0, epoch):
        if scheduler != None:
            scheduler.step()
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


def testTanh(model, codebookSize):
    f, a, b, infVal = TanhToStep.extractModelParameters(model, codebookSize)
    classificationCounter = np.zeros(codebookSize)
    test_loss = 0
    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model.l1(data)
        output = model.l2(output)
        output_data = output.data
        output_numpy = output_data.numpy()
        for ii in range(0, output_data.size(0)):
            for jj in range(0, output_data.size(1)):
                output[ii, jj], kk = TanhToStep.QuantizeTanh(
                    output_numpy[ii, jj], f, b, infVal, codebookSize)
                classificationCounter[kk] += 1
        output = model.l3(output)
        output = model.l4(output)
        output = model.l5(output)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))

    print('Num. of clasifications by word:', classificationCounter)
    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)
    return test_loss.detach().numpy()

def testStretchedTanh(model, codebookSize, stretchFactor):
    f, a, b, infVal = TanhToStep.extractModelParameters(model, codebookSize)
    classificationCounter = np.zeros(codebookSize)
    test_loss = 0
    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model.l1(data)
        output = model.l2(output)
        output = stretchFactor*output
        output_data = output.data
        output_numpy = output_data.numpy()
        for ii in range(0, output_data.size(0)):
            for jj in range(0, output_data.size(1)):
                output[ii, jj], kk = TanhToStep.QuantizeTanh(
                    output_numpy[ii, jj], f, b, infVal, codebookSize)
                classificationCounter[kk] += 1
        output = (1/stretchFactor)*output
        output = model.l3(output)
        output = model.l4(output)
        output = model.l5(output)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))

    print('Num. of clasifications by word:', classificationCounter)
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

# responsible for the learning rate decay
def lambda_linUniformQunat(epoch): return 0.8 ** epoch

def lambda_linAnalogSign(epoch): return 0.1 ** epoch

def lambda_linDigitalSign(epoch): return 0.1 ** epoch

# Get the class containing the train data from dataLoader.py
trainData = ShlezDatasetTrain()
# define training dataloader
trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# Do the same for the test data
testData = ShlezDatasetTest()
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=True)



########################################################################
###                     Model initilizations                         ###
########################################################################
constantPermutationns = [(slope, epoch, lr, codebookSize) for slope in SLOPE_RANGE for epoch in EPOCH_RANGE for lr in LR_RANGE for codebookSize in M_RANGE]
for constPerm in constantPermutationns:
    slope = constPerm[0]
    epoch = constPerm[1]
    lr = constPerm[2]
    codebookSize = constPerm[3]
    QUANTIZATION_RATE = math.log2(codebookSize)*OUTPUT_DIMENSION / INPUT_DIMENSION

    # Generate uniform codebooks
    X_codebook = UniformQuantizer.codebook_uniform(trainData.X_var, codebookSize)
    S_codebook = UniformQuantizer.codebook_uniform(trainData.S_var, codebookSize)

    # model_linSignQunat: Basic linear network with sign activation as the
    # quantization
    model_linSignQunat = linearModels.SignQuantizerNet()
    # model_linUniformQunat: Basic linear network with uniform quantization instead
    # of sign
    model_linUniformQunat = linearModels.UniformQuantizerNet(S_codebook)
    # model_linAnalogSign: Basic linear network which learns to prepare the analog
    # signal data for quantization
    model_linAnalogSign = linearModels.AnalogProcessNet()
    # model_linDigitalSign: Basic linear network which learns to perform the digital
    # processing after the quantization and results the channel coefficients
    model_linDigitalSign = linearModels.DigitalProcessNet()
    # Replacing quantizer with sum of tanh function for the learning:
    model_tanhQuantize = linearModels.tanhQuantizeNet(tanhSlope=slope, codebookSize=codebookSize)

    model_stretchTanhQuantize = linearModels.StretchedtanhQuantizeNet(tanhSlope=slope, codebookSize=codebookSize, stretchFactor=5)
    # model_rnnSignQuant: Basic linear network with sign activation and
    # pre-quantization RNN layer
    model_rnnSignQuant = RNNmodels.SignQuantizerNetRNN()
    # model_lstmSignQuant: Basic linear network with sign activation and
    # pre-quantization LSTM layer
    model_lstmSignQuant = RNNmodels.SignQuantizerNetLSTM()



    criterion = nn.MSELoss()
    optimizer_linSignQunat = optim.SGD(model_linSignQunat.parameters(),
                                       lr=lr, momentum=0.5)
    optimizer_linUniformQunat = optim.SGD(model_linUniformQunat.parameters(),
                                          lr=lr, momentum=0.5)
    optimizer_linAnalogSign = optim.SGD(model_linAnalogSign.parameters(),
                                        lr=lr, momentum=0.5)
    optimizer_linDigitalSign = optim.SGD(model_linDigitalSign.parameters(),
                                         lr=lr, momentum=0.5)
    optimizer_rnnSignQuant = optim.SGD(model_rnnSignQuant.parameters(),
                                       lr=lr, momentum=0.5)
    optimizer_lstmSignQuant = optim.SGD(model_lstmSignQuant.parameters(),
                                        lr=lr, momentum=0.5)
    optimizer_tanhQuantize = optim.SGD(model_tanhQuantize.parameters(),
                                       lr=lr, momentum=0.5)
    optimizer_stretchTanhQuantize = optim.SGD(model_stretchTanhQuantize.parameters(),
                                       lr=lr, momentum=0.5)
    scheduler_linUniformQunat = optim.lr_scheduler.ExponentialLR(optimizer_linUniformQunat, gamma=0.7, last_epoch=-1)
    scheduler_tanhQuantize = optim.lr_scheduler.ExponentialLR(optimizer_tanhQuantize, gamma=0.7, last_epoch=-1)
    scheduler_stretchTanhQuantize = optim.lr_scheduler.ExponentialLR(optimizer_stretchTanhQuantize, gamma=0.7, last_epoch=-1)


    ########################################################################
    ###               Training and testing all networks                  ###
    ########################################################################

    # ------------------------------
    # ---       Training         ---
    # ------------------------------


    UI.trainHeding()

    if 'Linear sign quantization' in modelsToActivate and slope == SLOPE_RANGE[0]:
        model_linSignQunat_runtime = datetime.now()
        modelname = 'Linear sign quantization'
        UI.trainMessage(modelname)
        train(modelname, epoch, model_linSignQunat,
              optimizer_linSignQunat)
        model_linSignQunat_runtime = datetime.now() - model_linSignQunat_runtime

    if 'Linear uniform codebook' in modelsToActivate and slope == SLOPE_RANGE[0]:
        model_linUniformQunat_runtime = datetime.now()
        modelname = 'Linear uniform codebook'
        UI.trainMessage(modelname)
        train(modelname, epoch, model_linUniformQunat,
              optimizer_linUniformQunat, scheduler_linUniformQunat)
        model_linUniformQunat_runtime = datetime.now() - \
            model_linUniformQunat_runtime

    if 'Linear SOM learning codebook' in modelsToActivate and slope == SLOPE_RANGE[0]:
        model_linSOMQuant_runtime = datetime.now()
        modelname = 'Linear SOM learning codebook'
        UI.trainMessage(modelname)
        model_SOM = trainSOM(modelname, epoch, S_codebook)
        model_linSOMQuant_runtime = datetime.now() - model_linSOMQuant_runtime

    if 'Analog sign quantization' in modelsToActivate and slope == SLOPE_RANGE[0]:
        modelname = 'Analog sign quantization'
        UI.trainMessage(modelname)
        trainAnalogDigital(modelname, epoch, model_linAnalogSign,
                           model_linDigitalSign,
                           optimizer_linAnalogSign, optimizer_linDigitalSign,
                           S_codebook)

    if 'RNN sign quantization' in modelsToActivate and slope == SLOPE_RANGE[0]:
        modelname = 'RNN sign quantization'
        UI.trainMessage(modelname)
        train(modelname, epoch, model_rnnSignQuant,
              optimizer_rnnSignQuant)

    if 'LSTM sign quantization' in modelsToActivate and slope == SLOPE_RANGE[0]:
        modelname = 'LSTM sign quantization'
        UI.trainMessage(modelname)
        train(modelname, epoch, model_lstmSignQuant,
              optimizer_lstmSignQuant)

    if 'Tanh quantization' in modelsToActivate:
        modelname = 'Tanh quantization'
        UI.trainMessage(modelname)
        model_tanhQuantize_runtime = datetime.now()
        train(modelname, epoch, model_tanhQuantize,
              optimizer_tanhQuantize, scheduler_tanhQuantize)
        model_tanhQuantize_runtime = datetime.now() - model_tanhQuantize_runtime

    if 'Stretched Tanh quantization' in modelsToActivate:
        modelname = 'Stretched Tanh quantization'
        UI.trainMessage(modelname)
        model_stretchTanhQuantize_runtime = datetime.now()
        train(modelname, epoch, model_stretchTanhQuantize,
              optimizer_stretchTanhQuantize, scheduler_stretchTanhQuantize)
        model_stretchTanhQuantize_runtime = datetime.now() - model_stretchTanhQuantize_runtime


    # ------------------------------
    # ---        Testing         ---
    # ------------------------------

    UI.testHeding()

    UI.horizontalLine()

    print('PRINTING RESULTS UNDER THE FOLLOWING PARAMETERS:')
    print('EPOCHS: ', epoch, '\nLEARNING RATE: ', lr, '\nM: ', codebookSize, '\nTANH SLOPE: ', slope)

    UI.horizontalLine()

    if 'Linear sign quantization' in modelsToActivate and slope == SLOPE_RANGE[0]:
        modelname = 'Linear sign quantization'
        UI.testMessage()
        model_linSignQunat_loss = test(model_linSignQunat)
        UI.testResults(QUANTIZATION_RATE, model_linSignQunat_loss)
        log.log(QUANTIZATION_RATE, model_linSignQunat_loss,
                algorithm=modelname,
                runtime=model_linSignQunat_runtime)

    if 'Linear uniform codebook' in modelsToActivate and slope == SLOPE_RANGE[0]:
        modelname = 'Linear uniform codebook'
        UI.testMessage(modelname)
        model_linUniformQunat_loss = test(model_linUniformQunat)
        log.log(QUANTIZATION_RATE, model_linUniformQunat_loss,
                algorithm=modelname,
                runtime=model_linUniformQunat_runtime)

    if 'Linear SOM learning codebook' in modelsToActivate and slope == SLOPE_RANGE[0]:
        modelname = 'Linear SOM learning codebook'
        UI.testMessage()
        model_linSOMQuant_loss = test(model_SOM)
        log.log(QUANTIZATION_RATE, model_linSOMQuant_loss,
                algorithm=modelname,
                runtime=model_linSOMQuant_runtime)

    if 'RNN sign quantization' in modelsToActivate and slope == SLOPE_RANGE[0]:
        UI.testMessage('RNN sign quantization')
        test(model_rnnSignQuant)

    if 'LSTM sign quantization' in modelsToActivate and slope == SLOPE_RANGE[0]:
        UI.testMessage('LSTM sign quantization')
        test(model_lstmSignQuant)

    if 'Tanh quantization' in modelsToActivate:
        modelname = 'Tanh quantization'
        UI.testMessage(modelname)
        model_tanhQuantize_loss = testTanh(model_tanhQuantize, codebookSize)
        UI.testResults(QUANTIZATION_RATE, model_tanhQuantize_loss)
        log.log(QUANTIZATION_RATE, model_tanhQuantize_loss, 'dontshow',
                algorithm=modelname,
                runtime=model_tanhQuantize_runtime,
                epochs=epoch)
                # GOSHA NEED TO ADD slope=slope
    if 'Stretched Tanh quantization' in modelsToActivate:
        modelname = 'Stretched Tanh quantization'
        UI.testMessage(modelname)
        model_stretchTanhQuantize_loss = testStretchedTanh(model_stretchTanhQuantize, codebookSize, 5)
        UI.testResults(QUANTIZATION_RATE, model_stretchTanhQuantize_loss)
        log.log(QUANTIZATION_RATE, model_stretchTanhQuantize_loss, 'dontshow',
                algorithm=modelname,
                runtime=model_stretchTanhQuantize_runtime,
                epochs=epoch)
                # GOSHA NEED TO ADD slope=slope
