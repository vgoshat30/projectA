''' Simple Network, Using data generated from matlab file

Created: 10 May 2018

Creators:   Gosha Tsintsadze
            Matan Shohat

Description:
    This code utilizes a neural network using pytorch, while the data is
    generated from a .m matlab script whitten by Nir Shlezinger, saved in a .mat
    file in the python code directory and imported to python using scipy.io

    The data consists of channel (S) and observations (X) couples
'''
import torch
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import time
import scipy.io as sio
import matlab.engine


class ShlezDatasetTrain(Dataset):
    # Data class fo the training data set (X and S pairs)

    def __init__(self):
        # Loading .mat file
        shlezMat = sio.loadmat('shlezingerMat.mat')
        # Getting train data variables from the .mat file:
        Xdata = shlezMat['trainX']
        Sdata = shlezMat['trainS']

        # Converting numpy arrays to pytorch tensors:
        self.X_data = torch.from_numpy(Xdata)
        self.S_data = torch.from_numpy(Sdata)

        print('DEBUGGING 1: X size: ', Xdata.shape, 'S size: ', Sdata.shape)

        # Number of X, S couples:
        self.len = Sdata.shape[0]

    def __getitem__(self, index):
        return self.X_data[index], self.S_data[index]

    def __len__(self):
        return self.len


class ShlezDatasetTest(Dataset):
    # Data class fo the training data set (X and S pairs)

    def __init__(self):
        # Loading .mat file
        shlezMat = sio.loadmat('shlezingerMat.mat')
        # Getting test data variables from the .mat file:
        Xdata = shlezMat['dataX']
        Sdata = shlezMat['dataS']

        # Converting numpy arrays to pytorch tensors:
        self.X_data = torch.from_numpy(Xdata)
        self.S_data = torch.from_numpy(Sdata)

        print('DEBUGGING 2: X size: ', Xdata.shape, 'S size: ', Sdata.shape)

        # Number of X, S couples:
        self.len = Sdata.shape[0]

    def __getitem__(self, index):
        return self.X_data[index], self.S_data[index]

    def __len__(self):
        return self.len


class SignQuantizerNet(nn.Module):

    def __init__(self):
        super(SignQuantizerNet, self).__init__()
        self.l1 = nn.Linear(240, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = torch.sign(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.float()), Variable(target.float())
        optimizer.zero_grad()
        output = model(data)
        print("DEBUGGING 6:", target)
        print("DEBUGGING 7:", target.view(-1))
        print("DEBUGGING 8:", target.shape)
        '''
        There is a bug here. If the code is like it's now, an error of
        "Expected object of type torch.LongTensor but found type
        torch.FloatTensor for argument #2 'target'" is thrown. To fix this
        problem, tried to add .long() type cast to 'target' tensor bu got error
        of: "multi-target not supported" [target is not one dimentional tensor
        (vector) but a two dimentional one (matrix)]. Tried running both
        target.view(-1) and target.long().view(-1) to get one dimentional
        tensor but in both cases got the same error: "Expected input batch_size
        (8) to match target batch_size (640)." (which is expected because the
        shape of target before the reshape was: ([8, 80])).
        '''
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('DEBUGGING 4: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]

    test_loss /= len(test_loader.dataset)
    print('\nDEBUGGING 5: Test set: Average loss: {:.4f}\n'.format(test_loss))


# # Run matlab .m script to generate .mat file with test and training data
# eng = matlab.engine.start_matlab()
# eng.shlezDataGen(nargout=0)


datasetTrainLoader = ShlezDatasetTrain()

train_loader = DataLoader(dataset=datasetTrainLoader, batch_size=8, shuffle=True)

datasetTestLoader = ShlezDatasetTest()

test_loader = DataLoader(dataset=datasetTestLoader, batch_size=8, shuffle=True)


for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process
print("DEBUGGING 3:", epoch, i)


model = SignQuantizerNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(1, 10):
    train(epoch)
test()
