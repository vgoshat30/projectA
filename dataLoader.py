''' Simple Network, Using data generated from matlab file

Created: 13 May 2018

Creators:   Gosha Tsintsadze
            Matan Shohat

Description:
    This code utilizes a neural network using pytorch, while the data is
    generated from a .m matlab script whitten by Nir Shlezinger, saved in a .mat
    file in the python code directory and imported to python using scipy.io

    The data consists of channel (S) and observations (X) couples,
    It implements quantization process with 1 - Bit scalar quantizers with the sign() quantization logic
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np


#  import matlab.engine


BATCH_SIZE = 8
EPOCHS = 2


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
        self.l5 = nn.Linear(120, 80)

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
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test():
    model.eval()
    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target)
    test_loss /= (len(test_loader.dataset)/BATCH_SIZE)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


# # Run matlab .m script to generate .mat file with test and training data
# eng = matlab.engine.start_matlab()
# eng.shlezDataGen(nargout=0)


datasetTrainLoader = ShlezDatasetTrain()
train_loader = DataLoader(dataset=datasetTrainLoader, batch_size=BATCH_SIZE, shuffle=True)
datasetTestLoader = ShlezDatasetTest()
test_loader = DataLoader(dataset=datasetTestLoader, batch_size=BATCH_SIZE, shuffle=True)


for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process


model = SignQuantizerNet()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print('\n\nTRAINING...')
for epoch in range(0, EPOCHS):
    train(epoch)

print('\n\nTESTING...')
test()
