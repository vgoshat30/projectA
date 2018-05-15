''' Simple Network, Using data generated from matlab file

Created: 16 May 2018

Creators:   Gosha Tsintsadze
            Matan Shohat

Description:
    This code utilizes a neural network using pytorch, while the data is
    generated from a .m matlab script whitten by Nir Shlezinger, saved in a .mat
    file in the python code directory and imported to python using scipy.io

    The data consists of channel (S) and observations (X) couples,
    It implements quantization process with 1 - Bit scalar quantizers with the
    sign() quantization logic
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


#  We propose an alternative model using RNN

class SignQuantizerNetRNN(nn.Module):

    def __init__(self):
        super(SignQuantizerNetRNN, self).__init__()

        # One cell RNN input_dim (240) -> output_dim (80).sequance: 1
        self.rnn = nn.RNN(input_size=240, hidden_size=80)

        # hidden : (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
        self.hidden = Variable(torch.randn(1, BATCH_SIZE, 80))
        self.l1 = nn.Linear(80, 240)
        self.l2 = nn.Linear(240, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 80)

    def forward(self, x):   # x rank (seq_len = 1, batch = BATCH_SIZE, input_size = 240)
        x, self.hidden = self.rnn(x, self.hidden)
        x = self.l1(x)
        x = self.l2(x)
        x = torch.sign(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)



def train(epoch):
    model.train()
    RNN_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.float()), Variable(target.float())

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 1), target.view(-1, 1))
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        data_RNN_model = data.view(1, BATCH_SIZE, 240)
        output_RNN_model = RNN_model(data_RNN_model)
        loss_RNN_model = criterion(output_RNN_model.view(-1,1), target.view(-1,1))
        loss_RNN_model.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLinear Loss: {:.6f}\tRNN Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss, loss_RNN_model))


def test():
    model.eval()
    RNN_model.eval()
    test_loss = test_RNN_loss = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))

        data_RNN_model = data.view(1, BATCH_SIZE, 240)
        output_RNN_model = RNN_model(data_RNN_model)
        # sum up batch RNN loss
        test_RNN_loss += criterion(output_RNN_model.view(-1, 1), target.view(-1, 1))

    test_loss /= (len(test_loader.dataset)/BATCH_SIZE)
    test_RNN_loss /= (len(test_loader.dataset)/BATCH_SIZE)

    print('\nTest set: Average loss: {:.4f}\tTest set: Average RNN loss: {:.4f}\n'.format(test_loss,test_RNN_loss))


# # Run matlab .m script to generate .mat file with test and training data
# eng = matlab.engine.start_matlab()
# eng.shlezDataGen(nargout=0)


datasetTrainLoader = ShlezDatasetTrain()
train_loader = DataLoader(dataset=datasetTrainLoader, batch_size=BATCH_SIZE, shuffle=True)
datasetTestLoader = ShlezDatasetTest()
test_loader = DataLoader(dataset=datasetTestLoader, batch_size=BATCH_SIZE, shuffle=True)


model = SignQuantizerNet()
RNN_model = SignQuantizerNetRNN()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print('\n\nTRAINING...')
for epoch in range(0, EPOCHS):
    train(epoch)

print('\n\nTESTING...')
test()
