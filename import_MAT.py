import scipy.io as sio
import torch

shlezMat = sio.loadmat('shlezingerMat.mat')

m_fDataS = torch.transpose(torch.from_numpy(shlezMat['trainX']), 0, 1)

print(m_fDataS.size(0))
