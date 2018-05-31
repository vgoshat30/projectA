import scipy.io as sio
import scipy.optimize as optim
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from random import random
from datetime import datetime
# from Logger import *
import Logger as log
from torch.nn.parameter import Parameter
import torch

log.content(1)

# def hipTan(x):
#     return np.tanh(x)
#
#
# def hipTanDeriv1(x):s
#     return np.gradient(np.tanh(x), 0.0001)
#
#
# def hipTanDeriv2(x):
#     return np.gradient(np.gradient(np.tanh(x), 0.0001), 0.0001)
#
#
# symX = sym.symbols('x')
# tanhyp = sym.tanh(symX)
# print(tanhyp)
#
# x = np.linspace(-10, 10, 1000)
# y = tanhyp.subs(symX, 1)
# plt.plot(x, y)
# plt.show()
