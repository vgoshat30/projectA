import scipy.io as sio
import scipy.optimize as optim
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from random import random
from datetime import datetime
import Logger as log
from torch.nn.parameter import Parameter
import torch
import sys
import time
from TanhToStep import QuantizeTanh


# Coefficients of the tanh
a = [-0.6450403, -1.0656009, -0.78059906, -1.2263683, -1.9039588, -1.9144812, -0.10208344]
b = [0.16478617, 0.99923134, -0.080281906, -0.98418796, -1.8564541, 1.8370808, 0.40390304]
# X axis limit
xlim = [-10, 10]
# Number of points in the graph
resolution = 10000
slope = 100
# Create symbolic variable x
symX = sym.symbols('x')
# Probably should write a loop but was too lazy...
sym_tanh = a[0] * sym.tanh(symX + b[0]) + \
    a[1] * sym.tanh(slope * (symX + b[1])) + \
    a[2] * sym.tanh(slope * (symX + b[2])) + \
    a[3] * sym.tanh(slope * (symX + b[3])) + \
    a[4] * sym.tanh(slope * (symX + b[4])) + \
    a[5] * sym.tanh(slope * (symX + b[5])) + \
    a[6] * sym.tanh(slope * (symX + b[6]))
# Convert the symbolic functions to numpy friendly (for substitution)
np_tanh = sym.lambdify(symX, sym_tanh, "numpy")
# Create avector of the quantize function
quantized = []
x = np.linspace(xlim[0], xlim[1], num=resolution)
for ii in range(0, np.size(x)):
    quantized.append(QuantizeTanh(x[ii], np_tanh, b, sum(a))[0])
# Plot
plt.plot(x, np_tanh(x), label='Sum of tanh')
plt.plot(x, quantized, label='Quantization function')
plt.legend()
plt.show()
