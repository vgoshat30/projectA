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
a = [0.61832136, 1.1391762, 0.2488268, 1.4357071, 1.6707542, 0.12944213, 1.2324758]
b = [0.21479227, 1.2251697, 0.2885009, 0.92580044, -0.9657938, 0.44270614, -1.2941284]
# X axis limit
xlim = [-3, 3]
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
# Create symbolic hiperbolic tangent and its second derivative function
sym_tanh_deriv1 = sym.diff(sym_tanh, symX, 1)
sym_tanh_deriv2 = sym.diff(sym_tanh, symX, 2)
# Convert the symbolic functions to numpy friendly (for substitution)
np_tanh = sym.lambdify(symX, sym_tanh, "numpy")
np_tanh_deriv1 = sym.lambdify(symX, sym_tanh_deriv1, "numpy")
np_tanh_deriv2 = sym.lambdify(symX, sym_tanh_deriv2, "numpy")
# Plot
x = np.linspace(xlim[0], xlim[1], num=resolution)
plt.plot(x, np_tanh(x))
# plt.plot(x, np_tanh_deriv2(x), color='green')
# plt.plot(x, QuantizeTanh(x, sym_tanh, b, sum(a)))
plt.show()
