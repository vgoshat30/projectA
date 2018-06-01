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

aa = [1, 2, 3]
print(type(aa))
aa = np.asarray(aa)
print(type(aa))

# The bouderies for the solver. It will search zeros of the second derivative in
# the range: (a[i]-searchField, a[i]+searchField)
searchField = 3
# Coefficients of the tanh
a = [5, -5, -15, -25, -35]
b = [1, 5, 6, 7, 10]
# X axis limit
xlim = [-10, 45]
# Number of points in the graph
resolution = 1000
# Create symbolic variable x
symX = sym.symbols('x')
# Probably should write a loop but was too lazy...
sym_tanh = b[0] * sym.tanh(symX + a[0]) + \
    b[1] * sym.tanh(symX + a[1]) + \
    b[2] * sym.tanh(symX + a[2]) + \
    b[3] * sym.tanh(symX + a[3]) + \
    b[4] * sym.tanh(symX + a[4]) + sum(b)
# Create symbolic hiperbolic tangent and its second derivative function
sym_tanh_deriv1 = sym.diff(sym_tanh, symX, 1)
sym_tanh_deriv2 = sym.diff(sym_tanh, symX, 2)
# Convert the symbolic functions to numpy friendly (for substitution)
np_tanh = sym.lambdify(symX, sym_tanh, "numpy")
np_tanh_deriv1 = sym.lambdify(symX, sym_tanh_deriv1, "numpy")
np_tanh_deriv2 = sym.lambdify(symX, sym_tanh_deriv2, "numpy")
# Find the wanted zero of the second derivative of tanh
# IMPORTANT!!!
# This is actually cheating! Because the root is always in a[ii]!
np_tanh_deriv2_zeros = optim.newton(np_tanh_deriv2, -a[0])
for ii, value in enumerate(a):
    if ii is not 0:
        np_tanh_deriv2_zeros = np.append(np_tanh_deriv2_zeros,
                                         optim.newton(np_tanh_deriv2, -a[ii]))
print('Relevant zeros of the second derivative:\n', np_tanh_deriv2_zeros)
# Plot
x = np.linspace(xlim[0], xlim[1], num=resolution)
plt.plot(x, np_tanh(x))
plt.plot(x, np_tanh_deriv2(x), color='green')
plt.plot(np_tanh_deriv2_zeros, np_tanh(np_tanh_deriv2_zeros), 'x', color='red')
# plt.show()
