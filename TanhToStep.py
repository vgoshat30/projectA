import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import UniformQuantizer
from projectConstants import *
import numpy as np
from torch.autograd import Variable
import scipy.io as sio
import scipy.optimize as optim
import sympy as sym
from datetime import datetime
import matplotlib.pyplot as plt
from random import random
import Logger as log


def extractModelParameters(tanhModel):
    """Return the codebook extracted from tanhQuantizeNet instance

    Parameters
    ----------
        tanhModel
            instance of the tanhQuantizeNet model

    Returns
    -------
        codebook
            dictionary containing the generated codebook where the key
            represents the word center region and the value represents the
            quantized value accosiated with the key word
            """

    parameters = tanhModel.q1.weight.data.numpy()

    # Coefficients of the tanh
    a = []
    b = []
    for ii in range(0, M - 1):
        a.append(parameters[2*ii, 0])
        b.append(parameters[2*ii + 1, 0])
    extraKey = max(b) + 1
    # X axis limit
    xlim = [-10, 45]
    # Number of points in the graph
    resolution = 1000
    # Create symbolic variable x
    symX = sym.symbols('x')
    # Probably should write a loop but was too lazy...
    sym_tanh = a[0] * sym.tanh(symX + b[0]) + \
        a[1] * sym.tanh(symX + b[1]) + \
        a[2] * sym.tanh(symX + b[2]) + \
        a[3] * sym.tanh(symX + b[3]) + \
        a[4] * sym.tanh(symX + b[4]) + \
        a[5] * sym.tanh(symX + b[5]) + \
        a[6] * sym.tanh(symX + b[6])
    # Create symbolic hiperbolic tangent and its second derivative function
    sym_tanh_deriv1 = sym.diff(sym_tanh, symX, 1)
    sym_tanh_deriv2 = sym.diff(sym_tanh, symX, 2)
    # Convert the symbolic functions to numpy friendly (for substitution)
    np_tanh = sym.lambdify(symX, sym_tanh, "numpy")
    np_tanh_deriv1 = sym.lambdify(symX, sym_tanh_deriv1, "numpy")
    np_tanh_deriv2 = sym.lambdify(symX, sym_tanh_deriv2, "numpy")
    f = np_tanh
    infVal = sum(a)
    print('Tanh amplitudes (a):\n', a)
    print('Tanh shifts (b):\n', b)
    return f, a, b, infVal


def QuantizeTanh(input, f, borders, infVal):
    borders.sort()
    if input <= borders[0]:
        return -infVal, 0
    if input > borders[-1]:
        return infVal, M - 1
    for ii in range(0, len(borders) - 1):
        if borders[ii] < input and input <= borders[ii + 1]:
            return f((borders[ii + 1] + borders[ii])/2), ii + 1

    print("No available return value")
    return -1, -1
