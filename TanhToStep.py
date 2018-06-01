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

    # The bouderies for the solver. It will search zeros of the second
    # derivative in the range: (a[i]-searchField, a[i]+searchField)
    searchField = 3
    # Coefficients of the tanh

    a = []
    b = []
    for ii in range(0, M):
        a.append(parameters[2*ii, 0])
        b.append(parameters[2*ii + 1, 0])
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
        a[6] * sym.tanh(symX + b[6]) + \
        a[7] * sym.tanh(symX + b[7])

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
    np_tanh_deriv2_zeros = np.asarray(b)
    codebook = {}
    for ii in range(0, np_tanh_deriv2_zeros.size):
        key = np_tanh_deriv2_zeros[ii]
        val = np_tanh(np_tanh_deriv2_zeros[ii])
        codebook[key] = val
    print('Relevant zeros of the second derivative:\n', np_tanh_deriv2_zeros)
    # Plot
    x = np.linspace(xlim[0], xlim[1], num=resolution)
    plt.plot(x, np_tanh(x))
    plt.plot(x, np_tanh_deriv2(x), color='green')
    plt.plot(np_tanh_deriv2_zeros, np_tanh(np_tanh_deriv2_zeros), 'x',
             color='red')
    # plt.show()
    return codebook


def QuantizeWithDict(input, codebook):
    codebookKeys = list(codebook.keys())
    codebookKeys = sorted(codebookKeys)
    qunatized_input = UniformQuantizer.get_optimal_word(input,
                                                        tuple(codebookKeys))
    qunatized_input = qunatized_input[1]
    return qunatized_input
