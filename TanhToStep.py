import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import UniformQuantizer
from projectConstants import *
import numpy as np
from torch.autograd import Variable

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
    # The bouderies for the solver. It will search zeros of the second derivative in
    # the range: (a[i]-searchField, a[i]+searchField)
    searchField = 3
    # Coefficients of the tanh

    a = []
    b = []
    for ii in range(0, M):
        a.appand(parameters[2*ii])
        b.appand(parameters[2*ii + 1])
    # X axis limit
    xlim = [-10, 45]
    # Number of points in the graph
    resolution = 1000
    # Create symbolic variable x
    symX = sym.symbols('x')
    # Probably should write a loop but was too lazy...
    for ii in range(0, M):
        sym_tanh = sym_tanh + a[ii] * sym.tanh(symX + b[ii])

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
    for ii, value in enumerate(b):
        if ii is not 0:
            np_tanh_deriv2_zeros = np.append(np_tanh_deriv2_zeros,
                                             optim.newton(np_tanh_deriv2, -b[ii]))
    codebook = {}
    for ii, value in enumerate(np_tanh_deriv2_zeros):
        codebook[value] = np_tanh(value)

    print('Relevant zeros of the second derivative:\n', np_tanh_deriv2_zeros)
    # Plot
    x = np.linspace(xlim[0], xlim[1], num=resolution)
    plt.plot(x, np_tanh(x))
    plt.plot(x, np_tanh_deriv2(x), color='green')
    plt.plot(np_tanh_deriv2_zeros, np_tanh(np_tanh_deriv2_zeros), 'x', color='red')
    plt.show()
