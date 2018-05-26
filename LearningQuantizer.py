import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from UniformQuantizer import *


class LearningQuantizerLayer(Module):
    """Applies a quantization process to the incoming scalar data

    Args
    ----
        M
            size of codebook

    Shape
    -----
        Input
            :math:`(N, *, in\_features)` where :math:`*` means any number of
            additional dimensions
        Output
            :math:`(N, *, out\_features)` where all but the last dimension are
            the same shape as the input.

    Attributes
    ----------
        codebook
            the learnable codebook of the module of shape `(M x 1)`

    """

    def __init__(self, M):
        super(LearningQuantizerLayer, self).__init__()
        self.M = M
        self.codebook = Parameter(torch.Tensor(M, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.codebook.data, _ = torch.sort(torch.randn(self.M, 1))

    def forward(self, input):
        # self.codebook.data, _ = torch.sort(self.codebook.data)
        input_value = input.item()
        if input_value < self.codebook.data[0]:
            return self.codebook.data[0]
        elif input_value > self.codebook.data[self.M - 1]:
            return self.codebook.data[self.M - 1]
        for ii in range(0, self.M - 1):
            if(input_value > self.codebook.data[ii] and input_value <
               self.codebook.data[ii + 1]):
                ret = self.codebook.data[ii] if input_value <= (
                    (self.codebook.data[ii + 1] - self.codebook.data[ii])/2) \
                    else self.codebook.data[ii + 1]
                return ret
        return self.codebook.data[0]

    def extra_repr(self):
        return 'M={}'.format(self.M)


class QuantizerUniformLayer(Module):
    def __init__(self, codebook):
        super(QuantizerUniformLayer, self).__init__()
        self.codebook = codebook

    def forward(self, input):
        input_data = input.data
        input_numpy = input_data.numpy()
        qunatized_input = torch.zeros(input.size())
        for ii in range(0, input_data.size(0)):
            for jj in range(0, input_data.size(1)):
                qunatized_input[ii][jj] = get_optimal_word(input_numpy[ii, jj],
                                                           self.codebook)
        return qunatized_input

    def extra_repr(self):
        return 'codebook=%s'.format(self.codebook)


class MyQuantizerUniformActivation(torch.autograd.Function):
    """
    In order to keep propegating through step activations we have to construct
    our own quantization activation function with the apropriate forward and
    backward propegations.
    """

    def __init__(self, codebook):
        self.codebook = codebook

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward
        method.
        """
        ctx.save_for_backward(input)
        input_data = input.data
        input_numpy = input_data.numpy()
        qunatized_input = torch.zeros(input.size())
        for ii in range(0, input_data.size(0)):
            for jj in range(0, input_data.size(1)):
                qunatized_input[ii][jj] = get_optimal_word(input_numpy[ii, jj],
                                                           self.codebook)
        return qunatized_input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of
        the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None
