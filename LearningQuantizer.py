
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import UniformQuantizer
from projectConstants import *
import numpy as np
from torch.autograd import Variable

# Inherit from Function
class signActivation(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input



# Inherit from Function
class QuantizationFunction(torch.autograd.Function):
    """Applies a quantization process to the incoming data.
        Can be integrated as activation layer of NN, therefore useful for cases
        we want to check the performance while keeping the whole system as one
        neural network. This function is "hidden" to the backpropegation, i.e.,
        it moves backward its input gradient.

    Shape
    -----
        Input
            To be filled...
        Output
            qunatized_input
                the quantized input formed using codebook. The quantized input
                is the closest codeword avaliable in codeword.

    Attributes
    ----------
        codebook
            the fixed codebook of the module of shape `(M x 1)`
            which will construct the quantized input

    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, codebook):
        ctx.save_for_backward(input)
        input_data = input.data
        input_numpy = input_data.numpy()
        qunatized_input = torch.zeros(input.size())
        for ii in range(0, input_data.size(0)):
            for jj in range(0, input_data.size(1)):
                qunatized_input[ii][jj], __ = UniformQuantizer.get_optimal_word(
                    input_numpy[ii, jj], codebook)
        return qunatized_input

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None






# Inherit from Function
class LearningSOMFunction(torch.autograd.Function):
    """Applies a quantization process to the incoming scalar data using SOM


    Shape
    -----
        Input
            To be filled...
        Output
            qunatized_input
                the quantized input formed using codebook. The quantized input
                is the closest codeword avaliable in codeword.
            retcodebook
                the learnable codebook using SOM architecture of shape `(M x 1)`
                will be in learning mode during the train proccess and will be
                tested in the test proccess as the argument for new
                linearModels.UniformQuantizerNet

    Attributes
    ----------
        codebook
            the fixed codebook of the module of shape `(M x 1)`
            which will construct the quantized input



    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, codebook, testCodebook):
        ctx.save_for_backward(input)
        input_data = input.data
        input_numpy = input_data.numpy()
        qunatized_input = torch.zeros(input.size())
        retcodebook = list(testCodebook.data.numpy())
        for ii in range(0, input_data.size(0)):
            for jj in range(0, input_data.size(1)):
                qunatized_input[ii][jj], __ = UniformQuantizer.get_optimal_word(
                    input_numpy[ii, jj], codebook)
                itrVal, quantized_idx = UniformQuantizer.get_optimal_word(
                    input_numpy[ii, jj], tuple(retcodebook))
                # update winner codeword
                retcodebook[quantized_idx] = retcodebook[quantized_idx] + CODEBOOK_LR*(input_numpy[ii, jj] - itrVal)
        retcodebook = torch.from_numpy(np.asarray(retcodebook))
        retcodebook = Variable(retcodebook.float())
        return qunatized_input, retcodebook

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None
