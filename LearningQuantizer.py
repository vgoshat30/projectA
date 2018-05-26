
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import UniformQuantizer
from projectConstants import *

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
            To be filled...

    Attributes
    ----------
        codebook
            the learnable codebook of the module of shape `(M x 1)`

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
class LearningQuantizerFunction(torch.autograd.Function):
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
        # Note that both forward and backward are @staticmethods
        @staticmethod
        # bias is an optional argument
        def forward(ctx, input, codebook, testCodebook):
            ctx.save_for_backward(input)
            input_data = input.data
            input_numpy = input_data.numpy()
            qunatized_input = torch.zeros(input.size())
            quantized_idx = torch.zeros(input.size())
            itrVal = 0;
            retcodebook = list(testCodebook);
            for ii in range(0, input_data.size(0)):
                for jj in range(0, input_data.size(1)):
                    qunatized_input[ii][jj], __ = UniformQuantizer.get_optimal_word(
                        input_numpy[ii, jj], codebook)
                    itrVal, quantized_idx = UniformQuantizer.get_optimal_word(
                        input_numpy[ii, jj], tuple(retcodebook))
                    retcodebook[quantized_idx] = retcodebook[quantized_idx] +  CODEBOOK_LR*(input_numpy[ii, jj] - itrVal)

            return qunatized_input, tuple(retcodebook)

        # This function has only a single output, so it gets only one gradient
        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.saved_tensors
            grad_input = grad_output.clone()
            return grad_input, None

    # def __init__(self, M):
    #     super(LearningQuantizerFunction, self).__init__()
    #     self.M = M
    #     self.codebook = Parameter(torch.Tensor(M, 1))
    #
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.codebook.data, _ = torch.sort(torch.randn(self.M, 1))
    #
    # def forward(self, input):
    #     # self.codebook.data, _ = torch.sort(self.codebook.data)
    #     input_value = input.item()
    #     if input_value < self.codebook.data[0]:
    #         return self.codebook.data[0]
    #     elif input_value > self.codebook.data[self.M - 1]:
    #         return self.codebook.data[self.M - 1]
    #     for ii in range(0, self.M - 1):
    #         if(input_value > self.codebook.data[ii] and input_value < self.codebook.data[ii + 1]):
    #             ret = self.codebook.data[ii] if input_value <= ((self.codebook.data[ii + 1] - self.codebook.data[ii])/2) else self.codebook.data[ii + 1]
    #             return ret
    #     return self.codebook.data[0]
    #
    # def extra_repr(self):
    #     return 'M={}'.format(self.M)
