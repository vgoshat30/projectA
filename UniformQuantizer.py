"""
Here we implement a simple uniform quantizer:

Inputs: input_variance, M
    - **input_variance**: Simple number containing the variance of the input x in R^1 to the scalar quantizers.
    - **M**: The number of codewords expected in the output.

Outputs: codewords
    - **codewords** of shape `(M,1)`: tensor containing the output codewords of the quantizer. This will be the codebook dictionary.

"""
def generate_codebook(input_variance, M):
    # We first divide the R plane into two regions, where the codewords will lay, we use the symmetricity of this quantizers
    codebook = []
    LowerBound = -3*input_variance
    UpperBound = 3*input_variance
    dx = (UpperBound - LowerBound)/(M+1)
    for ii in range(1,M+1):
        codebook.append(LowerBound + ii*dx)
    return tuple(codebook)
