""" Constants file """
import math

################################################################################
###                       Neural Network constants                           ###
################################################################################

BATCH_SIZE = 8
DEFAULT_EPOCHS = 2
INPUT_DIMENSION = 240
OUTPUT_DIMENSION = 80
# QUANTIZATION_BITS = 3
# M = int(math.pow(2, QUANTIZATION_BITS))
# QUANTIZATION_RATE = QUANTIZATION_BITS * OUTPUT_DIMENSION / INPUT_DIMENSION
CODEBOOK_LR = 0.1

# Only uncommented models will be trained and tested
modelsToActivate = [
    # 'Linear sign quantization',
    # 'Linear uniform codebook',
    #'Linear SOM learning codebook',
    # 'Analog sign quantization',
    # 'RNN sign quantization',
    # 'LSTM sign quantization',
    'Tanh quantization'
]

# ------------------------------
# Epochs of each of the models
EPOCHS_linSignQunat = DEFAULT_EPOCHS
EPOCHS_linUniformQunat = DEFAULT_EPOCHS
EPOCHS_linSOMQuant = DEFAULT_EPOCHS
EPOCHS_ADSignQuant = DEFAULT_EPOCHS
EPOCHS_rnnSignQuant = DEFAULT_EPOCHS
EPOCHS_lstmSignQuant = DEFAULT_EPOCHS
EPOCHS_tanhQuantize = DEFAULT_EPOCHS

SLOPE_RANGE = [10, 100, 1000]
EPOCH_RANGE = [2, 5, 7, 10]
LR_RANGE = [0.1, 0.01, 0.001]
M_RANGE = [2, 4, 8, 16]
RATE_RANGE = [1*OUTPUT_DIMENSION / INPUT_DIMENSION, 2*OUTPUT_DIMENSION / INPUT_DIMENSION, 3*OUTPUT_DIMENSION / INPUT_DIMENSION, 4*OUTPUT_DIMENSION / INPUT_DIMENSION]
