""" Constants file """
import math

################################################################################
###                       Neural Network constants                           ###
################################################################################

BATCH_SIZE = 8
INPUT_DIMENSION = 240
OUTPUT_DIMENSION = 80
CODEBOOK_LR = 0.1

# Only uncommented models will be trained and tested
modelsToActivate = [
    # 'Linear sign quantization',
    'Linear uniform codebook',
    #'Linear SOM learning codebook',
    # 'Analog sign quantization',
    # 'RNN sign quantization',
    # 'LSTM sign quantization',
    # 'Tanh quantization',
    'Stretched Tanh quantization'
]

# ------------------------------

SLOPE_RANGE = [10, 20, 30, 100, 1000]
EPOCH_RANGE = [2, 5, 7]
LR_RANGE = [0.2, 0.1, 0.05, 0.01]
STRETCH_RANGE = [2, 5, 10]
M_RANGE = [2, 3, 4, 5, 6, 7, 8]
