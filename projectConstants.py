""" Constants file """
import math

################################################################################
###                       Neural Network constants                           ###
################################################################################

BATCH_SIZE = 8
DEFAULT_EPOCHS = 2
INPUT_DIMENSION = 240
OUTPUT_DIMENSION = 80
QUANTIZATION_BITS = 3
M = int(math.pow(2, QUANTIZATION_BITS))
QUANTIZATION_RATE = QUANTIZATION_BITS * OUTPUT_DIMENSION / INPUT_DIMENSION
CODEBOOK_LR = 0.01

# Only uncommented models will be trained and tested
modelsToActivate = [
    'Linear sign quantization',
    # 'Linear uniform codebook',
    # 'Linear SOM learning codebook',
    # 'Analog sign quantization',
    # 'RNN sign quantization',
    # 'LSTM sign quantization'
]

# ------------------------------
# Epochs of each of the models
EPOCHS_lin1 = DEFAULT_EPOCHS
EPOCHS_lin2 = DEFAULT_EPOCHS
EPOCHS_lin3 = DEFAULT_EPOCHS
EPOCHS_lin4 = DEFAULT_EPOCHS
EPOCHS_lin5 = DEFAULT_EPOCHS
EPOCHS_RNN1 = DEFAULT_EPOCHS
EPOCHS_RNN2 = DEFAULT_EPOCHS
