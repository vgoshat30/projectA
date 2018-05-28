"""Generate data using MATLAB

    Runs a MATLAB function that generates data into .mat file that can be
    handeled using dataLoader.py

    See the documentation of shlezDataGen
"""
import matlab.engine

# Run matlab .m script to generate .mat file with test and training data
eng = matlab.engine.start_matlab()
eng.shlezDataGen(nargout=0)
