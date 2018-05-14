import matlab.engine

# Run matlab .m script to generate .mat file with test and training data
eng = matlab.engine.start_matlab()
eng.shlezDataGen(nargout=0)
