import matlab.engine
eng = matlab.engine.start_matlab()
eng.shlezDataGen('autosave', 'off', 'Directory', '~/Desktop',
                 'filename', 'aleks', nargout=0)
