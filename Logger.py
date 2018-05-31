'''Keeps a log of test results

This module is used to save, view, edit or delete test result into a .mat MATLAB
file. It does not contain any calculations nor learning algorithms and deals
only with menaging the log and user inteface.

Notes
-----
    Functions intendes to be used outside of the module:
        - log(rate, error, *handleMethod, **kwargs)
        - delete(test)
        - edit(test, **kwargs)
        - content(test)
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from cycler import cycler

matFileName = 'testLog.mat'

chosenMarkersize = 2
regMarkerSize = 5
indexAlpha = 0.5
datatipAlpha = 1
dataTipFontsize = 6
textboxAlpha = 0.8
textOffset = 0.015
sizeOfFigure = (8, 5)  # in inches

# Define which lines to plot
whichToPlot = [1,  # No quantization
               1,  # Asymptotic optimal task-based
               0,  # LBG task-based
               1,  # Asymptotic optimal task-ignorant
               0,  # LBG task-ignorant
               1]  # Hardware limited upper bound

# Set the legend labels
labels = ['No quantization',
          'Asymptotic optimal task-based',
          'LBG task-based',
          'Asymptotic optimal task-ignorant',
          'LBG task-ignorant',
          'Hardware limited upper bound']

markers = ['x', '', '', '', '', '']
lineStyles = [':', '--', '-', '--', '-', '--']
linecolors = ['black', 'red', 'red', 'blue', 'red', 'lime']
lineWidths = [1, 1, 1, 1, 1, 1.5]
markerSizes = [4, 1, 1, 1, 1, 1]
markerLinewidths = [1, 1, 1, 1, 1, 1]
pointMarker = 'x'
pointsColor = 'orange'
chosenMarker = 'o'
chosenColor = 'red'
tooltipBoxStyle = 'round'
tooltipBoxColor = 'wheat'


def reandomTickPos(x, y):
    """Add a random position "noise" to a test number attached to a point

    Parameters
    ----------
    x: numpy.float64
        X position of a test log(rate)
    y: numpy.float64
        Y position of a test log(average loss)

    Returns
    -------
    (X_new, Y_new): tuple
        The input X and Y arguments, with added noise
    """

    noise = 0.03
    bias = 0.4
    randX = 2*random.random() - 1
    while abs(randX) < bias:
        randX = 2*random.random() - 1
    randY = 2*random.random() - 1
    while abs(randY) < bias:
        randY = 2*random.random() - 1
    return x+noise*randX, y+noise*randY


def removeCellFormat(formatted):
    '''Clear a string formatting caused by saving as cell in mat file

        Parameters
        ----------
        formatted : numpy.ndarray
            An ndarray containing a sting formatted as: ['stuff']

        Returns
        -------
        unformatted : str
            The string contained in the input, without the characters: [  '  ]

        Example
        -------
        >>> import numpy as np
        >>> from Logger import *
        >>> dirty = np.append("['first']", "['second']")
        >>> print(dirty[0])
        ['first']
        >>> clean = removeCellFormat(dirty[0])
        >>> print(clean)
        first
    '''
    unformatted = str(formatted).replace("[", "")
    unformatted = unformatted.replace("]", "")
    unformatted = unformatted.replace("'", "")
    return unformatted


def getTextAlignment(x, y, textOffset):
    '''Set tooltip textbox alignment and its offset from the corresponding point
        on the graph

        Parameters
        ----------
        x : numpy.ndarray
            X position of a test log (rate)
        y : numpy.ndarray
            Y position of a test log (average loss)
        textOffset : float
            The absolute value of the offset of the tooltip from the point

        Returns
        -------
        (ha, va, xOffset, yOffset) : tuple
            ha : str
                Horizontal alignment as accepted by the text property:
                horizontalalignment of the matplotlib.
                Possible values: [ 'right' | 'left' ]
            ha : str
                Vertical alignment as accepted by the text property:
                verticalalignment of the matplotlib.
                Possible values: [ 'top' | 'bottom' ]
            xOffset : float
                The X axis offset of the textbox
            yOffset : float
                The Y axis offset of the textbox
    '''
    axs = plt.gca()
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()

    if x < xlim[1] / 2:
        ha = 'left'
        xOffset = textOffset
    else:
        ha = 'right'
        xOffset = -textOffset

    if y < ylim[1]/2:
        va = 'bottom'
        yOffset = textOffset
    else:
        va = 'top'
        yOffset = -textOffset

    return (ha, va, xOffset, yOffset)


def log(rate=None, error=None, *handleMethod, **kwargs):
    '''Plot a test result (rate and loss) in respect to theoretical graphs

        IMPORTANT!!!
            The function opens a figure which will pause all your code until
            closed! If you want the code to continue, use 'dontshow' (see below)

        The function opens a new figure, plots all the theoretical bounds and
        adds a point at (rate, error) of the current test.
        Also, all previously saved results are plotted and enumerated according
        to their logging order.
        Click any datapoint to open a tooltip with additional imformation.
        To see all imformation available for a test, see help for the function
        content(test)

        Parameters
        ----------
            When called with no args, plots all previously saved tests

            rate : float
                The quantization rate of the current test
            error : float
                The resulting average loss of the current test
            *args
                'dontshow'
                    Only saving the test result and not openning a figure
                'dontsave'
                    Only plotting result and not saving it
            **kwargs
                algorithm : str
                    The name of the algorithm used (preferably a short one)
                runtime : datetime.timedelta
                    The runtime of the algorithm
                epochs : int
                    Number of training epochs
                note : str
                    A note regarding the test (can be accessed only trough the
                    function content(test))

        Example
        -------
        >>> from datetime import datetime
        >>> import Logger as log
        >>> start = datetime.now()
        >>> end = datetime.now() - start
        >>> log.log(0.2, 0.6, 'dontshow', algorithm='SOM', runtime=end, epochs=2,
        ...         note='This is an exaple of the log function')
        Saved result of test number 1
    '''

    def pick_handler(event):
        '''Handles the choosing of plotted results
        '''
        # Get the pressed artist
        artist = event.artist

        # If the clicked point was already chosen, clear all tooltips and return
        if artist.get_marker() is chosenMarker:
            clearDatatips(event, -1)
            return

        # Mark the chosen point
        artist.set(marker=chosenMarker, markersize=chosenMarkersize,
                   color=chosenColor)

        # Get the index of the clicked point in the test log
        chosenIndex = resList.index(artist)
        # Hide current result index
        indexText[chosenIndex].set(alpha=0)
        # Show chosen texbox
        textBoxes[chosenIndex] = dict(boxstyle=tooltipBoxStyle,
                                      facecolor=tooltipBoxColor,
                                      alpha=textboxAlpha)
        # Show chosen tooltip
        tooltips[chosenIndex].set(alpha=datatipAlpha,
                                  bbox=textBoxes[chosenIndex])

        # Clear other tooltips
        clearDatatips(event, chosenIndex)

        # Update figure
        fig.canvas.draw()
        fig.canvas.flush_events()

    def clearDatatips(event, exeptInex):
        '''Clears tooltips

            Parameters
            ----------
            event : matplotlib.backend_bases.PickEvent
                Callback pick event
            exeptInex : int
                Dont clear the tooltip of the point at the specified index
                Pass -1 to clear all tooltips
        '''

        for ii in range(0, len(resList)):
            if ii is exeptInex:
                continue
            # Unmark all other points
            resList[ii].set(marker=pointMarker, markersize=regMarkerSize,
                            color=pointsColor)
            # Show all result indecies
            indexText[ii].set(alpha=indexAlpha)
            # Hide all textBoxes
            textBoxes[ii] = dict(boxstyle=tooltipBoxStyle,
                                 facecolor=tooltipBoxColor,
                                 alpha=0)
            # Hide all result tooltips
            tooltips[ii].set(alpha=0, bbox=textBoxes[ii])

        # Update figure
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Load data from file
    theoryBounds = sio.loadmat(matFileName)

    # Get x (rate) vector and y (errors) matrix and previous results rate and
    # errors
    m_fCurves = theoryBounds['m_fCurves']
    v_fRate = theoryBounds['v_fRate']
    rateResults = theoryBounds['rateResults']
    errorResults = theoryBounds['errorResults']
    runTimeResults = theoryBounds['runTime']
    timeResults = theoryBounds['time']
    algorithmName = theoryBounds['algorithmName']
    trainEpochs = theoryBounds['trainEpochs']
    notes = theoryBounds['notes']

    # Create fill vectors
    xFill = np.concatenate((v_fRate[0], np.flip(v_fRate[0], 0)), axis=0)
    yFill = np.concatenate((m_fCurves[5, :], np.flip(m_fCurves[1, :], 0)),
                           axis=0)

    # Check all var args
    algToSave = np.append(algorithmName, '')
    timeToSave = np.append(timeResults, np.array(str(datetime.now())))
    runTimeToSave = np.append(runTimeResults, '')
    epochsToSave = np.append(trainEpochs, np.empty((1, 1), float))
    notesToSave = np.append(notes, '')
    for key in kwargs:
        # Check if an algorithm name provided
        if key is 'algorithm':
            algToSave[-1] = kwargs[key]
        # Check if runtime provided
        if key is 'runtime':
            runTimeToSave[-1] = str(kwargs[key])
        if key is 'epochs':
            epochsToSave[-1] = kwargs[key]
        if key is 'note':
            notesToSave[-1] = kwargs[key]

    if not(('dontsave' in handleMethod)) and not(rate is None) and \
            not(error is None):
            # Append the results to the mat file
        sio.savemat(matFileName, {'m_fCurves': m_fCurves,
                                  'v_fRate': v_fRate,
                                  'rateResults': np.append(rateResults,
                                                           np.array(rate)),
                                  'errorResults': np.append(errorResults,
                                                            np.array(error)),
                                  'time': timeToSave,
                                  'algorithmName': algToSave,
                                  'runTime': runTimeToSave,
                                  'trainEpochs': epochsToSave,
                                  'notes': notesToSave})
        print('Saved result of test number', rateResults.shape[1]+1)

    # Display the results in respect to the theoretical bounds
    if not(('dontshow' in handleMethod)):
        plt.close('all')

        # Create figure and ge axes handle
        fig = plt.figure(figsize=sizeOfFigure)
        ax = fig.add_subplot(111)

        # Connect figure to callback
        fig.canvas.mpl_connect('pick_event', pick_handler)
        fig.canvas.mpl_connect('figure_leave_event', lambda event,
                               temp=-1: clearDatatips(event, temp))

        # Plot all theoretical bounds
        for ii in range(0, m_fCurves.shape[0]):
            if whichToPlot[ii]:
                ax.plot(v_fRate[0], m_fCurves[ii, :], label=labels[ii],
                        marker=markers[ii], color=linecolors[ii],
                        linestyle=lineStyles[ii], linewidth=lineWidths[ii],
                        markersize=markerSizes[ii])

        # Plot fill
        ax.fill(xFill, yFill, c='c', alpha=0.3)

        # Plot previous results
        resList = ax.plot(rateResults, errorResults, marker='x',
                          markersize=regMarkerSize,
                          color=pointsColor, picker=5)

        # Plot result
        if not((rate is None)) and not((error is None)):
            resList += ax.plot(rate, error, marker='x',
                               markersize=regMarkerSize, color=pointsColor,
                               label='Results', picker=5)

        indexText = []
        tooltips = []
        textBoxes = []
        for ii in range(0, rateResults.shape[1]):
            # Enumerate result points in the figure
            indexText.append(ax.text(*reandomTickPos(rateResults[0, ii],
                                                     errorResults[0, ii]),
                                     ii+1, fontsize=8, alpha=indexAlpha,
                                     verticalalignment='center',
                                     horizontalalignment='center'))
            # Create all textboxes and dont display them
            textBoxes.append(dict(boxstyle='round', facecolor='wheat',
                                  alpha=0))

            # Create all data tooltip and dont display them

            currIterDateTime = datetime.strptime(timeToSave[ii],
                                                 '%Y-%m-%d %H:%M:%S.%f')
            if runTimeToSave[ii]:
                currRuntime = datetime.strptime(removeCellFormat(
                    runTimeToSave[ii]),
                    '%H:%M:%S.%f')
            if algToSave[ii] and runTimeToSave[ii]:
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nAlgorithm:\n' + removeCellFormat(algToSave[ii]) + \
                    '\nRuntime: ' + currRuntime.strftime('%H:%M:%S.%f') + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            elif algToSave[ii] and not(runTimeToSave[ii]):
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nAlgorithm:\n' + removeCellFormat(algToSave[ii]) + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            elif not(algToSave[ii]) and runTimeToSave[ii]:
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nRuntime: ' + currRuntime.strftime('%H:%M:%S.%f') + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            else:
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            textAlign = getTextAlignment(resList[ii].get_xdata(),
                                         resList[ii].get_ydata(),
                                         textOffset)
            tooltips.append(ax.text(resList[ii].get_xdata()+textAlign[2],
                                    resList[ii].get_ydata()+textAlign[3],
                                    textToDisplay,
                                    alpha=0, fontsize=dataTipFontsize,
                                    bbox=textBoxes[ii],
                                    ha=textAlign[0], va=textAlign[1]))

        if not((rate is None)) and not((error is None)):
            # Last result index text
            lastResultIndex = rateResults.shape[1]
            indexText.append(ax.text(*reandomTickPos(rate, error),
                                     lastResultIndex+1,
                                     fontsize=8, alpha=indexAlpha,
                                     verticalalignment='center',
                                     horizontalalignment='center'))
            # Last textbox
            textBoxes.append(dict(boxstyle='round', facecolor='wheat',
                                  alpha=0))
            currIterDateTime = datetime.strptime(timeToSave[-1],
                                                 '%Y-%m-%d %H:%M:%S.%f')
            if runTimeToSave[-1]:
                currRuntime = datetime.strptime(removeCellFormat(
                    runTimeToSave[-1]),
                    '%H:%M:%S.%f')
            # Last result tooltip
            if algToSave[-1] and runTimeToSave[-1]:
                textToDisplay = 'Rate: ' + str(rate) + \
                    '\nAvg. Distortion: ' + str(error) + \
                    '\nAlgorithm:\n' + removeCellFormat(algToSave[-1]) + \
                    '\nRuntime: ' + currRuntime.strftime('%H:%M:%S.%f') + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            elif algToSave[-1] and not(runTimeToSave[-1]):
                textToDisplay = 'Rate: ' + str(rate) + \
                    '\nAvg. Distortion: ' + str(error) + \
                    '\nAlgorithm:\n' + removeCellFormat(algToSave[-1]) + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            elif not(algToSave[-1]) and runTimeToSave[-1]:
                textToDisplay = 'Rate: ' + str(rate) + \
                    '\nAvg. Distortion: ' + str(error) + \
                    '\nRuntime: ' + currRuntime.strftime('%H:%M:%S.%f') + \
                    '\nDate: ' + datetime.now().strftime('%d/%m/%y') + \
                    '\nTime: ' + datetime.now().strftime('%H:%M:%S')
            else:
                textToDisplay = 'Rate: ' + str(rate) + \
                    '\nAve. Distortion: ' + str(error) + \
                    '\nDate: ' + datetime.now().strftime('%d/%m/%y') + \
                    '\nTime: ' + datetime.now().strftime('%H:%M:%S')
            textAlign = getTextAlignment(resList[lastResultIndex].get_xdata(),
                                         resList[lastResultIndex].get_ydata(),
                                         textOffset)
            tooltips.append(ax.text(resList[lastResultIndex].get_xdata() +
                                    textAlign[2],
                                    resList[lastResultIndex].get_ydata() +
                                    textAlign[3], textToDisplay,
                                    alpha=0, fontsize=dataTipFontsize,
                                    bbox=textBoxes[lastResultIndex],
                                    ha=textAlign[0], va=textAlign[1]))

        # Labeling and graph appearance
        plt.xlabel('Rate', fontsize=18, fontname='Times New Roman')
        plt.ylabel('Average Distortion', fontsize=18,
                   fontname='Times New Roman')
        ax.legend(fontsize=6)
        ax.autoscale(enable=True, axis='x', tight=True)
        # Show figure
        plt.show()


def delete(test=None):
    '''Delete specific test or clear all test log

        Parameters
        ----------
            test : int OR tuple of int OR nothing
                If type(test) is int
                    Deleting specified test
                If type(test) is tuple
                    Deleting all test specified in the tuple
                If test is empty, clearing all test log

        Example
        -------
        >>> import Logger as log
        >>> log.log(0.1, 0.5, 'dontshow')
        Saved result of test number 1
        >>> log.log(0.1, 0.4, 'dontshow')
        Saved result of test number 2
        >>> log.log(0.3, 0.2, 'dontshow')
        Saved result of test number 3
        >>> log.log(0.4, 0.5, 'dontshow')
        Saved result of test number 4
        >>> log.delete(4)
        Deleted test nuber 4
        >>> log.delete((1, 3))
        Deleted tests: (1, 3)
        >>> log.delete()
        Cleared test log
    '''
    # Load data from file
    theoryBounds = sio.loadmat(matFileName)

    # Get x (rate) vector and y (errors) matrix and previous results rate and
    # errors
    m_fCurves = theoryBounds['m_fCurves']
    v_fRate = theoryBounds['v_fRate']
    rateResults = theoryBounds['rateResults']
    errorResults = theoryBounds['errorResults']
    runTimeResults = theoryBounds['runTime']
    timeResults = theoryBounds['time']
    algorithmName = theoryBounds['algorithmName']
    trainEpochs = theoryBounds['trainEpochs']
    notes = theoryBounds['notes']

    # If specified number of test(s)
    if not(test is None):
        if type(test) is tuple:
            testIndex = ()
            for i, testNum in enumerate(test):
                testIndex += (testNum - 1,)
            print('Deleted tests:', test)
        else:
            testIndex = test - 1
            print('Deleted test nuber', test)

        rateResults = np.delete(rateResults, testIndex, 1)
        errorResults = np.delete(errorResults, testIndex, 1)
        runTimeResults = np.delete(runTimeResults, testIndex, 1)
        timeResults = np.delete(timeResults, testIndex)
        algorithmName = np.delete(algorithmName, testIndex, 1)
        trainEpochs = np.delete(trainEpochs, testIndex, 1)
        notes = np.delete(notes, testIndex, 1)
    else:
        rateResults = np.empty((0, 1), float)  # MATLAB Array of doubles
        errorResults = np.empty((0, 1), float)  # MATLAB Array of doubles
        runTimeResults = np.empty((0, 1), object)  # MATLAB Cell
        timeResults = ''  # MATLAB Char array
        algorithmName = np.empty((0, 1), object)  # MATLAB Cell
        trainEpochs = np.empty((0, 1), float)  # MATLAB Array of doubles
        notes = np.empty((0, 1), object)  # MATLAB Cell
        print('Cleared test log')

    # Save data back to mat file
    sio.savemat(matFileName, {'m_fCurves': m_fCurves,
                              'v_fRate': v_fRate,
                              'rateResults': rateResults,
                              'errorResults': errorResults,
                              'time': timeResults,
                              'algorithmName': algorithmName,
                              'runTime': runTimeResults,
                              'trainEpochs': trainEpochs,
                              'notes': notes})


def edit(test, **kwargs):
    '''Edit test log

        Parameters
        ----------
            test
                Number of test to edit

            **kwargs
                algorithm
                    New algorithm name
                epochs
                    New train epochs number
                note
                    New note

            Example
            -------
            >>> import Logger as log
            >>> log.log(0.2, 0.6, 'dontshow')
            Saved result of test number 1
            >>> log.edit(1, algorithm='RNN')
            Changed algorithm of test number 1 to: RNN
    '''
    # Load data from file
    theoryBounds = sio.loadmat(matFileName)

    # Get x (rate) vector and y (errors) matrix and previous results rate and errors
    m_fCurves = theoryBounds['m_fCurves']
    v_fRate = theoryBounds['v_fRate']
    rateResults = theoryBounds['rateResults']
    errorResults = theoryBounds['errorResults']
    runTimeResults = theoryBounds['runTime']
    timeResults = theoryBounds['time']
    algorithmName = theoryBounds['algorithmName']
    trainEpochs = theoryBounds['trainEpochs']
    notes = theoryBounds['notes']

    for key in kwargs:
        if key is 'algorithm':
            algorithmName[0, test-1] = kwargs[key]

        if key is 'epochs':
            trainEpochs[0, test-1] = kwargs[key]

        if key is 'note':
            notes[0, test-1] = kwargs[key]

    print('Changed', key, 'of test number', test, 'to:', kwargs[key])

    # Save data back to mat file
    sio.savemat(matFileName, {'m_fCurves': m_fCurves,
                              'v_fRate': v_fRate,
                              'rateResults': rateResults,
                              'errorResults': errorResults,
                              'time': timeResults,
                              'algorithmName': algorithmName,
                              'runTime': runTimeResults,
                              'trainEpochs': trainEpochs,
                              'notes': notes})


def content(test):
    '''Print all information available for specified test

        Parameters
        ----------
            test : int
                Number of test to print data for

        Example
        -------
        >>> import Logger as log
        >>> log.log(0.3, 0.4, 'dontshow', algorithm='Best One', epochs=10,
        ...         note='Used for an example of the content(test) function')
        Saved result of test number 1
        >>> log.content(1)

        \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

        Test 1 Info

        Rate:		0.3
        Loss:		0.4
        Algorithm:	Best One
        Train Runtime:	________________
        Train Epochs:	10.0
        Logging Time:	2018-05-31 13:50:17.117906
        Note:		Used for an example of the content(test) function

        \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    '''

    dontExistMessage = '________________'

    # Load data from file
    theoryBounds = sio.loadmat(matFileName)

    # Get x (rate) vector and y (errors) matrix and previous results rate and errors
    m_fCurves = theoryBounds['m_fCurves']
    v_fRate = theoryBounds['v_fRate']
    rateResults = theoryBounds['rateResults']
    errorResults = theoryBounds['errorResults']
    runTimeResults = theoryBounds['runTime']
    timeResults = theoryBounds['time']
    algorithmName = theoryBounds['algorithmName']
    trainEpochs = theoryBounds['trainEpochs']
    notes = theoryBounds['notes']

    if not(type(test) is tuple):
        testNum = (test,)
    else:
        testNum = test

    for i in testNum:
        if not algorithmName[0, i-1]:
            algToPrint = dontExistMessage
        else:
            algToPrint = removeCellFormat(algorithmName[0, i-1])

        if not runTimeResults[0, i-1]:
            runtimeToPrint = dontExistMessage
        else:
            runtimeToPrint = removeCellFormat(runTimeResults[0, i-1])

        if not trainEpochs[0, i-1]:
            epochToPrint = dontExistMessage
        else:
            epochToPrint = trainEpochs[0, i-1]

        if not notes[0, i-1]:
            noteToPrint = dontExistMessage
        else:
            noteToPrint = removeCellFormat(notes[0, i-1])

        print('\n\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\n\n'
              'Test {} Info\n\n'
              'Rate:\t\t{}\nLoss:\t\t{}\nAlgorithm:\t{}\n'
              'Train Runtime:\t{}\nTrain Epochs:\t{}\nLogging Time:\t{}\n'
              'Note:\t\t{}'
              '\n\n\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\n'
              .format(i, rateResults[0, i-1], errorResults[0, i-1],
                      algToPrint, runtimeToPrint,
                      epochToPrint, timeResults[i-1], noteToPrint))
