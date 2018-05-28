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
pointsColor = 'orange'


def reandomTickPos(x, y):
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
        formatted
            Formatted str (or numpy array - its casted to string anyway) to be
            cleared.

        Returns
        -------
        unformatted
            Unformatted string.

        Example
        -------
        ans = removeCellFormat(['hello'])
        print(ans)
        >>> hello
    '''
    unformatted = str(formatted).replace("[", "")
    unformatted = unformatted.replace("]", "")
    unformatted = unformatted.replace("'", "")
    return unformatted


def getTextAlignment(x, y, textOffset):
    '''Set the textbox alignment according to its position on the graph

        Parameters
        ----------
        x   y
            x and y values of the datapoint

        Returns
        -------
        (ha, va, xOffset, yOffset)
            Horizontal alignment and vertical alignment and offset of tooltop
            from the datapoint
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


def logResult(rate, error, *handleMethod, **kwargs):
    '''Plot the result compares to the theory

        IMPORTANT!!!
            The function opens a figure which will pause all your code until
            closed! If you want the code to continue, use 'dontshow' (see below)

        Opens a new figure, plots all the theoretical bounds and add a point at
        (rate, error). Also showing all previously saved results and enumerates
        them on the plot itself.
        Click any datapoint to oped a tooltip with all the information available
        for it.

        Parameters
        ----------
            rate

                The quantization rate of the current test

            error

                The resulting error of the current test

            optional (add as a string input, PLACE BEFORE algorithm and runtime):

                'dontshow'

                    Only saves the new test results and dont display plot

                'dontsave'

                    Only displays results and dont save results

            algorithm

                Spesify a string with the name of the algorithm (preferably a
                short one) for example:

                    logResult(0.3, 0.04, algorithm='Simple Linear')

            runtime

                The runtime of the algorithm. Accepts only timedelta types of
                the datetime packege

            epochs

                Specify number of training epochs
    '''

    def pick_handler(event):
        '''Handles the choosing of plotted results
        '''
        # Get the pressed artist
        artist = event.artist

        if artist.get_marker() is 'o':
            clearDatatips(event, -1)
            return

        # Mark the chosen point
        artist.set(marker='o', markersize=chosenMarkersize, color='r')

        # Handle chosen result:
        chosenIndex = resList.index(artist)
        # Hide current result index
        indexText[chosenIndex].set(alpha=0)
        # Show chosen texbox
        textBoxes[chosenIndex] = dict(boxstyle='round', facecolor='wheat',
                                      alpha=textboxAlpha)
        # Show chosen datatip
        datatips[chosenIndex].set(alpha=datatipAlpha,
                                  bbox=textBoxes[chosenIndex])

        # Clear other dataTips
        clearDatatips(event, chosenIndex)

        # Update figure
        fig.canvas.draw()
        fig.canvas.flush_events()

    def clearDatatips(event, exeptInex):
        '''Clears all datatips
        '''

        for ii in range(0, len(resList)):
            if ii is exeptInex:
                continue
            # Unmark all other points
            resList[ii].set(marker='x', markersize=regMarkerSize,
                            color=pointsColor)
            # Show all result indecies
            indexText[ii].set(alpha=indexAlpha)
            # Hide all textBoxes
            textBoxes[ii] = dict(boxstyle='round', facecolor='wheat',
                                 alpha=0)
            # Hide all result datatips
            datatips[ii].set(alpha=0, bbox=textBoxes[ii])

        # Update figure
        fig.canvas.draw()
        fig.canvas.flush_events()

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

    # Create fill vectors
    xFill = np.concatenate((v_fRate[0], np.flip(v_fRate[0], 0)), axis=0)
    yFill = np.concatenate((m_fCurves[5, :], np.flip(m_fCurves[1, :], 0)),
                           axis=0)

    # Check all var args
    algToSave = np.append(algorithmName, '')
    timeToSave = np.append(timeResults, np.array(str(datetime.now())))
    runTimeToSave = np.append(runTimeResults, '')
    epochsToSave = np.append(trainEpochs, np.empty((1, 1), float))
    for key in kwargs:
        # Check if an algorithm name provided
        if key is 'algorithm':
            algToSave[-1] = kwargs[key]
        # Check if runtime provided
        if key is 'runtime':
            runTimeToSave[-1] = str(kwargs[key])
        if key is 'epochs':
            epochsToSave[-1] = kwargs[key]

    if not(('dontsave' in handleMethod)):
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
                                  'trainEpochs': epochsToSave})
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
        resList += ax.plot(rate, error, marker='x',
                           markersize=regMarkerSize, color=pointsColor,
                           label='Results', picker=5)

        indexText = []
        datatips = []
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
                currRuntime = datetime.strptime(removeCellFormat(runTimeToSave[ii]),
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
            datatips.append(ax.text(resList[ii].get_xdata()+textAlign[2],
                                    resList[ii].get_ydata()+textAlign[3],
                                    textToDisplay,
                                    alpha=0, fontsize=dataTipFontsize,
                                    bbox=textBoxes[ii],
                                    ha=textAlign[0], va=textAlign[1]))

        # Last result index text
        lastResultIndex = rateResults.shape[1]
        indexText.append(ax.text(*reandomTickPos(rate, error),
                                 lastResultIndex+1,
                                 fontsize=8, alpha=indexAlpha,
                                 verticalalignment='center',
                                 horizontalalignment='center'))
        # CLast textbox
        textBoxes.append(dict(boxstyle='round', facecolor='wheat',
                              alpha=0))
        currIterDateTime = datetime.strptime(timeToSave[-1],
                                             '%Y-%m-%d %H:%M:%S.%f')
        if runTimeToSave[-1]:
            currRuntime = datetime.strptime(removeCellFormat(runTimeToSave[-1]),
                                            '%H:%M:%S.%f')
        # Last result datatip
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
        datatips.append(ax.text(resList[lastResultIndex].get_xdata() +
                                textAlign[2],
                                resList[lastResultIndex].get_ydata() +
                                textAlign[3], textToDisplay,
                                alpha=0, fontsize=dataTipFontsize,
                                bbox=textBoxes[lastResultIndex],
                                ha=textAlign[0], va=textAlign[1]))

        # Labeling and graph appearance
        plt.xlabel('Rate', fontsize=18, fontname='Times New Roman')
        plt.ylabel('Average Distortion', fontsize=18, fontname='Times New Roman')
        ax.legend(fontsize=6)
        ax.autoscale(enable=True, axis='x', tight=True)
        # Show figure
        plt.show()


def deleteResult(*args, **kwargs):
    '''Delete all or part of the data log

        Parameters
        ----------
            optional (add as a string input, PLACE IN THE BEGINNING):

                'clear'
                    Reaets all the log

            index
                Specify index of test to delete. For example:
                    deleteResult(index=1)
                deletes the second test
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

    for key in kwargs:
        if key is 'index':
            rateResults = np.delete(rateResults, kwargs[key])
            errorResults = np.delete(errorResults, kwargs[key])
            runTimeResults = np.delete(runTimeResults, kwargs[key])
            timeResults = np.delete(timeResults, kwargs[key])
            algorithmName = np.delete(algorithmName, kwargs[key])
            trainEpochs = np.delete(trainEpochs, kwargs[key])

    if 'clear' in args:
        rateResults = np.empty((0, 1), float)
        errorResults = np.empty((0, 1), float)
        runTimeResults = np.empty((0, 1), object)
        timeResults = ''
        algorithmName = np.empty((0, 1), object)
        trainEpochs = np.empty((0, 1), float)

    # Save data back to mat file
    sio.savemat(matFileName, {'m_fCurves': m_fCurves,
                              'v_fRate': v_fRate,
                              'rateResults': rateResults,
                              'errorResults': errorResults,
                              'time': timeResults,
                              'algorithmName': algorithmName,
                              'runTime': runTimeResults,
                              'trainEpochs': trainEpochs})
