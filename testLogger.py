import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime


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


def logResult(rate, error, *handleMethod, **kwargs):
    '''Plot the result compares to the theory

        Opens a new figure, plots all the theoretical bounds and add a point at
        (rate, error). Also showing all previously saved results and enumerates
        them on the plot itself.

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

                Specify a string with the runtime of the algorithm
    '''

    def pick_handler(event):
        '''Handles the choosing of plotted results
        '''
        # Get the pressed artist
        artist = event.artist

        if artist.get_marker is 'o':
            clearDatatips(event)
            return

        clearDatatips(event)

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

        # Update figure
        fig.canvas.draw()
        fig.canvas.flush_events()

    def clearDatatips(event):
        '''Clears all datatips
        '''

        for ii in range(0, len(resList)):
            # Unmark all other points
            resList[ii].set(marker='x', markersize=regMarkerSize,
                            color='orange')
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

    matFileName = 'testLog.mat'

    chosenMarkersize = 2
    regMarkerSize = 5
    indexAlpha = 0.5
    datatipAlpha = 1
    dataTipFontsize = 6
    textboxAlpha = 0.8
    textOffset = 0.015

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

    # Create fill vectors
    xFill = np.concatenate((v_fRate[0], np.flip(v_fRate[0], 0)), axis=0)
    yFill = np.concatenate((m_fCurves[5, :], np.flip(m_fCurves[1, :], 0)),
                           axis=0)

    # Check all var args
    algToSave = np.append(algorithmName, '')
    runTimeToSave = np.append(runTimeResults, '')
    for key in kwargs:
        # Check if an algorithm name provided
        if key is 'algorithm':
            algToSave[-1] = kwargs[key]
        # Check if runtime provided
        if key is 'runtime':
            runTimeToSave[-1] = kwargs[key]

    if not(('dontsave' in handleMethod)):
        # Append the results to the mat file
        sio.savemat(matFileName, {'m_fCurves': m_fCurves,
                                  'v_fRate': v_fRate,
                                  'rateResults': np.append(rateResults,
                                                           np.array(rate)),
                                  'errorResults': np.append(errorResults,
                                                            np.array(error)),
                                  'time': np.append(timeResults,
                                                    np.array(str(datetime.now()))),
                                  'algorithmName': algToSave,
                                  'runTime': runTimeToSave})
        print('Saved result of test number', rateResults.shape[1]+1)

    # Display the results in respect to the theoretical bounds
    if not(('dontshow' in handleMethod)):
        plt.close('all')

        # Create figure and ge axes handle
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Connect figure to callback
        fig.canvas.mpl_connect('pick_event', pick_handler)
        fig.canvas.mpl_connect('figure_leave_event', clearDatatips)

        # Plot all theoretical bounds
        for ii in range(0, m_fCurves.shape[0]):
            if whichToPlot[ii]:
                ax.plot(v_fRate[0], m_fCurves[ii, :], label=labels[ii])

        # Plot fill
        ax.fill(xFill, yFill, c='c', alpha=0.3)

        # Plot previous results
        resList = ax.plot(rateResults, errorResults, marker='x',
                          markersize=regMarkerSize,
                          color="orange", picker=5)

        # Plot result
        resList += ax.plot(rate, error, marker='x',
                           markersize=regMarkerSize, color="orange",
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
            currIterDateTime = datetime.strptime(timeResults[ii],
                                                 '%Y-%m-%d %H:%M:%S.%f')
            if algToSave[ii] and runTimeToSave[ii]:
                algName = str(algToSave[ii]).replace("[", "")
                algName = algName.replace("]", "")
                algName = algName.replace("'", "")
                runT = str(runTimeToSave[ii]).replace("[", "")
                runT = runT.replace("]", "")
                runT = runT.replace("'", "")
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nAlgorithm:\n' + algName + \
                    '\nRuntime: ' + runT + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            elif algToSave[ii] and not(runTimeToSave[ii]):
                algName = str(algToSave[ii]).replace("[", "")
                algName = algName.replace("]", "")
                algName = algName.replace("'", "")
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nAlgorithm:\n' + algName + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            elif not(algToSave[ii]) and runTimeToSave[ii]:
                runT = str(runTimeToSave[ii]).replace("[", "")
                runT = runT.replace("]", "")
                runT = runT.replace("'", "")
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nRuntime: ' + runT + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
            else:
                textToDisplay = 'Rate: ' + str(rateResults[0, ii]) + \
                    '\nAvg. Distortion: ' + str(errorResults[0, ii]) + \
                    '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                    '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')

            currIterDateTime.strftime('%H:%M:%S')
            datatips.append(ax.text(resList[ii].get_xdata()+textOffset,
                                    resList[ii].get_ydata()+textOffset,
                                    textToDisplay,
                                    alpha=0, fontsize=dataTipFontsize,
                                    bbox=textBoxes[ii]))

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
        # Last result datatip
        if algToSave[-1] and runTimeToSave[-1]:
            algName = str(algToSave[-1]).replace("[", "")
            algName = algName.replace("]", "")
            algName = algName.replace("'", "")
            runT = str(runTimeToSave[-1]).replace("[", "")
            runT = runT.replace("]", "")
            runT = runT.replace("'", "")
            textToDisplay = 'Rate: ' + str(rate) + \
                '\nAvg. Distortion: ' + str(error) + \
                '\nAlgorithm:\n' + algName + \
                '\nRuntime: ' + runT + \
                '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
        elif algToSave[-1] and not(runTimeToSave[-1]):
            algName = str(algToSave[-1]).replace("[", "")
            algName = algName.replace("]", "")
            algName = algName.replace("'", "")
            textToDisplay = 'Rate: ' + str(rate) + \
                '\nAvg. Distortion: ' + str(error) + \
                '\nAlgorithm:\n' + algName + \
                '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
        elif not(algToSave[-1]) and runTimeToSave[-1]:
            runT = str(runTimeToSave[-1]).replace("[", "")
            runT = runT.replace("]", "")
            runT = runT.replace("'", "")
            textToDisplay = 'Rate: ' + str(rate) + \
                '\nAvg. Distortion: ' + str(error) + \
                '\nRuntime: ' + runT + \
                '\nDate: ' + datetime.now().strftime('%d/%m/%y') + \
                '\nTime: ' + datetime.now().strftime('%H:%M:%S')
        else:
            textToDisplay = 'Rate: ' + str(rate) + \
                '\nAve. Distortion: ' + str(error) + \
                '\nDate: ' + datetime.now().strftime('%d/%m/%y') + \
                '\nTime: ' + datetime.now().strftime('%H:%M:%S')
        datatips.append(ax.text(resList[lastResultIndex].get_xdata() +
                                textOffset,
                                resList[lastResultIndex].get_ydata() +
                                textOffset, textToDisplay,
                                alpha=0, fontsize=dataTipFontsize,
                                bbox=textBoxes[lastResultIndex]))

        # Labeling and graph appearance
        plt.xlabel('Rate', fontsize=18, fontname='Times New Roman')
        plt.ylabel('Average Distortion', fontsize=18, fontname='Times New Roman')
        ax.legend(fontsize=8)
        ax.autoscale(enable=True, axis='x', tight=True)
        # Show figure
        plt.show()
