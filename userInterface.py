def trainHeding():
    print('\n\n=========================',
          '\n\tTRAINING\n'
          '=========================\n\n')


def testHeding():
    print('\n\n=========================',
          '\n\tTESTING\n'
          '=========================\n\n')


def trainMessage(model):
    horizontalLine()
    print('\nTraining {} model:\n'.format(model))


def trainIteration(modelname, epoch, batch_idx, data, trainLoader, loss):
    print('{}:\tEpoch: {} [{}/{} ({:.0f}%)]\tLinear Loss: {:.6f}'
          .format(modelname, epoch+1, batch_idx * len(data),
                  len(trainLoader.dataset),
                  100. * batch_idx / len(trainLoader), loss))


def testMessage(model):
    horizontalLine()
    print('\nTesting {} model...\n'.format(model))


def testResults(rate, loss):
    print('Test Results:\nRate:\t{}\tAverage Loss:\t{}\n'.format(rate, loss))


def horizontalLine():
    print('-------------------------------------------'
          '-------------------------------------------')
