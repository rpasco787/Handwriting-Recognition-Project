import numpy as np
import pandas as pd

# This file loads in (E)MNIST data and processes it into 3D arrays that can be used by the neural network

# np arrays where the first column of each row is the digit the image represents, remaining columns are pixel values (0-255)
def getData():
    mnist_training_data = np.array(pd.read_csv('/Users/ryanpascual/Documents/mnist_train.csv'))
    mnist_test_data = np.array(pd.read_csv('/Users/ryanpascual/Documents/mnist_test.csv'))

    return mnist_training_data, mnist_test_data

# return a list of 60,000 ordered pairs where the first entry is the 784x1 vector representing pixel values of an image, second entry is 10x1 vector 
# with the desired output
# returns another list of 10,000 ordered pairs where first entry is 784x1 vector for pixel vlaues, second entry is scalar desired output
def loadData():
    training, test = getData()

    # reshape the arrays to be in the correct format
    training_input = np.array([row[1:].reshape(784,1) for row in training])
    training_input = np.reshape(training_input, (60000,784))
    training_input = training_input/255.0

    training_output = np.array([toVector(row[0], 10) for row in training])
    training_output = np.reshape(training_output, (60000,10))

    test_input = np.array([row[1:].reshape(784,1) for row in test])
    test_input = np.reshape(test_input, (10000, 784))
    test_input = test_input/255.0

    test_output = np.array([row[0] for row in test])
    test_output = np.reshape(test_output, (10000,1))

    return training_input, training_output, test_input, test_output

# takes in an integer x and integer size and returns the xth standard basis with size 'size'
def toVector(x, size):
    arr = np.zeros((size,1))
    arr[x] = 1.0
    return arr

# returns a Numpy array with all the pixel values of the image (785x1)
# one index represents one image, the 0th index being the actual value
def getEmnistData():
    mnist_training_data = np.array(pd.read_csv('/Users/ryanpascual/Documents/emnisttrain.csv'))
    mnist_test_data = np.array(pd.read_csv('/Users/ryanpascual/Documents/emnisttest.csv'))

    return mnist_training_data, mnist_test_data

# takes in integer parameters for the training size and test size
# returns the training/test inputs, outputs of the EMNIST data set correctly formatted
def loadEmnistData(train_size, test_size):
    training, test = getEmnistData()

    # max size 697931
    training_input = np.array([(row[1:].reshape(28,28).transpose()).reshape(784) for row in training[:train_size]])
    training_input = training_input/255.0

    training_output = np.array([toVector(row[0], 47) for row in training[:train_size]])
    training_output = np.reshape(training_output, (train_size, 47))

    # max size 116322
    test_input = np.array([row[1:].reshape(28,28).transpose().reshape(784,1) for row in test[:test_size]])
    test_input = np.reshape(test_input, (test_size, 784))
    test_input = test_input/255.0

    test_output = np.array([row[0] for row in test[:test_size]])
    test_output = np.reshape(test_output, (test_size,1))

    return training_input, training_output, test_input, test_output




