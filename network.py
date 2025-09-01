import numpy as np
from numpy import asarray
import mnist
import random
from segmenter import Segmenter
import json
import cv2

# this class represents a neural network. its parameters are the weights and biases connecting the "neurons" of each network (the neurons dont actually exist)
# has the ability to train (edit weights/biases) and test on a data set
class Network:

    # size is an int array, int at nth index is the number of neurons in the nth layer (count input layer as 0th layer
    #assume size has length 3
    def __init__(self, sizes):
        self.sizes = sizes

        # nth index is the biases in the (n+1)th layer, where the input layer is the 0th layer
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

        # randomly assign the weights of layers according to normal distribution
        # weights[l][j][k] is the weight from the kth neuron in the (l-1)th layer to the jth neuron in the lth layer
        #nth index is the weights from nth layer to (n+1)th layer, where input layer is the 0th layer
        self.weights = [np.random.normal(0, 1/np.sqrt(sizes[x-1]),(sizes[x], sizes[x-1])) for x in range(1, len(sizes))]
    
    # takes in the 784x1 input of the image and feeds the input forward through the network, returning an array with the pre-activated and activated values 
    def feedforward(self, input):
        # the nth index is the activation/z value of the nth layer. input is the 0th index for activation, z[0] = 0
        activation = [np.zeros(self.sizes[x]) for x in range(len(self.sizes))]
        z_values = [np.zeros(self.sizes[x]) for x in range(len(self.sizes))]

        # activation of next layer is given by W*A + b
        activation[0] = input
        for x in range(1, len(self.sizes)):
            z_values[x] = np.reshape(np.dot(self.weights[x-1], activation[x-1]), (self.sizes[x], 1)) + self.biases[x-1] 
            activation[x] = sigmoid(z_values[x])

        return z_values, activation
    
    # uses z_values, activation, desired input to calculate the gradient of the cost function with respect to all the weights and biases in the network
    def backprop(self, z_values, activation, desired):
        # array of delta values; nth index is the error in the (n+1)th layer (0th layer is input layer). 
        deltas = [np.zeros(self.sizes[x]) for x in range(1, len(self.sizes))]
        delta_outer = np.multiply(costDerivative(activation[len(activation)-1], np.reshape(desired, (self.sizes[len(self.sizes)-1],1))), sigmoidPrime(z_values[len(z_values)-1]))
        deltas[len(deltas) - 1] = delta_outer 
        # backpropagate through network, calculating delta
        for x in range(len(deltas) - 2, -1, -1):
            deltas[x] = np.dot(np.transpose(self.weights[x+1]), deltas[x+1])*sigmoidPrime(z_values[x+1])

        # assign bias_deriv/weight_deriv based on delta
        biases_deriv = deltas
        weights_deriv = [np.outer(deltas[x], activation[x]) for x in range(len(deltas))]

        return biases_deriv, weights_deriv
        
    # eta is learning rate, mini_batch is a list of tuples where first corrresponds to deisred output, seconcd outut is pixel values
    # trains the network by modifying weights and biases to reduce cost function
    def train(self, training_input, training_output, mini_batch_size, epochs, eta, lbda):
        history = np.zeros(20)
        # create a validation set to prevent overtraining
        validation_input = training_input[int(len(training_input) * 0.9):]
        validation_output = [np.argmax(x) for x in training_output[int(len(training_output)*0.9):]]
        training_input = training_input[:int(len(training_input) *0.9)]
        training_output = training_output[:int(len(training_output)*0.9)]

        for x in range(epochs):
            count = 0
            # shuffle code from https://stackoverflow.com/questions/13343347/randomizing-two-lists-and-maintaining-order-in-python
            combined = list(zip(training_input, training_output))
            random.shuffle(combined)

            # train on mini batches
            mini_batches = [combined[y:y+mini_batch_size] for y in range(int(len(combined)/mini_batch_size))]

            for minibatch in mini_batches:
                count = count + self.update(minibatch, eta, lbda, len(training_input))

            print("Epoch " + str(x) + " done " + str(count) + "/" + str(training_input.shape[0]))

            # record past 20 epochs of accuracies
            if (x < 20):
                history[x] = self.test(validation_input, validation_output)
            else:
                history = shift(history)
                history[-1] = self.test(validation_input, validation_output)

            # stop training if there is no improvement in validation accuracy for the past 20 epochs
            if (isMaxValueLast(history) == False and x > 20):
                break


    
    # takes in a mini_batch, eta (learning rate), lambda (regularization rate), length of mini_batch
    # calculates the gradient of each training example and updates the weights/biases with an average of summed gradients. also returns the count of how many were correct
    def update(self, mini_batch, eta, lbda, length):
        count = 0
        dbiases = [np.zeros((x,1)) for x in self.sizes[1:]]
        dweights = [np.zeros((self.sizes[x], self.sizes[x-1])) for x in range(1, len(self.sizes))]

        # sum the gradients of cost function with respect to weights and biases
        for input, desired in mini_batch:
            z_values, activation = self.feedforward(input)
            biases_deriv, weights_deriv = self.backprop(z_values, activation, desired)
            dbiases = [db + bd for db, bd in zip(dbiases, biases_deriv)]
            dweights = [dw + wd for dw, wd in zip(dweights, weights_deriv)]

            # count if prediction is correct
            if (np.argmax(desired) == np.argmax(activation[len(activation)-1])):
                count += 1

        # update based on update rule
        self.biases = [x - (eta/len(mini_batch))*y  for x, y in zip(self.biases, dbiases)]
        self.weights = [x - (eta/len(mini_batch))*y - eta * lbda * x/length for x,y in zip(self.weights, dweights)]

        return count
    
    # tests the code with purely the forward propagation algorithm given test_input and test_output lists
    # returns the accuracy 
    def test(self, test_input, test_output):
        count = 0.0
        for x in range(0, len(test_input)):
            input = test_input[x]
            output = test_output[x]

            z_values, activation = self.feedforward(input)
            if (output == np.argmax(activation[len(activation)-1])):
                count = count + 1

        return count/len(test_input)
    
    # test singular image
    def testImage(self, testImage):
        z_values, activation = self.feedforward(testImage)
        return unicodeToOutput(np.argmax(activation[len(activation)-1]))
    
    # saves the network's parameters to a json file
    # modified from http://neuralnetworksanddeeplearning.com/chap3.html
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        with open(filename, "w") as json_file:
            json.dump(data, json_file)

# returs true if the maximum value of the array is last
def isMaxValueLast(arr):
    lastValue = arr[-1]
    for x in range(len(arr)-1):
        if (lastValue > arr[x]):
            return True
    return False

# shifts each value of the array left
def shift(arr):
    for x in range(len(arr)-1):
        arr[x] = arr[x+1]
    return arr

# activation function; behaves like a smooth step function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x)) 

# derivative of cross entropy cost function
def costDerivative(output, desired):
    return -1.0 * (np.divide(desired,output) - np.divide(1 - desired, 1 - output))

# derivative of activation function
def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# returns a network with the parameters from saved network
# modified from http://neuralnetworksanddeeplearning.com/chap3.html
def load(filename):
     f = open(filename, "r")
     data = json.load(f)
     f.close()
     net = Network(data["sizes"])
     net.weights = [np.array(w) for w in data["weights"]]
     net.biases = [np.array(b) for b in data["biases"]]
     return net

# takes in an integer representing the Unicode code of a character, returns the character
def unicodeToOutput(u):
    data = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116])
    return chr(data[u])

# take in an image path and network and create a guess
def predict(network, image_path):
    segment = Segmenter(image_path)
    segment.segment() 

    sentence = ""
    # preprocess each segmented character and predict its output
    for character in segment.characters:
        if (len(character) != 0):
            # preprocess
            img = np.pad(character, pad_width=((0, 0), (15, 15)), mode='constant', constant_values=0)
            img = cv2.resize(character, (28,28), interpolation = cv2.INTER_AREA)
            thresh, im_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
            img = im_bw.flatten()/255.0
            # predict
            sentence = sentence + str(network.testImage(img))
        else:
            sentence = sentence + " "
    return sentence



