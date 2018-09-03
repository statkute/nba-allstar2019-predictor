from numpy import exp, array, random, dot
from collections import defaultdict
import csv
import numpy as np

class NeuralNetwork():
    def __init__(self):

        random.seed(1)

        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = training_set_inputs.T.dot(error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment


    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    training_set_inputs = array([
     [0.46,0.4,0.98039216],
     [0.58,0.0,0.98039216],
     [0.2,1.0,0.39215686],
     [0.1,0.4,0.45960784],
     [0.74,0.53333333,0.19607843],
     [0.48,0.93333333,0.0],
     [0.38,0.7,0.98039216],
     [0.02,0.53333333,1.0],
     [0.,0.03333333,0.88235294],
     [1.0,0.8,0.78431373]])

    training_set_outputs = array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    print ("Considering new situation [0.5,0.5,0.5] -> ?: ")
    test = [0.9,0.2,0.2]
    print (neural_network.think(array(test)))
