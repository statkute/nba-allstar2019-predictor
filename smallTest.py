from numpy import exp, array, random, dot
from collections import defaultdict
import csv
import numpy as np

class NeuralNetwork():
    def __init__(self):

        random.seed(1)

        self.synaptic_weights = 2 * random.random((5, 1)) - 1

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

def readInputCsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0
    a = []
    firstRow = True

    for row in reader:
        if firstRow:
            firstRow = False
            continue
        datalist = row[2:-1]
        for r in range (len(datalist)):
            if datalist[r] == '':
                datalist[r] = '0'

        datalist = [ float(x) for x in datalist ]
        a.append (datalist)
        rownum += 1

    ifile.close()

    ax = np.array(a)
    a_normed = ax / ax.max(axis=0)
    return a_normed





if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    training_set_inputs = array(readInputCsv("training_data/2016-2017.csv"))
    training_set_outputs = array(readOutputCsv("training_data/2016-2017.csv")).T

    print (training_set_inputs)
    print (training_set_outputs)

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    print ("Considering new situation [...] -> ?: ")
    test = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    print (neural_network.think(array(test)))
