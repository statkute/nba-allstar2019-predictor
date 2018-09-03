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
    reader = csv.reader(ifile, delimiter=",")
    zeros = list()
    ones = list()

    rownum = 0
    a = []
    firstRow = True

    for row in reader:
        if firstRow:
            firstRow = False
            continue
        if not firstRow:
            output = row.pop()
            datalist = row[2:-1]
            for r in range (len(datalist)):
                if datalist[r] == '':
                    datalist[r] = '0'

            datalist = [ float(x) for x in datalist ]

            if (output == '0'):
                zeros.append(datalist);
            else:
                ones.append(datalist);

            a.append (datalist)
            rownum += 1

    ifile.close()

    if (len (ones) < len (zeros)):
        diff = len (zeros) // len (ones)
        while (diff != 0):
            for x in ones:
                a.append(x)
            diff -= 1
    else:
        diff = len (ones) - len (zeros)
        while (diff != 0):
            for x in zeros:
                a.append(x)
            diff -= 1

    ax = np.array(a)
    a_normed = (ax - ax.min(0)) / ax.ptp(0)

    return a_normed


def readOutputCsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")
    zeros = 0;
    ones = 0;

    rownum = 0
    a = []
    firstRow = True

    for row in reader:
        if firstRow:
            firstRow = False
            continue
        if not firstRow:
            datalist = row.pop()

            if (datalist == '0'):
                zeros += 1;
            else:
                ones += 1;

            datalist = [ float(x) for x in datalist ]
            a.append (datalist)
            rownum += 1

    ifile.close()
    merged_list = []
    for l in a:
        merged_list += l

    if (ones < zeros):
        diff = zeros // ones
        while (diff != 0):
            for x in range (ones):
                merged_list.append(1.0)
            diff -= 1
    else:
        diff = ones // zeros
        while (diff != 0):
            for x in range (zeros):
                merged_list.append(0.0)
            diff -= 1

    merged_list = [merged_list]

    return merged_list


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    training_set_inputs = readInputCsv("training_data/mini.csv")
    training_set_outputs = array(readOutputCsv("training_data/mini.csv")).T

    print (training_set_inputs)
    print (training_set_outputs)
    print (len(training_set_inputs))
    print (len(training_set_outputs))

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    print ("Considering new situation [...] -> ?: ")

    test = [0.5,0.5,0.5,0.5,0.5]
    print (neural_network.think(array(test)))
