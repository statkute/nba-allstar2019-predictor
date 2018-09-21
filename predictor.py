from numpy import exp, array, random, dot
from collections import defaultdict
import csv
import numpy as np

class NeuralNetwork():
    def __init__(self):

        random.seed(1)

        self.synaptic_weights = 2 * random.random((26, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1.0 - np.tanh(x)**2

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            # adjustment = training_set_inputs.T.dot(error * self.__sigmoid_derivative(output))
            adjustment = training_set_inputs.T.dot(error * self.tanh_deriv(output))
            adjustment *= 0.0001

            self.synaptic_weights += adjustment


    def think(self, inputs):
        # return self.__sigmoid(dot(inputs, self.synaptic_weights))
        return self.tanh(dot(inputs, self.synaptic_weights))

def readInputCsv(filename, delim):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=delim)
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
            datalist = row[2:-1]
            for r in range (len(datalist)):
                if datalist[r] == '':
                    datalist[r] = '0'

            datalist = [ float(x) for x in datalist ]

            rowcopy = row
            output = rowcopy.pop()

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


def readOutputCsv(filename, delim):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=delim)
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
        diff = ones - zeros
        while (diff != 0):
            for x in range (zeros):
                merged_list.append(0.0)
            diff -= 1

    merged_list = [merged_list]

    return merged_list

def readTestCsv(filename, delim):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=delim)

    rownum = 0
    a = []
    firstRow = True

    for row in reader:
        if firstRow:
            firstRow = False
            continue
        if not firstRow:
            datalist = row[2:]
            for r in range (len(datalist)):
                if datalist[r] == '':
                    datalist[r] = '0'

            datalist = [ float(x) for x in datalist ]

            rowcopy = row
            output = rowcopy.pop()

            a.append (datalist)
            rownum += 1

    ifile.close()

    ax = np.array(a)
    a_normed = (ax - ax.min(0)) / ax.ptp(0)

    return a_normed

def read_names (filename, delim):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=delim)

    player_names = list()

    rownum = 0;
    firstRow = True

    for row in reader:
        if firstRow:
            firstRow = False
            continue
        if not firstRow:
            name = row[0]

            player_names.append(name)



    ifile.close()
    return player_names



def predict_season (filename, delim):
    results = {}
    stats = readTestCsv (filename, delim)
    player_names = read_names (filename, delim)
    list_number = 0

    for i in range (len (stats)):
        # print (player_names[list_number])
        results[player_names[list_number]] = neural_network.think(array(stats[i]))
        # print (results[player_names[list_number]])
        list_number += 1

    sorted_predictions = sorted(results.items(), key=lambda kv: kv[1], reverse=True)

    predicted_selections = sorted_predictions[:35]
    # print (predicted_selections[0][0])

    for player in predicted_selections:
        print(player[0][0:player[0].find('/')])

    # (neural_network.think(array(test)))



if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)


    # training_set_inputs12 = readInputCsv("training_data/2004-2005.csv", ";")
    # training_set_outputs12 = array(readOutputCsv("training_data/2004-2005.csv", ";")).T
    # neural_network.train(training_set_inputs12, training_set_outputs12, 10000)
    #
    # training_set_inputs11 = readInputCsv("training_data/2005-2006.csv", ";")
    # training_set_outputs11 = array(readOutputCsv("training_data/2005-2006.csv", ";")).T
    # neural_network.train(training_set_inputs11, training_set_outputs11, 10000)

    training_set_inputs10 = readInputCsv("training_data/2006-2007.csv", ";")
    training_set_outputs10 = array(readOutputCsv("training_data/2006-2007.csv", ";")).T
    neural_network.train(training_set_inputs10, training_set_outputs10, 10000)

    training_set_inputs9 = readInputCsv("training_data/2007-2008.csv", ",")
    training_set_outputs9 = array(readOutputCsv("training_data/2007-2008.csv", ",")).T
    neural_network.train(training_set_inputs9, training_set_outputs9, 10000)

    training_set_inputs8 = readInputCsv("training_data/2008-2009.csv", ",")
    training_set_outputs8 = array(readOutputCsv("training_data/2008-2009.csv", ",")).T
    neural_network.train(training_set_inputs8, training_set_outputs8, 10000)

    training_set_inputs7 = readInputCsv("training_data/2009-2010.csv", ",")
    training_set_outputs7 = array(readOutputCsv("training_data/2009-2010.csv", ",")).T
    neural_network.train(training_set_inputs7, training_set_outputs7, 10000)

    training_set_inputs6 = readInputCsv("training_data/2010-2011.csv", ",")
    training_set_outputs6 = array(readOutputCsv("training_data/2010-2011.csv", ",")).T
    neural_network.train(training_set_inputs6, training_set_outputs6, 10000)

    training_set_inputs5 = readInputCsv("training_data/2011-2012.csv", ",")
    training_set_outputs5 = array(readOutputCsv("training_data/2011-2012.csv", ",")).T
    neural_network.train(training_set_inputs5, training_set_outputs5, 10000)

    training_set_inputs4 = readInputCsv("training_data/2012-2013.csv", ",")
    training_set_outputs4 = array(readOutputCsv("training_data/2012-2013.csv", ",")).T
    neural_network.train(training_set_inputs4, training_set_outputs4, 10000)

    training_set_inputs3 = readInputCsv("training_data/2013-2014.csv", ",")
    training_set_outputs3 = array(readOutputCsv("training_data/2013-2014.csv", ",")).T
    neural_network.train(training_set_inputs3, training_set_outputs3, 10000)

    training_set_inputs2 = readInputCsv("training_data/2014-2015.csv", ",")
    training_set_outputs2 = array(readOutputCsv("training_data/2014-2015.csv", ",")).T
    neural_network.train(training_set_inputs2, training_set_outputs2, 10000)

    training_set_inputs1 = readInputCsv("training_data/2015-2016.csv", ";")
    training_set_outputs1 = array(readOutputCsv("training_data/2015-2016.csv", ";")).T
    neural_network.train(training_set_inputs1, training_set_outputs1, 10000)

    training_set_inputs = readInputCsv("training_data/2016-2017.csv", ";")
    training_set_outputs = array(readOutputCsv("training_data/2016-2017.csv", ";")).T
    neural_network.train(training_set_inputs, training_set_outputs, 10000)


    indexx = 0
    training_test = list()
    for x in range (len(training_set_outputs3)):
        if training_set_outputs3[x][0] == 0.0:
            for y in training_set_inputs3[x]:
                training_test.append(y)
            indexx = x
            break

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    print ("Considering new situation [...] -> ?: ")

    # test = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    print (training_test)
    print (training_set_outputs3[indexx])
    test = training_test

    print (neural_network.think(array(test)))

    predict_season("2017-2018.csv", ",")
