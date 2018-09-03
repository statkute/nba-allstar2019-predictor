import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import csv

def readInputCsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0
    a = []
    firstRow = True

    for row in reader:
        if firstRow:
            firstRow = False
            continue
        if not firstRow:
            datalist = row[1:-1]
            for r in range (len(datalist)):
                if datalist[r] == '':
                    datalist[r] = '0'

            datalist = [ float(x) for x in datalist ]
            a.append (datalist)
            rownum += 1

    ifile.close()
    # ax = np.array(a)
    # a_normed = (ax - ax.min(0)) / ax.ptp(0)


    return a


def readOutputCsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0
    a = []
    firstRow = True

    for row in reader:
        if firstRow:
            firstRow = False
            continue
        if not firstRow:
            datalist = row.pop()
            datalist = [ float(x) for x in datalist ]
            a.append (datalist)
            rownum += 1

    ifile.close()
    merged_list = []
    for l in a:
        merged_list += l
    # merged_list = [merged_list]

    return merged_list


if __name__ == "__main__":
    training_set_inputs = readInputCsv("training_data/mini-sample.csv")
    training_set_outputs = readOutputCsv("training_data/mini-sample.csv")
    print (training_set_inputs)
    print (training_set_outputs)
    X = training_set_inputs
    y = training_set_outputs

    p = Perceptron(random_state=42,max_iter=10)
    p.fit(X, y)

    print(p.predict([[0.9,0.9,0.9]]))
