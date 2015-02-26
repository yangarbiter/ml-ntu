#!/usr/bin/env python

import numpy as np

def readdat():
    with open("hw4_knn_train.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        trainX = np.hsplit(data, [len(data[0])-1, len(data[0])])[0]
        trainy = np.hsplit(data, [len(data[0])-1, len(data[0])])[1].reshape(-1)

    with open("hw4_knn_test.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        testX = np.hsplit(data, [len(data[0])-1, len(data[0])])[0]
        testy = np.hsplit(data, [len(data[0])-1, len(data[0])])[1].reshape(-1)

    return trainX, testX, [int(i) for i in trainy], [int(i) for i in testy.tolist()]

distance = lambda x1, x2: np.linalg.norm(x1-x2)

def prob1516():
    trainX, testX, trainy, testy = readdat()
    pred = []
    for x in trainX:
        predi = sorted([(i, np.linalg.norm(trainX[i] - x)) for i in range(len(trainX))],
                key=lambda ent: ent[1])[0][0]
        pred.append(trainy[predi])

    print "[prob15] Ein = %lf" % (np.sum(np.array(pred) != np.array(trainy)) /
            float(len(trainy)))

    pred = []
    for x in testX:
        predi = sorted([(i, np.linalg.norm(trainX[i] - x)) for i in range(len(trainX))],
                key=lambda ent: ent[1])[0][0]
        pred.append(trainy[predi])

    print "[prob16] Eout = %lf" % (np.sum(np.array(pred) != np.array(testy)) /
            float(len(testy)))

def prob1718():
    trainX, testX, trainy, testy = readdat()
    pred = []
    for x in trainX:
        sp = sorted([(i, np.linalg.norm(trainX[i] - x)) for i in range(len(trainX))],
                key=lambda ent: ent[1])[0:5]
        predi = 0
        for p in sp:
            predi += trainy[p[0]]
        pred.append(np.sign(predi))

    print "[prob17] Ein = %lf" % (np.sum(np.array(pred) != np.array(trainy)) /
            float(len(trainy)))

    pred = []
    for x in testX:
        sp = sorted([(i, np.linalg.norm(trainX[i] - x)) for i in range(len(trainX))],
                key=lambda ent: ent[1])[0:5]
        predi = 0
        for p in sp:
            predi += trainy[p[0]]
        pred.append(np.sign(predi))

    print "[prob18] Eout = %lf" % (np.sum(np.array(pred) != np.array(testy)) /
            float(len(testy)))

def main():
    prob1516()
    prob1718()

if __name__ == '__main__':
    main()
