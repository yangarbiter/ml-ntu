#!/usr/bin/env python

import numpy as np

def readdat():
    with open("hw4_train.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split(' ')[0:]] for line in f.readlines()])
        trainX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        trainX = np.hstack((np.ones((np.shape(trainX)[0], 1)), trainX))
        trainy = np.hsplit(data, [len(data[0])-1,
            len(data)])[1].reshape(np.shape(trainX)[0], 1)

    with open("hw4_test.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split(' ')[0:]] for line in f.readlines()])
        testX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        #add constant to feature
        testX = np.hstack((np.ones((np.shape(testX)[0], 1)), testX))
        testy = np.hsplit(data, [len(data[0])-1,
            len(data)])[1].reshape(np.shape(testX)[0], 1)

    return trainX, testX, trainy, testy

LAMBDA = 10

norm2 = lambda x: np.sum(x**2)
err = lambda w, X, y: (LAMBDA*norm2(w) + norm2(np.dot(X, w) - y)) \
        / np.shape(X)[0]
grad = lambda w, X, y: (LAMBDA*2*w + \
        2 * np.dot(X.T, np.dot(X, w) - y)) / np.shape(X)[0]
wreg = lambda X, y: np.dot( np.dot( np.linalg.pinv(
                np.dot(X.T, X) + LAMBDA*np.eye(np.shape(X)[1])),
                X.T),
                y)
predict = lambda w, X: np.sign(np.dot(X, w))
err01 = lambda y, y_ans: np.sum(y != y_ans)

def prob13():
    global LAMBDA
    LAMBDA = 10
    trainX, testX, trainy, testy = readdat()
    w = wreg(trainX, trainy)
    print "[prob13]Ein:",
    print err01(predict(w, trainX), trainy) / float(np.shape(trainX)[0])
    print "[prob13]Eout:",
    print err01(predict(w, testX), testy) / float(np.shape(testX)[0])
    """ gradient decent
    trainX, testX, trainy, testy = readdat()
    w = np.random.random([np.shape(trainX)[1],1])
    for i in range(100):
        print err(w, trainX, trainy)
        w -= 0.5*grad(w, trainX, trainy)
    print err(wreg(trainX, trainy), trainX, trainy)
    """

def prob1415():
    global LAMBDA
    trainX, testX, trainy, testy = readdat()
    for lamb in np.power(10.0, np.arange(2, -11, -1)):
        LAMBDA = lamb
        w = wreg(trainX, trainy)
        print "[prob14&15]log Lambda:",
        print np.log10(LAMBDA),
        print "Ein:",
        print err01(predict(w, trainX), trainy) / float(np.shape(trainX)[0]),
        print "Eout:",
        print err01(predict(w, testX), testy) / float(np.shape(testX)[0])

def prob161718():
    global LAMBDA
    trainX, testX, trainy, testy = readdat()
    valX = trainX[120:200, :]
    traX = trainX[0:120, :]
    valy = trainy[120:200, :]
    tray = trainy[0:120, :]
    bestTrain = [100.0, 0, 0]
    bestVal = [100.0, 0, 0]
    for lamb in np.power(10.0, np.arange(2, -11, -1)):
        LAMBDA = lamb
        w = wreg(traX, tray)
        Etrain = err01(predict(w, traX), tray) / float(np.shape(traX)[0])
        if  Etrain < bestTrain[0]:
            bestTrain[0] = Etrain
            bestTrain[1] = w
            bestTrain[2] = lamb

        Eval = err01(predict(w, valX), valy) / float(np.shape(valX)[0])
        if  Eval < bestVal[0]:
            bestVal[0] = Eval
            bestVal[1] = w
            bestVal[2] = lamb

    print "[prob16]log Lambda:",
    print np.log10(bestTrain[2]),
    print "Etrain:",
    print err01(predict(bestTrain[1], traX), tray) / float(np.shape(traX)[0]),
    print "Eval:",
    print err01(predict(bestTrain[1], valX), valy) / float(np.shape(valX)[0]),
    print "Eout:",
    print err01(predict(bestTrain[1], testX), testy) / float(np.shape(testX)[0])

    print "[prob17]log Lambda:",
    print np.log10(bestVal[2]),
    print "Etrain:",
    print err01(predict(bestVal[1], traX), tray) / float(np.shape(traX)[0]),
    print "Eval:",
    print err01(predict(bestVal[1], valX), valy) / float(np.shape(valX)[0]),
    print "Eout:",
    print err01(predict(bestVal[1], testX), testy) / float(np.shape(testX)[0])

    LAMBDA = bestVal[2]
    w = wreg(trainX, trainy)
    Ein = err01(predict(w, trainX), trainy) / float(np.shape(trainX)[0])

    print "[prob18]log Lambda:",
    print np.log10(bestVal[2]),
    print "Ein:",
    print err01(predict(w, trainX), trainy) / float(np.shape(trainX)[0]),
    print "Eout:",
    print err01(predict(w, testX), testy) / float(np.shape(testX)[0])

def prob1920():
    #5-folds cross validation
    global LAMBDA
    trainX, testX, trainy, testy = readdat()
    folds = 5 #5-folds
    fold_size = len(trainX)/folds
    bestLamb = [100.0, 0]
    for lamb in np.power(10.0, np.arange(2, -11, -1)):
        LAMBDA = lamb
        avg_err = 0.0
        for i in range(folds):
            valX = trainX[fold_size*i: fold_size*(i+1)]
            valy = trainy[fold_size*i: fold_size*(i+1)]
            X = np.concatenate((trainX[0: fold_size*i, :],
                                trainX[fold_size*(i+1):, :]))
            y = np.concatenate((trainy[0: fold_size*i, :],
                                trainy[fold_size*(i+1):, :]))
            w = wreg(X, y)
            avg_err += err01(predict(w, valX), valy) / float(np.shape(valX)[0])
        avg_err /= folds
        if avg_err < bestLamb[0]:
            bestLamb[0] = avg_err
            bestLamb[1] = lamb
    w = wreg(trainX, trainy)

    print "[prob19]log Lambda:",
    print np.log10(bestLamb[1]),
    print "Ecv:",
    print bestLamb[0]

    print "[prob20]Ein:",
    print err01(predict(w, trainX), trainy) / float(np.shape(trainX)[0]),
    print "Eout:",
    print err01(predict(w, testX), testy) / float(np.shape(testX)[0])

def main():
    prob13()
    print "====================================================="
    prob1415()
    print "====================================================="
    prob161718()
    print "====================================================="
    prob1920()

if __name__ == '__main__':
    main()
