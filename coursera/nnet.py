#!/usr/bin/env python

import numpy as np

def readdat():
    with open("hw4_nnet_train.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        trainX = np.hsplit(data, [len(data[0])-1, len(data[0])])[0]
        trainy = np.hsplit(data, [len(data[0])-1, len(data[0])])[1].reshape(-1)

    with open("hw4_nnet_test.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        testX = np.hsplit(data, [len(data[0])-1, len(data[0])])[0]
        testy = np.hsplit(data, [len(data[0])-1, len(data[0])])[1].reshape(-1)

    return trainX, testX, [int(i) for i in trainy], [int(i) for i in testy.tolist()]

class mlp:
    def __init__(self, m, d, r):
        self.m = m
        self.d = d
        self.W = np.array([2*r*np.random.rand(m+1, d)-r,
                            2*r*np.random.rand(d+1, 1)-r])

    def backprop(self, X, y, eta):
        X = np.hstack(np.ones((np.shape(X)[0], 1)), X)

        s = [np.dot(X, self.W[0])]
        s.append(
            np.dot(np.hstack(np.ones((np.shape(X)[0], 1)), np.tanh(s[-1])),
                    self.W[1]
                )
            )

        delta = [-2*(y-xl[-1]) * ]

    def predict(self, x):
        np.tanh(np.dot(np.tanh(np.dot(x, self.W[0])), self.W[1]))



def prob11():
    T = 50000

def main():
    prob11()

if __name__ == '__main__':
    main()
