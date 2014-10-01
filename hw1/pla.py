#!/usr/bin/python
import numpy as np
import random

class PLA():
    def __init__(self, init_w, eta=1.0):
        self.w = init_w
        self.eta = eta
        self.all_w = []
    
    def _sign(self, x):
        if x > 0:
            return 1
        else: return -1

    def train(self, X, y):
        idx = 0
        while True:
            update = False
            self.all_w.append(self.w)
            for i in xrange(len(X)):
                x = X.pop()
                X.insert(0, x)
                _y = y.pop()
                y.insert(0, _y)
                if self._sign(np.dot(self.w, np.array(x))) != _y:
                    self.w = self.w + self.eta * np.array(x) * _y
                    update = True
                    #print x, self.w
                    break
            if update == False:
                break
        self.all_w.append(self.w)

def exp15(X, y):
    pla = PLA(np.array([0] * np.shape(X)[1]))
    pla.train(X, y)
    print("total updates: %d" % len(pla.all_w))

def exp16(X, y):
    ans = 0
    for i in range(2000):
        t1 = zip(X, y)
        random.shuffle(t1)
        X, y = zip(*t1)
        X = list(X)
        y = list(y)

        pla = PLA(np.array([0] * np.shape(X)[1]))
        pla.train(X, y)
        ans += len(pla.all_w)
    print("avg total updates: %f" % (ans / 2000.0))

def exp17(X, y):
    ans = 0
    for i in range(2000):
        t1 = zip(X, y)
        random.shuffle(t1)
        X, y = zip(*t1)
        X = list(X)
        y = list(y)

        pla = PLA(np.array([0] * np.shape(X)[1]), eta=0.5)
        pla.train(X, y)
        ans += len(pla.all_w)
    print("avg total updates: %f" % (ans / 2000.0))


def main():
    X = []
    y = []
    with open("hw1_15_train.dat", "r") as f:
        line = f.readline()
        while line != '':
            x = [1.0]
            y.append(int(line.split('\t')[-1].strip('\n')))
            line = line.split('\t')[0]
            for l in line.split(' '):
                x.append(float(l.strip('\t')))
            X.append(x)
            line = f.readline()

    print("===============Running experiments==============")
    exp15(X, y)
    exp16(X, y)
    exp17(X, y)


if __name__ == "__main__":
    main()
