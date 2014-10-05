#!/usr/bin/python
import numpy as np
import random

class Pocket():
    def __init__(self, init_w, eta=1.0):
        self.w = init_w
        self.best_w = init_w
        self.eta = eta
        
    def _sign(self, x):
        if x > 0:
            return 1
        else: return -1

    def _list_sign(self, x):
        ret = []
        for i in x:
            if i > 0.0: ret.append(1)
            else: ret.append(-1)
        return ret

    def _error(self, yp, y):
        return sum(np.array(yp) != np.array(y))

    def train(self, X, y, iterations):
        for i in xrange(iterations):
            update = False

            t1 = zip(X, y)
            random.shuffle(t1)
            X, y = zip(*t1)
            X = list(X)
            y = list(y)

            for j in xrange(len(X)):
                x = X[j]
                _y = y[j]
                if self._sign(np.dot(self.w, np.array(x))) != _y:
                    self.w = self.w + self.eta * np.array(x) * _y
                    update = True
                    if self._error(self.predict(X, self.w), y) < self._error(self.predict(X, self.best_w), y):
                        self.best_w = self.w
                    #print x, self.w
                    break
            if update == False:
                break
    
    def predict(self, X, w):
        return self._list_sign(np.array(X).dot(np.transpose(w)))


def exp1819(X, y, X_test, y_test):
    ans18 = 0
    ans19 = 0
    for i in range(2000):
        pocket = Pocket(np.array([0] * np.shape(X)[1]))
        pocket.train(X, y, 50)
        ans18 += pocket._error(pocket.predict(X_test, pocket.best_w), y_test) / float(len(X_test))
        ans19 += pocket._error(pocket.predict(X_test, pocket.w), y_test) / float(len(X_test))
    print pocket.w, pocket.best_w
    print("[18]testing error %f" % (ans18 / 2000.0))
    print("[19]testing error %f" % (ans19 / 2000.0))

def exp20(X, y, X_test, y_test):
    ans = 0
    for i in range(2000):
        pocket = Pocket(np.array([0] * np.shape(X)[1]))
        pocket.train(X, y, 100)
        ans += pocket._error(pocket.predict(X_test, pocket.best_w), y_test) / float(len(X_test))
    print("[20]testing error %f" % (ans / 2000.0))

def main():
    X = []
    y = []
    X_test = []
    y_test = []
    with open("hw1_18_train.dat", "r") as f:
        line = f.readline()
        while line != '':
            x = [1.0]
            y.append(int(line.split('\t')[-1].strip('\n')))
            line = line.split('\t')[0]
            for l in line.split(' '):
                x.append(float(l.strip('\t')))
            X.append(x)
            line = f.readline()

    with open("hw1_18_test.dat", "r") as f:
        line = f.readline()
        while line != '':
            x = [1.0]
            y_test.append(int(line.split('\t')[-1].strip('\n')))
            line = line.split('\t')[0]
            for l in line.split(' '):
                x.append(float(l.strip('\t')))
            X_test.append(x)
            line = f.readline()

    print("===============Running experiments==============")
    exp1819(X, y, X_test, y_test)
    exp20(X, y, X_test, y_test)


if __name__ == "__main__":
    main()
