
import numpy as np
import itertools
from multiprocessing import Pool
def readdat():
    with open("hw2_lssvm_all.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        X = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        y = np.hsplit(data, [len(data[0])-1, len(data)])[1].reshape(np.shape(X)[0])

    return X, y

gamma = [32.0, 2.0, 0.125]
lambd = [0.001, 1, 1000]
rbf = lambda gamma, x, xp: np.exp(-gamma * np.linalg.norm(x-xp, 2)**2)
X, y = readdat()
testX = np.array(X[400:])
testy = np.array(y[400:])
X = np.array(X[:400])
y = np.array(y[:400])
N = 400
print np.shape(np.array(y))
print np.shape(np.array(X))

def train(params):
    print params,
    g, l = params
    K = [[ rbf(g, X[i], X[j]) for i in range(N)] for j in range(N)]
    beta = np.dot(np.linalg.inv(g * np.eye(N) - K), y)
    predin = []
    predout = []
    for x in X:
        r = 0
        for i in range(N):
            r += beta[i] * rbf(g, X[i], x)
        predin.append(np.sign(r))
    for x in testX:
        r = 0
        for i in range(N):
            r += beta[i] * rbf(g, X[i], x)
        predin.append(np.sign(r))
    print np.sum(np.array(predin) != y)/float(len(y)),
    print np.sum(np.array(predout) != testy)/float(len(testy))

def main():
    p = Pool(10)
    p.map(train, [param for param in itertools.product(gamma, lambd)])

if __name__ == '__main__':
    main()
