
import numpy as np

def readdat():
    with open("hw6_train.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        trainX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        trainy = np.hsplit(data, [len(data[0])-1, len(data)])[1].reshape(np.shape(trainX)[0])

    with open("hw6_test.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        testX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        testy = np.hsplit(data, [len(data[0])-1, len(data)])[1].reshape(np.shape(testX)[0])

    return trainX, testX, trainy, testy

class decision_stump(object):
    def __init__(self):
        self.err = lambda ypre, y, u: float(np.sum(np.dot(u.T,
            (ypre!=y))))/np.sum(u)

    def train(self, X, y, u):
        predict = lambda Xj, b: np.sign(Xj - b)
        param = [0, 9999999.0] #param, Ein
        for i in range(np.shape(X)[1]):
            xi = sorted(X[:, i].tolist() + [max(X[:,i])+1])
            for j in range(len(xi)-1):
                b = (xi[j] + xi[j+1]) / 2.0
                errin = self.err(predict(X[:, i], b), y, u)
                if errin < param[1]:
                    param = [(i, b, 1), errin]
                elif (1-errin) < param[1]:
                    param = [(i, b, -1), (1-errin)]
        self.param = param[0]
        return param[1]

    def predict(self, X):
        return np.sign(X[:, self.param[0]] - self.param[1]) * self.param[2]

class adaboost(object):
    def __init__(self):
        self.g = []
        self.alpha = []
        self.e = []
        self.U = []

    def train(self, X, y, T):
        self.u = np.array([1.0/len(X) for i in range(len(X))])
        for i in xrange(T):
            ds = decision_stump()
            err = ds.train(X, y, self.u)
            scale = np.sqrt((1-err)/err)

            #print err, scale, sum(self.u), ds.param
            self.u[ds.predict(X) == y] /= scale
            self.u[ds.predict(X) != y] *= scale
            #print ds.err(ds.predict(X), y, self.u) #should be 0.5
            self.e.append(err)
            self.U.append(np.sum(self.u))

            self.g.append(ds)
            self.alpha.append(np.log(scale))

    def predict(self, X):
        gts = np.array([gt.predict(X) for gt in self.g])
        return np.sign(np.dot(np.array([self.alpha]), gts)).reshape(-1)

err = lambda ypre, y: float(np.sum(ypre!=y)) / len(ypre)

def probs():
    trainX, testX, trainy, testy = readdat()
    ada = adaboost()
    ada.train(trainX, trainy, 300)
    print "prob12: g_1Ein=%f" % ada.e[0]
    print "prob13: GEin=%f" % err(ada.predict(trainX), trainy)
    print "prob14: U_2=%f" % ada.U[1]
    print "prob15: U_T=%f" % np.sum(ada.u)
    print "prob16: minE=%f" % min(ada.e)
    print "prob17: g_1Eout=%f" % err(ada.g[0].predict(testX), testy)
    print "prob18: GEout=%f" % err(ada.predict(testX), testy)

def main():
    probs()

if __name__ == '__main__':
    main()
