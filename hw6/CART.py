
import numpy as np
import multiprocessing as mp
import random, itertools

def readdat():
    with open("hw6_train.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        trainX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        trainy = np.hsplit(data, [len(data[0])-1, len(data)])[1].reshape(np.shape(trainX)[0])

    with open("hw6_test.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        testX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        testy = np.hsplit(data, [len(data[0])-1, len(data)])[1].reshape(np.shape(testX)[0])

    trainy.astype(np.int)
    testy.astype(np.int)
    return trainX, testX, trainy, testy

class Node(object):
    def __init__(self):
        self.clf = None
        self.ln = None
        self.rn = None

class decision_stump(object):
    def __init__(self):
        pass

    def gini(self, y, K):
        if len(y) == 0: return 0
        ret = 0.0
        for k in np.unique(y):
            ret += (np.sum((y == k)) / float(len(y)))**2
        ret = 1 - ret / K
        return ret

    def target_function(self, ypre, y):
        #print self.gini(y[ypre==-1], 2), self.gini(y[ypre!=-1], 2)
        return np.sum(ypre==-1) * self.gini(y[ypre == -1], 2) +\
                np.sum(ypre!=-1) * self.gini(y[ypre != -1], 2)

    def train(self, X, y):
        predict = lambda X, b: np.sign(X - b)
        param = [0, 9999999.0, 99999.0] #param, Ein, diff
        for i in range(np.shape(X)[1]):
            xi = sorted(X[:, i].tolist() + [max(X[:,i])+1])
            for j in range(len(xi)-1):
                b = (xi[j] + xi[j+1]) / 2.0
                ypre = predict(X[:, i], b)
                tar = self.target_function(ypre, y)
                #print param, abs(np.sum(ypre==-1)-np.sum(ypre!=-1))
                if tar < param[1]:
                    param = [(i, b, 1), tar,
                            abs(np.sum(ypre==-1)-np.sum(ypre!=-1))]
                elif tar == param[1] and\
                        abs(np.sum(ypre==-1)-np.sum(ypre!=-1)) > param[2]:
                    param = [(i, b, 1), tar,
                            abs(np.sum(ypre==-1)-np.sum(ypre!=-1))]

        self.param = param[0]
        return param[1]

    def predict(self, X):
        return np.sign(X[:, self.param[0]] - self.param[1]) * self.param[2]

branch_funcs = 0

def build_tree(node, X, y):
    global branch_funcs
    node.clf = decision_stump()
    node.clf.train(X, y)
    #print np.sum(node.clf.predict(X)==-1), np.sum(node.clf.predict(X)!=-1)
    if np.sum((node.clf.predict(X)==-1))==0 or np.sum((node.clf.predict(X)!=-1))==0:
        node.clf = y[0]
        node.ln = None
        node.rn = None
    else:
        branch_funcs += 1
        node.ln = Node()
        build_tree(node.ln,
                X[node.clf.predict(X)==-1], y[node.clf.predict(X)==-1])
        node.rn = Node()
        build_tree(node.rn,
                X[node.clf.predict(X)!=-1], y[node.clf.predict(X)!=-1])

def predict_tree(root, x):
    if root.ln == None and root.rn == None:
        return root.clf
    deci = root.clf.predict(x)
    if deci == -1:
        return predict_tree(root.ln, x)
    else:
        return predict_tree(root.rn, x)

def prob161718():
    global branch_funcs
    trainX, testX, trainy, testy = readdat()
    root = Node()
    build_tree(root, trainX, trainy)
    print "prob16: num of branch functions = %d" % branch_funcs
    pred = []
    for x in trainX:
        pred.append(predict_tree(root, np.array([x])))
    Ein = float(np.sum(np.array(pred)!=trainy)) / len(trainy)
    print "prob17: Ein = %f" % Ein
    pred = []
    for x in testX:
        pred.append(predict_tree(root, np.array([x])))
    Eout = float(np.sum(np.array(pred)!=testy)) / len(testy)
    print "prob18: Eout = %f" % Eout

def buildForest(N, X, y):
    ret = []
    for i in range(N):
        ret.append(Node())
        sample = [random.choice(range(len(y))) for i in range(len(y))]
        build_tree(ret[-1], X[sample], y[sample])
    return re

def predictForest(forest, X):
    pred = []
    for x in X:
        vote = [0, 0]
        for g in forest:
            vote[int(predict_tree(g, np.array([x]))+1)/2] += 1
        pred.append(vote.index(max(vote))*2 - 1)
    return pred

def expRF(trainX, trainy, testX, testy):
    forest = buildForest(300, trainX, trainy)
    print "finish building forest"
    pred = predictForest(forest, testX)
    """
    pred = []
    for x in testX:
        vote = [0, 0]
        for g in forest:
            vote[int(predict_tree(g, np.array([x]))+1)/2] += 1
        pred.append(vote.index(max(vote))*2 - 1)
    """
    Eout = float(np.sum(np.array(pred)!=testy)) / len(testy)
    return Eout

def mapExpRF(args):
    return expRF(*args)

def prob19():
    trainX, testX, trainy, testy = readdat()
    pool = mp.Pool(processes=20)
    eouts = pool.map(mapExpRF, [(trainX, trainy, testX, testy)]*100)
    #for k in range(100):
    #    eouts.append(expRF(trainX, trainy, testX, testy))
    print "prob19: average Eout = %f" % (sum(eouts) / float(len(eouts)))

def prob20():
    eoutgt = [0.0] * 300
    eoutGt = [0.0] * 300
    eoutGt2 = [0.0] * 300
    trainX, testX, trainy, testy = readdat()
    num_of_exp = 100
    num_of_tree = 300
    for _ in range(num_of_exp):
        forest = []
        #print _,
        for i in range(num_of_tree):
            #print i
            forest.append(Node())
            sample = [random.choice(range(len(trainy))) for j in range(len(trainy))]
            build_tree(forest[-1], trainX[sample], trainy[sample])

            pred = [predict_tree(forest[-1], np.array([x])) for x in testX]
            err = float(np.sum(np.array(pred)!=testy)) / len(testy)
            eoutgt[i] += err / num_of_exp

            pred = predictForest(forest, testX)
            err = float(np.sum(np.array(pred)!=testy)) / len(testy)
            eoutGt[i] += err / num_of_exp
            eoutGt2[i] += err**2 / num_of_exp
    print "prob20: average Eout(g_t): ", eoutgt
    print "prob20: average Eout(G_t): ", eoutGt
    print "prob20: variance Eout(G_t): ", (np.array(eoutGt2) -\
            np.array(eoutGt)**2).tolist()

def main():
    prob161718()
    #prob19()
    prob20()

if __name__ == '__main__':
    main()

