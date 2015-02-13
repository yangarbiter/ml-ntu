
import numpy as np
import random

def gen_artificial_data(size):
    x = np.random.uniform(-1,1,size)
    noise = np.random.uniform(0,1,size)
    y = [-1 if (i<0 and n>0.2) or (i>=0 and n<=0.2) else 1 for i, n in zip(x, noise)]
    return np.array([x]).T, np.array([y]), sum(noise<=0.2)/float(size)

def readdat():
    with open("hw2_train.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split(' ')[1:]] for line in f.readlines()])
        trainX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        trainy = np.hsplit(data, [len(data[0])-1, len(data)])[1].reshape(np.shape(trainX)[0])

    with open("hw2_test.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split(' ')[1:]] for line in f.readlines()])
        testX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        testy = np.hsplit(data, [len(data[0])-1, len(data)])[1].reshape(np.shape(testX)[0])

    return trainX, testX, trainy, testy


def decision_stump(trainX, trainy):
    DIM = np.shape(trainX)[1]
    best = [0, 0, 0, len(trainX)] #s, i, t, e
    for i in xrange(DIM):
        xi = sorted(trainX[:, i].tolist() + [max(trainX[:,i])+1])
        for t in xrange(len(xi)-1):
            if xi[t] == xi[t+1]:
                continue
            theta = (xi[t]+xi[t+1])/2
            err = len(trainX) - np.sum(trainy == np.sign(trainX[:, i]-theta))
            errn = len(trainX) - np.sum(trainy == (-1)*np.sign(trainX[:, i]-theta))
            if best[3] > err:
                best[0] = 1
                best[1] = i
                best[2] = theta
                best[3] = err
            if best[3] > errn:
                best[0] = -1
                best[1] = i
                best[2] = theta
                best[3] = errn
    return best[0], best[1], best[2], best[3]/float(np.shape(trainX)[0])

def no19():
    trainX, testX, trainy, testy = readdat()
    s, i, theta, err = decision_stump(trainX, trainy)
    print "In sample error rate: %f" % (err)

    err = len(testy) - np.sum(testy == s*np.sign(testX[:, i]-theta))
    print "Out sample error rate: %f" % (err/float(np.shape(testy)[0]))

def no17():
    avg_err_in = 0
    avg_err_out = 0
    for i in xrange(5000):
        x, y, e_out = gen_artificial_data(20)
        s, i, theta, err = decision_stump(x, y)
        avg_err_in += err
        avg_err_out += 0.5 + 0.3*s*(np.abs(theta)-1)
    print "Average in sample error rate: %f" % (avg_err_in/5000.0)
    print "Average out sample error rate: %f" % (avg_err_out/5000.0)



def main():
    no17()
    no19()
    exit()

if __name__ == '__main__':
    main()
