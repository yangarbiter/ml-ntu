
import numpy as np

def readdat():
    with open("hw3_train.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split(' ')[1:]] for line in f.readlines()])
        trainX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        trainX = np.hstack((np.ones((np.shape(trainX)[0], 1)), trainX))
        trainy = np.hsplit(data, [len(data[0])-1,
            len(data)])[1].reshape(np.shape(trainX)[0], 1)

    with open("hw3_test.dat", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split(' ')[1:]] for line in f.readlines()])
        testX = np.hsplit(data, [len(data[0])-1, len(data)])[0]
        #add constant to feature
        testX = np.hstack((np.ones((np.shape(testX)[0], 1)), testX))
        testy = np.hsplit(data, [len(data[0])-1,
            len(data)])[1].reshape(np.shape(testX)[0], 1)

    return trainX, testX, trainy, testy


sigmoid = lambda s: 1.0/(1.0+np.e**(-s))
h = lambda w, X: sigmoid(np.dot(X, w.T))
cross_entropy_err = lambda w, X, y: \
    np.sum(-np.log(sigmoid(y*np.dot(X, w.T)))) / float(len(y))
err_grad = lambda w, X, y: np.sum(sigmoid(-y*np.dot(X, w.T)) * (-y*X), axis=0) \
    / float(len(y))

pred_to_label = lambda y: np.ceil(y-0.5)*2-1
predict = lambda w, X: pred_to_label(h(w, X) > 0.5)
zo_error = lambda y, ans: np.sum(y != ans) / float(len(y))

def quiz18():
    eta = 0.001
    T = 2000
    trainX, testX, trainy, testy = readdat()
    w = np.array([[0] * np.shape(trainX)[1]])
    for i in range(T):
        w = w - eta * err_grad(w, trainX, trainy)
        #print cross_entropy_err(w, trainX, trainy)
    #print h(w, testX)
    print("quiz 18: zero-one error on test set: %f(eta=0.001)" %
            zo_error(predict(w, testX), testy))

def quiz19():
    eta = 0.01
    T = 2000
    trainX, testX, trainy, testy = readdat()
    w = np.array([[0] * np.shape(trainX)[1]])
    for i in range(T):
        w = w - eta * err_grad(w, trainX, trainy)
        #print cross_entropy_err(w, trainX, trainy)
    #print h(w, testX)
    print("quiz 19: zero-one error on test set: %f(eta=0.01)" %
            zo_error(predict(w, testX), testy))

def quiz20():
    #sgd
    eta = 0.001
    T = 2000
    trainX, testX, trainy, testy = readdat()
    w = np.array([[0] * np.shape(trainX)[1]])
    for i in range(T/len(trainX)):
        for j in range(len(trainX)):
            w = w - eta * err_grad(w, trainX[j], trainy)
        #print cross_entropy_err(w, trainX, trainy)
    #print h(w, testX)
    print("quiz 20(pick example in cyclic order):"),
    print("zero-one error on test set: %f" %
            zo_error(predict(w, testX), testy))

def main():
    quiz18()
    quiz19()
    quiz20()

if __name__ == '__main__':
    main()
