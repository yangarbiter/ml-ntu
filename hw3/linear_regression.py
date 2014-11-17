
import numpy as np

ein = lambda w, X, y: np.sum((np.dot(X, w) - y)**2 / np.shape(X)[0])
predict = lambda w, X: np.sign(np.dot(X, w))
err01 = lambda y, y_ans: np.sum(y != y_ans)

def gen_data(size):
    X = np.random.random([size, 2]) * 2 - 1
    y = np.sign(X[:,0]**2 + X[:,1]**2 - 0.6) # target function
    y *= np.sign(np.random.random(np.shape(y)) - 0.1) # 0.1 of chance to be noise
    return X, y

def dat_transform(X):
    X = np.hstack((X,X[:,0:1]*X[:,1:2]))
    X = np.hstack((X, X[:,0:1]**2))
    X = np.hstack((X, X[:,1:2]**2))
    X = np.hstack((np.ones((np.shape(X)[0], 1)), X))
    return X

def quiz13():
    avg_err = 0
    for i in range(1000):
        X, y = gen_data(1000)
        X = np.hstack((np.ones((np.shape(X)[0], 1)), X))

        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        avg_err += err01(predict(w, X), y) / 1000.0
    avg_err /= 1000.0
    print("quiz 13: avg 01Ein: %f" % avg_err)

def quiz14():
    X, y = gen_data(1000)
    X = dat_transform(X)

    ws = [np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5]),
            np.array([-1, -0.05, 0.08, 0.13, 1.5, 15]),
            np.array([-1, -0.05, 0.08, 0.13, 15, 1.5]),
            np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05]),
            np.array([-1, -1.5, 0.08, 0.13, 0.05, 1.5])]

    print('quiz 14:')
    for s, w in zip(['a', 'b', 'c', 'd', 'e'], ws):
        print(s + ' ' + str(err01(predict(w, X), y) / 1000.0))

def quiz15():
    avg_err = 0
    for i in range(1000):
        X, y = gen_data(1000)
        X = dat_transform(X)

        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

        X_test, y_test = gen_data(1000)
        X_test = dat_transform(X_test)
        avg_err += err01(predict(w, X_test), y_test) / 1000.0
    avg_err /= 1000.0
    print('quiz 15: avg 01Eout is %f' % avg_err)

def main():
    quiz13()
    quiz14()
    quiz15()

if __name__ == '__main__':
    main()
