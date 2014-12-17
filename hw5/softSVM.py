#!/usr/bin/env python

import numpy as np
import svmutil, svm

def readdat():
    with open("features.train", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        trainX = np.hsplit(data, [1, len(data)])[1]
        trainy = np.hsplit(data, [1, len(data)])[0].reshape(-1)

    with open("features.test", "r") as f:
        data = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
        testX = np.hsplit(data, [1, len(data)])[1]
        testy = np.hsplit(data, [1, len(data)])[0].reshape(-1)

    return trainX.tolist(), testX.tolist(), [int(i) for i in trainy], [int(i)
            for i in testy.tolist()]

def mul_label_2_bin(trainy, testy, lab):
    for i in range(len(trainy)):
        if trainy[i] == lab:
            trainy[i] = 1
        else:
            trainy[i] = -1
    for i in range(len(testy)):
        if testy[i] == lab:
            testy[i] = 1
        else:
            testy[i] = -1

def libsvm_sv_to_vec():
    pass

def prob15():
    trainX, testX, trainy, testy = readdat()
    mul_label_2_bin(trainy, testy, 0)

    m = svmutil.svm_train(trainy, trainX, '-s 0 -t 0 -c 0.01 -h 0')
    p_label, p_acc, p_val = svmutil.svm_predict(testy, testX, m)

    sv_coef = [sv_c[0] for sv_c in m.get_sv_coef()]
    sv = []
    for v in m.get_SV():
        t = []
        for i in range(1, 3):
            if i in v:
                t.append(v[i])
            else: t.append(0)
        sv.append(t)
    w = np.dot(np.array(sv).T, np.array(sv_coef))
    print "prob15: |w| = %f" % np.sqrt(w[0]**2 + w[1]**2)

def prob1617():
    Ein = []
    alpha = []
    for i in range(0, 10, 2):
        trainX, testX, trainy, testy = readdat()
        mul_label_2_bin(trainy, testy, i)

        m = svmutil.svm_train(trainy, trainX,
                    '-s 0 -t 1 -c 0.01 -d 2 -g 1 -r 1 -h 0')
        p_label, p_acc, p_val = svmutil.svm_predict(testy, testX, m)

        sigma_alpha = sum([abs(sv_c[0]) for sv_c in m.get_sv_coef()])

        Ein.append("prob16: %d v.s. not %d, E_in = %f%%" % (i, i,
            100.0-p_acc[0]))
        alpha.append("prob17: %d v.s. not %d, sigma_alpha = %f" % (i, i,
            sigma_alpha))

    for s in Ein:
        print s
    for s in alpha:
        print s

def prob18():
    ans = {'dist':[], 'xi':[], 'n_sv': [], 'Eout': []}
    C = [0.001, 0.01, 0.1, 1, 10]
    for c in C:
        trainX, testX, trainy, testy = readdat()
        mul_label_2_bin(trainy, testy, 0)

        m = svmutil.svm_train(trainy, trainX, '-s 0 -t 2 -c %f -g 100 -h 0'%(c))
        p_label, p_acc, p_val = svmutil.svm_predict(testy, testX, m)

        b = m.rho[0]
        sv_coef = [sv_c[0] for sv_c in m.get_sv_coef()]
        sv = []
        for v in m.get_SV():
            t = []
            for i in range(1, 3):
                if i in v:
                    t.append(v[i])
                else: t.append(0)
            sv.append(t)

        for i in range(len(sv_coef)):
            if abs(sv_coef[i]) != c:
                break
        dist = 0.0
        for x, ay in zip(sv, sv_coef):
            dist += ay * np.exp(-100 * ((x[0]-sv[i][0])**2 + (x[1]-sv[i][1])**2))
        dist = abs(dist + trainy[i]*b)

        wsquare = 0.0
        for x, ay in zip(sv, sv_coef):
            for y, ay2 in zip(sv, sv_coef):
                wsquare += ay*ay2*np.exp(-100 * ((x[0]-y[0])**2 + (x[1]-y[1])**2))
        wsquare = np.sqrt(wsquare)

        dist = dist/wsquare

        xi = 0.0
        for x, ay in zip(sv, sv_coef):
            y = int(ay/abs(ay))
            if abs(ay) == c:
                xi += 1
                for sv_i, ay2 in zip(sv, sv_coef):
                    xi += -y * ay2 *\
                        np.exp(-100 * ((x[0]-sv_i[0])**2 + (x[1]-sv_i[1])**2))
                xi += -y * b
        #print xi

        ans['dist'].append(dist)
        ans['xi'].append(xi)
        ans['n_sv'].append(len(m.get_SV()))
        ans['Eout'].append((100.0-p_acc[0]))
        #ans['obj_value'].append(m.obj[0])
    #print ans

    for i in range(len(C)):
        print "prob18: c=%f," % (C[i]),
        for k in ans.keys():
            print "%s=%f" % (k, ans[k][i]),
        print ""
    print """objective value is printed in the libsvm verbose as obj, from the
    first verbose to the last one is c = 0.001, 0.01, 0.1, 1, 10"""

def prob19():
    gamma = [1, 10, 100, 1000, 10000]
    Eout = []
    for g in gamma:
        trainX, testX, trainy, testy = readdat()
        mul_label_2_bin(trainy, testy, 0)

        m = svmutil.svm_train(trainy, trainX, '-s 0 -t 2 -c 0.1 -g %f -h 0'%(g))
        p_label, p_acc, p_val = svmutil.svm_predict(testy, testX, m)

        Eout.append("prob19: gamma=%d Eout=%f%%" % (g, 100.0 - p_acc[0]))

    for e in Eout:
        print e

def prob20():
    import random
    gamma = [1, 10, 100, 1000, 10000]
    chosen = {1:0, 10:0, 100:0, 1000:0, 10000:0}
    for _ in range(100):
        Eout = []
        for g in gamma:
            trainX, testX, trainy, testy = readdat()
            mul_label_2_bin(trainy, testy, 0)

            trainX = zip(trainX, trainy)
            random.shuffle(trainX)
            trainX, trainy = zip(*trainX)
            valX = trainX[:1000]
            valy = trainy[:1000]
            trainX = trainX[1000:]
            trainy = trainy[1000:]

            m = svmutil.svm_train(trainy, trainX, '-s 0 -t 2 -c 0.1 -g %f -h 0'%(g))
            p_label, p_acc, p_val = svmutil.svm_predict(valy, valX, m)

            Eout.append(100.0 - p_acc[0])
        chosen[gamma[Eout.index(min(Eout))]] += 1
    print "prob20: ",
    for k in chosen.keys():
        print "gamma=%d:%d, " % (k, chosen[k]),
    print ""

def main():
    prob15()
    prob1617()
    prob18()
    prob19()
    prob20()

if __name__ == '__main__':
    main()
