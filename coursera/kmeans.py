#!/usr/bin/env python

import numpy as np

def readdat():
    with open("hw4_kmeans_train.dat", "r") as f:
        X = np.array([[float(i.strip()) for i in line.split()] for line in f.readlines()])
    return X

def prob1920():
    X = readdat()
    K = 10
    repeat = 20
    results = []

    for i in range(repeat):
        center = X[np.random.randint(len(X), size=K)]
        Ein = np.inf
        while True:
            group = [ [] for i in range(K) ]
            for x in X:
                g = min([(i, np.linalg.norm(center[i] - x)) for i in range(len(center))],
                        key=lambda ent: ent[1])[0]
                group[g].append(x)

            for i in range(K):
                center[i] = np.mean(group[i], axis=0)

            tEin = sum([np.sum((x-center[i])**2) \
                    for i in range(K) for x in group[i]]) / float(len(X))
            if tEin == Ein:
                break
            Ein = tEin
        results.append(Ein)
    print "[prob19] avgEin= %f" % (np.mean(results))

def main():
    prob1920()

if __name__ == '__main__':
    main()
