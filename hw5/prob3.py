
from sklearn import svm
import svmutil
import numpy as np
X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
y = [-1, -1, -1, 1, 1, 1, 1]

clf = svm.SVC(C=999999.0, kernel='poly', coef0=1, degree=2, gamma=1)
clf.fit(X, y)
print clf.support_vectors_
print clf.dual_coef_

m = svmutil.svm_train(y, X, '-s 0 -t 1 -d 2 -g 1 -r 1 -c 99999')
print m
print m.get_sv_coef()
print m.get_SV()
print m.rho[0]

b = []
for i in clf.support_:
    b.append([])
    for j, ya in zip(clf.support_, clf.dual_coef_[0]):
        b[-1].append(ya * (X[i][0]*X[j][0] + X[i][1]*X[j][1] + 1)**2)
    b[-1] = 9*(y[i] - sum(b[-1]))
print b

ayk = []
for x, ya in zip(clf.support_vectors_, clf.dual_coef_[0]):
    ayk.append([ya * x[1]**2, ya * x[0]**2, ya * 2 * x[1], ya * 2 * x[0], ya * 1])
#x2^2, x1^2, x2, x1, 1
print 9 * np.sum(np.array(ayk), axis=0)
