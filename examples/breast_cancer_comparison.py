import numpy as np
import os
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Read the csv file wherein the class is the first column and transform to data
# Everything is numeric so we can just use a numpy array right off the bat
data = np.genfromtxt(os.path.join('data', 'breast_cancer', 'wdbc.csv'), skip_header=False, delimiter=',')
X = data[:,2:]
y = [line.strip().split(',')[1] for line in open(os.path.join('data', 'breast_cancer', 'wdbc.csv'))]
y = np.array([1 if yy=='M' else 0 for yy in y])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

#knn1 = KNeighborsClassifier(n_neighbors=3)
#Xn = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#scores = cross_validation.cross_val_score(knn1, Xn, y, cv=10)
#print 'Normalized K Nearest Neighbors Score = %f +/- %f' % (scores.mean(), scores.std())

#knn2 = KNeighborsClassifier(n_neighbors=3)
#scores = cross_validation.cross_val_score(knn, X, y, cv=10)
#print 'Non-normalized K Nearest Neighbors Score = %f +/- %f' % (scores.mean(), scores.std())

# How good could we do, with say Random Forest
#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(X_train, y_train)
#scores = cross_validation.cross_val_score(rf, X, y, cv=10)
#print 'Random Forest Score = %f +/- %f' % (scores.mean(), scores.std())

# Now we'll learn a metric on the same dataset and see if we can improve
import sys
sys.path.append('..')
from metric_learn.itml.ItmlAlgorithm import ItmlAlgorithm
from metric_learn.ParameterizedKNeighborsClassifier import ParameterizedKNeighborsClassifier
from sklearn.cross_validation import KFold
kf = cross_validation.KFold(len(y), 10)
scores1, scores2, scores3, scoresrf = [], [], [], []
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    itml_alg = ItmlAlgorithm(X_train, y_test, 
        parameters={'constant_factor': 0.5, 'verbose': False, 'thresh' : 10e-4})
    itml = itml_alg.get_metric()
    knn1 = KNeighborsClassifier(n_neighbors=3)
    knn2 = KNeighborsClassifier(n_neighbors=3)
    rf = RandomForestClassifier(n_estimators=100)
    knn3 = ParameterizedKNeighborsClassifier(M=itml.get_M(), n_neighbors=3)
    knn1.fit(X_train, y_train)
    knn2.fit(X_train, y_train)
    knn3.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    scores1.append(knn1.score(X_test, y_test))
    scores2.append(knn2.score(X_test, y_test))
    scores3.append(knn3.score(X_test, y_test))
    scoresrf.append(rf.score(X_test, y_test))
scores1 = np.array(scores1)
scores2 = np.array(scores2)
scores3 = np.array(scores3)
scoresrf = np.array(scoresrf)
print 'Knn normalized Score = %f +/- %f' % (scores1.mean(), scores1.std())
print 'Knn                  = %f +/- %f' % (scores2.mean(), scores2.std())
print 'Knn with itml        = %f +/- %f' % (scores3.mean(), scores3.std())
print 'Random Forest        = %f +/- %f' % (scoresrf.mean(), scoresrf.std())
