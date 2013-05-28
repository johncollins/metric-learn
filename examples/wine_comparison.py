import numpy as np
import os

# Read the csv file wherein the class is the first column and transform to data
# Everything is numeric so we can just use a numpy array right off the bat
data = np.genfromtxt(os.path.join('data', 'wine', 'wine.csv'), skip_header=True, delimiter=',')
y, X = data[:,0], data[:,1:]
#X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# For comparison, do classification using the standars KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
score = knn.score(X, y)
print 'K Nearest Neighbors Score = %f' % score

# Now we'll learn a metric on the same dataset and see if we can improve
import sys
sys.path.append('..')
from metric_learn.itml.ItmlAlgorithm import ItmlAlgorithm
from metric_learn.ParameterizedKNeighborsClassifier import ParameterizedKNeighborsClassifier
itml_alg = ItmlAlgorithm(X, y, parameters={'constant_factor': 1})
itml = itml_alg.get_metric()
knn = ParameterizedKNeighborsClassifier(M=itml.get_M(), n_neighbors=3)
knn.fit(X, y)
print knn.score(X, y)
