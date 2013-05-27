from itml_alg import itml_alg
import numpy as np
import os
import sys

sys.path.append('..')
from cross_validate_knn import cross_validate_knn
from metric_learning import metric_learning
from set_default_params import set_default_params

print 'Loading iris data'

X = np.genfromtxt(os.path.join('..', 'data', 'test_X.csv'), delimiter = ',')
y = np.genfromtxt(os.path.join('..', 'data', 'test_y.csv'), delimiter = ',', dtype = int)

print 'Running ITML'
num_folds = 10
knn_neighbor_size = 5
params = set_default_params()
params['thresh'] = 0.1
A0 = np.eye(X.shape[1])

acc, preds = cross_validate_knn(y, X, lambda y, X: metric_learning(itml_alg, y, X, A0, params), num_folds, knn_neighbor_size)

print 'kNN cross-validated accuracy = %f' % acc 
