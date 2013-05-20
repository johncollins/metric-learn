import numpy as np
import os
import sys
from itml_alg import itml_alg

sys.path.append('..')
from set_default_params import set_default_params

X = np.genfromtxt(os.path.join('..', 'data', 'test_X.csv'), delimiter = ',')
y = np.genfromtxt(os.path.join('..', 'data', 'test_y.csv'), delimiter = ',')
C = np.genfromtxt(os.path.join('..', 'data', 'test_C.csv'), delimiter = ',')

C[:, 0] = C[:, 0] - 1.0
C[:, 1] = C[:, 1] - 1.0
A0 = np.eye(X.shape[1])
params = set_default_params(method='itml')
params['thresh'] = 0.1
A = itml_alg(C, X, A0, params)
A_matlab = np.genfromtxt(os.path.join('..', 'data', 'test_A_matlab.csv'), delimiter = ',')
if np.sum(np.fabs(A - A_matlab)) < 0.0001:
    print 'Perfect'
else:
    print A
    print A_matlab
    print 'Broken'
