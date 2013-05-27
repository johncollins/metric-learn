"""TestEfficacy.py

Test the efficacy of your learned metric on a set of data

The algorithm should learn a matrix A suxh that A^(1/2) * X

The result should have a better silhouette score than X.
"""

def test_efficacy(X, y, M):
    """
    M is the factor matix such that A = M * M'
    """
    from sklearn.metrics import silhouette_score
    Xt = np.dot(X, M).real

    S = silhouette_score(X, y)
    St = silhouette_score(Xt, y)
    if S > St:
        print 'Bad idea to use this'
    elif S == St:
        print 'Nothing to gain from using this'
    else:
        print 'This works'

    print 'S = %f' % S
    print 'St = %f' % St
    return (S, St)

if __name__ == "__main__":
    import os
    import numpy as np
    import scipy.linalg
    from metric_learn.set_default_params import set_default_params
    from metric_learn.itml.itml_alg import itml_alg
    import itertools

    X = np.genfromtxt(os.path.join('data', 'test_X.csv'), delimiter = ',')
    y = np.genfromtxt(os.path.join('data', 'test_y.csv'), delimiter = ',')
    C = np.genfromtxt(os.path.join('data', 'test_C.csv'), delimiter = ',')
    C[:, 0] = C[:, 0] - 1.0
    C[:, 1] = C[:, 1] - 1.0
    A0 = np.eye(X.shape[1])
    params = set_default_params(method='itml')
    params['thresh'] = 0.01
    A = itml_alg(C, X, A0, params)
    M = scipy.linalg.sqrtm(A)
    old_score, new_score = test_efficacy(X, y, M)
    print '%f ~> %f' % (old_score, new_score)
