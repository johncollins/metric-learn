# -*- coding: utf-8 -*-
"""
K Nearest Neighbor Classification
---------------------------------
This function is built especially for a learned metric
parameterized by the matrix A where this function takes
the matrix M such that A = M * M'
"""

# Author: John Collins <johnssocks@gmail.com>

def knn(ytr, Xtr, M, k, Xte):
    """K Nearest Neighbors classifier
       
    y_hat = knn(y, X, M, k, Xt)
    Perform knn classification on each row of Xt using a learned metric M
    M is the factor matrix : A = M * M'

    Parameters
    ----------

    ytr: vector of labels, string or int
        The known responses upon which to train our classifier
    Xtr: 2D n*p array of numbers
        The data matrix where rows are observations and columns are features
    M: The p*p factor matrix of the (assumedly) learned matrix A
    k: The number of nearest neighbors to use
    Xt: The new data from which to predict responses

    Attributes
    ----------
    `centroids_` : array-like, shape = [n_classes, n_features]
        Centroid of each class

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [7,7], [1,2], [7,8], [2,1], [8,7], [1,0], [8,9], [11, -4], [14, 1]])
    >>> y = np.array([1, 2, 1, 2, 1, 2, 1, 2, 2, 2])
    >>> M = np.eye(X.shape[1])
    >>> k = 4
    >>> Xtr, Xte = X[:8,:], X[8:,:]
    >>> ytr, yte = y[:8], y[8:]
    >>> print Xtr.shape
    >>> print Xte.shape
    >>> print 'Simple Test'
    >>> print '--------------'
    >>> print 'predictions'
    >>> print knn(ytr, Xtr, M, k, Xte)
    >>> print 'actual'
    >>> print yte
    >>> print '\n'
    
    [1]

    References
    ----------
    Tibshirani, R., Hastie, T., Narasimhan, B., & Chu, G. (2002). Diagnosis of
    multiple cancer types by shrunken centroids of gene expression. Proceedings
    of the National Academy of Sciences of the United States of America,
    99(10), 6567-6572. The National Academy of Sciences.

    """
    import numpy as np

    add1 = 0
    if min(ytr) == 0:
        ytr += 1
        add1 = 1
    
    (n, m) = Xtr.shape
    (nt, m) = Xte.shape
    
    K = np.dot(np.dot(Xtr, M), np.dot(M.T, Xte.T))
    l = np.zeros((n))
    for i in xrange(n):
        l[i] = np.dot(np.dot(Xtr[i, :], M), np.dot(M.T, Xtr[i, :].T))

    lt = np.zeros((nt))
    for i in xrange(nt):
        lt[i] = np.dot(np.dot(Xte[i, :], M), np.dot(M.T, Xte[i, :].T))

    D = np.zeros((n, nt));
    for i in xrange(n):
        for j in xrange(nt):
            D[i, j] = l[i] + lt[j] - 2 * K[i, j]
    
    inds = np.argsort(D, axis = 0)
    
    preds = np.zeros((nt), dtype = int)
    for i in xrange(nt):
        counts = [0 for ii in xrange(2)]
        for j in xrange(k):        
            if ytr[inds[j, i]] > len(counts):
                counts.append(1)
            else:
                counts[ytr[inds[j, i]] - 1] += 1
        v, preds[i] = (max(counts), int(np.argmax(counts) + 1))
    if add1 == 1:
        preds -= 1 
    return preds

if __name__ == "__main__":
    """ Simple Test """
    import numpy as np
    X = np.array([[1,1], [7,7], [1,2], [7,8], [2,1], [8,7], [1,0], [8,9], [11, -4], [14, 1]])
    y = np.array([1, 2, 1, 2, 1, 2, 1, 2, 2, 2])
    M = np.eye(X.shape[1])
    k = 4
    Xtr, Xte = X[:8,:], X[8:,:]
    ytr, yte = y[:8], y[8:]
    #print Xtr.shape
    #print Xte.shape
    print 'Simple Test'
    print '--------------'
    print 'predictions'
    print knn(ytr, Xtr, M, k, Xte)
    print 'actual'
    print yte
    
    """ Elaborate Test """
    import numpy as np
    import os
    X = np.genfromtxt(os.path.join('data', 'test_X.csv'), delimiter = ',', dtype = float)
    y = np.genfromtxt(os.path.join('data', 'test_y.csv'), delimiter = ',', dtype = int)
    M = np.eye(X.shape[1])
    k = 1
    inds = np.genfromtxt(os.path.join('data', 'test_inds.csv'), delimiter = ',', dtype = int)
    inds_tr = np.where(inds == 1)[0]
    inds_te = np.where(inds == 0)[0]
    Xtr = X[inds_tr, :]
    Xtr, Xte = X[inds_tr, :], X[inds_te, :]
    ytr, yte = y[inds_tr], y[inds_te]
    ypred = knn(ytr, Xtr, M, k, Xte)
    print 'Elaborate Test'
    print '--------------'
    print 'predictions'
    print ypred
    print 'actual'
    print yte
    matlab_accuracy = 0.958333
    accuracy = float(sum(yte == ypred)) / len(yte)
    if np.abs(matlab_accuracy - accuracy) > 0.00001:
        print 'Problem'
    else:
        print 'Perfect'
