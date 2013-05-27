def cross_validate_knn(y, X, tCL, k, knn_size):
    """
        Cross-validation for evaluating the k-nearest neighbor classifier with
        a learned metric.  Performs k-fold cross validation, training on the
        training fold and evaluating on the test fold

        y: (n x 1) true labels

        X: (n x m) data matrix

        tCL: Metric learning algorithm that takes in true labels as first
        argument, and data as a second

        k: Number of cross-validated folds

        knn_size: size of nearest neighbor window

        Returns: 
            acc: cross-validated accuracy
            pred: predictions on test set for each row in X
    """
    import numpy as np
    from knn import knn
    from scipy.linalg import sqrtm
    import os

    (n, m) = X.shape
    if n != len(y):
        print('ERROR: num rows of X must equal length of y');
        return
    
    # permute the rows of X and y
    shuffled_indices = np.random.permutation(range(len(y)))
    # test (random seed for testing)
    #shuffled_indices = np.genfromtxt(os.path.join('data', 'rand_inds_python.csv'), dtype = int)
    
    y = y[shuffled_indices] 
    X = X[shuffled_indices,:]
    pred = np.array(y) # copy y

    for i in xrange(k):
        test_start = np.ceil(float(n) / k) * i
        test_end = np.ceil(float(n) / k) * (i + 1)
        yt = np.array([], dtype = int)
        Xt = np.zeros((0, m))
        if (i > 0):
            yt = y[:test_start]
            Xt = X[:test_start, :]
        if (i < k):
            yt = np.append(y[test_end:len(y)], yt, axis = 0)
            Xt = np.append(X[test_end:len(y), :], Xt, axis = 0)
        
        # train model
        A = tCL(yt, Xt)
        M = sqrtm(A).real

        #evaluate model 
        XT = X[test_start:test_end, :]
        yT = y[test_start:test_end]
        
        pred[test_start:test_end] = knn(yt, Xt, M, knn_size, XT)
    acc = float(sum(pred==y)) / n
        
    return acc, pred

if __name__ == "__main__":

    """ Elaborate Test """
    from set_default_params import set_default_params
    from metric_learning import metric_learning
    from itml.itml_alg import itml_alg
    import numpy as np
    import os

    X = np.genfromtxt(os.path.join('data', 'test_X.csv'), delimiter = ',', dtype = float)
    y = np.genfromtxt(os.path.join('data', 'test_y.csv'), delimiter = ',', dtype = int)
    params = set_default_params(method='itml')
    params['thresh'] = 0.1
    K = 10 # number of folds
    A0 = np.eye(X.shape[1])
    acc, pred = cross_validate_knn(y, X, lambda y, X: metric_learning(itml_alg, y, X, A0, params), K, params['k'])
    print 'accuracy = %f' % acc
