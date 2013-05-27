def compute_distance_extremes(X, a, b, M):
    """
        Usage:
        from compute_distance_extremes import compute_distance_extremes
        (l, u) = compute_distance_extremes(X, a, b, M)

        Computes sample histogram of the distances between rows of X and returns
        the value of these distances at the a^th and b^th percentils.  This
        method is used to determine the upper and lower bounds for
        similarity / dissimilarity constraints.  

        Args:
        X: (n x m) data matrix 
        a: lower bound percentile between 1 and 100
        b: upper bound percentile between 1 and 100
        M: Mahalanobis matrix to compute distances 

        Returns:
        l: distance corresponding to a^th percentile
        u: distance corresponding the b^th percentile
    """
    import numpy as np
    import random
    random.seed(0)

    if (a < 1) or (a > 100):
        raise Exception('a must be between 1 and 100')

    if (b < 1) or (b > 100):
        raise Exception('b must be between 1 and 100')

    n = X.shape[0]

    num_trials = min(100, n * (n - 1) / 2);
    
    # sample with replacement
    dists = np.zeros((num_trials, 1))
    for i in xrange(num_trials):
        j1 = np.floor(random.uniform(0, n))
        j2 = np.floor(random.uniform(0, n))
        dists[i] = np.dot(np.dot((X[j1, :] - X[j2, :]), M), (X[j1, :] - X[j2, :]).T)

    # return frequencies and bin extremeties
    (f, ext) = np.histogram(dists, bins = 100) # specify bins by percentile
    # get bin centers
    c = [(ext[i]+float(ext[i+1])) / 2 for i in xrange(len(ext) - 1)]

    # get values at percentiles
    l = c[int(np.floor(a)) - 1]           # get counts for lower percentile
    u = c[int(np.floor(b)) - 1]           # get counts for higher percentile
    
    return l, u

def get_constraints(y, num_constraints, l, u):
    """
        get_constraints(y, num_constraints, l, u)
        Get ITML constraint matrix from true labels.  
    """
    
    import numpy as np
    import random
    random.seed(0)
    # Make quartets for pairs of indices [index1, index2, 1 or -1, l or u]
    # Note that l always goes with 1 and u always goes with -1
    m = len(y)
    C = np.zeros((num_constraints, 4))
    for k in xrange(num_constraints):
        i = np.floor(random.uniform(0, m))
        j = np.floor(random.uniform(0, m))
        if y[i] == y[j]:
            C[k, :] = (i, j, 1, l)
        else:
            C[k, :] = (i, j, -1, u)
    return np.array(C)

