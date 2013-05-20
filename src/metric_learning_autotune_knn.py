def metric_learning_autotune_knn(metric_learn_alg, y, X, method=method, params={}):
    """
        metric_learn_alg:   runs itml over various parameters of gamma, 
                            choosing that with the highest accuracy. 
        returns: Mahalanobis matrix A for learned distance metric
    """
    import numpy as np
    from metric_learning import metric_learning

    if params == {}:
        params = set_default_params()

    # regularize to the identity matrix
    A0 = np.eye(X.shape[1])

    # define gamma values for slack variables
    gammas = 10.0 ** np.arange(-4, 4, 1)
    
    accs = np.zeros((len(gammas), 1))
    for i, gamme in enumerate(gammas):
        print '\tTuning burg kernel learning: gamma = %f' % gamma
        params['gamma'] = gamma
        accs(i) = cross_validate_knn(y, X, lambda y, X: metric_learning(metric_learn_alg, y, X, A0, method=method, params=params), 2, params.k)
    
    max_ind = np.argmax(accs)
    gamma = gammas[max_ind]
    print('\tOptimal gamma value: %f' % gamma)
    params['gamma'] = gamma
    A = metric_learning(metric_learn_alg, y, X, A0, method=method, params=params)

    return A

if __name__ == "__main__":
    pass
