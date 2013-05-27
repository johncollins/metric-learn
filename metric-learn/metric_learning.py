def metric_learning(metric_learning_alg, y, X, A0 = None, method = 'itml', params = {}):
    """
        Wrapper script that takes in a set of data points and their true labels
        computes upper and lower bounds for these constraints, and then invokes
        metric learning
    """
    from set_default_params import set_default_params
    from itml.compute_distance_extremes import compute_distance_extremes
    from itml.get_constraints import get_constraints
    import numpy as np
    
    if params == {}:
        params = set_default_params(method=method)

    if A0 is None:
        A0 = np.eye(X.shape[1])
    
    # Determine similarity/dissimilarity constraints from the true labels
    (l, u) = compute_distance_extremes(X, 5, 95, A0)

    # Choose the number of constraints to be const_factors 
    # times the number of distinct pairs of classes
    k = len(set(y))
    num_constraints = int(params['const_factor'] * (k * (k - 1)))
    C = get_constraints(y, num_constraints, l, u)
    
    try:
        A = metric_learning_alg(C, X, A0, params) # might be an issue here if alg is a string
        print A
    except:
        print('Unable to learn mahalanobis matrix');
        A = np.zeros(X.shape[1])
    return A

if __name__ == "__main__":
    import numpy as np
    from eye.eye_alg import eye_alg
    X = np.random.normal(0, 1, (100, 10))
    y = np.random.random_integers(0, 1, 100)
    A0 = np.eye(X.shape[1])
    A = metric_learning(eye_alg, y, X, A0, method = 'itml', params = {})
