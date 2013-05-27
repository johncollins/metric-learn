def eye_alg(C, X, A0, params = {}):
    """
        Dummy function to return an identity matrix
        For comparison without a learned metric
    """
    import numpy as np
    return np.eye(X.shape[1], X.shape[1])
