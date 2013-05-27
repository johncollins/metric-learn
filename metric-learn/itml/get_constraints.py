def get_constraints(y, num_constraints, l, u):
    """
        C = GetConstraints(y, num_constraints, l, u)
        Get ITML constraint matrix from true labels.  
        See itml_alg.py for description of the constraint matrix format
    """
    
    import numpy as np
    import random
    random.seed(0)
    # Make quartets for pairs of indices
    # [index1, index2, 1 or -1, l or u]
    # Note that l always goes with 1
    # and u always goes with -1
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

if __name__ == "__main__":
    import random
    import numpy as np
    y = np.random.choice([1, 2, 3], 10)
    num_constraints = 20
    l = 20
    u = 80
    print get_constraints(y, num_constraints, l, u)
