from numba import double
from numba.decorators import jit, autojit
import numba

@autojit
def update(A, V, beta):
    return A + beta * A * (V * V.T) * A    

@jit(argtypes=(double[:,:],double[:,:],double))
def update2(A, V, beta):
    return A + beta * A * (V * V.T) * A    

def itml_alg(C, X, A0, params = {}):
    """
        Core ITML learning algorithm

        C: 4 column matrix
        column 1, 2: index of constrained points.  Indexes between 1 and n
        column 3: 1 if points are similar, -1 if dissimilar
        column 4: right-hand side (lower or upper bound, depending on 
                  whether points are similar or dissimilar)

        X: (n x m) data matrix - each row corresponds to a single instance

        A0: (m x m) regularization matrix

        params: algorithm parameters - see see SetDefaultParams for defaults
                params.thresh: algorithm convergence threshold
                params.gamma: gamma value for slack variables
                params.max_iters: maximum number of iterations

        returns A: learned Mahalanobis matrix
    """
    import numpy as np
    import sys
    sys.path.append('..')
    from set_default_params import set_default_params

    if params == {}:
	params = set_default_params()
    tol, gamma, max_iters = params['thresh'], params['gamma'], params['max_iters'];
    
    # check to make sure that no 2 constrained vectors 
    # are identical. If they are, remove the constraint
    invalid = []
    for i, c in enumerate(C):
        i1, i2 = C[i, :2]
        v = X[i1, :] - X[i2, :]
        if np.linalg.norm(v) < 10e-10:
            invalid.append(i)
    C = np.delete(C, invalid, 0)
    i = 0
    iteration = 0
    c = C.shape[0]
    lambdacurrent = np.zeros((c))
    bhat = np.array(C[:,3])
    lambdaold = np.array(lambdacurrent)
    conv = np.inf
    A = np.matrix(A0)
    
    while True:
        i1, i2 = C[i,:2]
        v = X[i1, :] - X[i2, :]
        V = np.asmatrix(v) # must be matrix interactive
        V = V.T 	   # make into column vector
        wtw = (V.T * A * V)[0, 0] # should be scalar

        if np.abs(bhat[i]) < 10e-10:
            print('bhat should never be 0!')
            exit()

        if np.inf == gamma:
            gamma_proj = 1
        else:
            gamma_proj = gamma / (gamma + 1)
            
        if C[i, 2] == 1:
            alpha = min(lambdacurrent[i], gamma_proj * (1.0 / (wtw) - 1.0 / bhat[i]))
            lambdacurrent[i] = lambdacurrent[i] - alpha
            beta = alpha / (1 - alpha * wtw)        
            bhat[i] = 1.0 / ((1.0 / bhat[i]) + (alpha / gamma))        
        elif C[i, 2] == -1:
            alpha = min(lambdacurrent[i], gamma_proj * (1.0 / bhat[i] - 1.0 / wtw))
            lambdacurrent[i] = lambdacurrent[i] - alpha
            beta = -1 * alpha / (1 + alpha * wtw) 
            bhat[i] = 1.0 / ((1.0 / bhat[i]) - (alpha / gamma))
        
        #print '... in iterative matrix mult'	
        #A += beta * A * (V * V.T) * A
        A = update2(A, V, beta) # numba version
        #print '... after iterative matrix mult'	
        #print np.linalg.det(A)
        if i == c - 1:
            normsum = np.linalg.norm(lambdacurrent) + np.linalg.norm(lambdaold)
            if normsum == 0:
                break
            else:
                conv = np.linalg.norm(lambdaold - lambdacurrent, ord = 1) / normsum
                if (conv < tol) or (iteration > max_iters):
                    break
            lambdaold = np.array(lambdacurrent)
        #print conv
        i = ((i+1) % c)
        iteration += 1
        if iteration % 5000 == 0:       
            print('itml iter: %d, conv = %f' % (iteration, conv))

    print('itml converged to tol: %f, iteration: %d' % (conv, iteration))
    return np.asarray(A)

if __name__ == "__main__":
    import numpy as np
    import sys
    import os
    np.random.seed(0)

    sys.path.append('..')
    from set_default_params import set_default_params
    from itml_alg import itml_alg

    X = np.genfromtxt(os.path.join('..', 'data', 'test_X.csv'), delimiter = ',')
    y = np.genfromtxt(os.path.join('..', 'data', 'test_y.csv'), delimiter = ',')
    C = np.genfromtxt(os.path.join('..', 'data', 'test_C.csv'), delimiter = ',')
    C[:, 0] = C[:, 0] - 1.0
    C[:, 1] = C[:, 1] - 1.0
    A0 = np.eye(X.shape[1])
    params = set_default_params()
    params['thresh'] = 0.1
    A = itml_alg(C, X, A0, params)
    A_matlab = np.genfromtxt(os.path.join('..', 'data', 'test_A_matlab.csv'), delimiter = ',')
    if np.sum(np.fabs(A - A_matlab)) < 0.0001:
        print 'Perfect'
    else:
        print A
        print A_matlab
        print 'Broken'
