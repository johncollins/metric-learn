"""
    @date: 5/27/2013
    @author: John Collins
    
    ItmlAlgorithm
    -------------
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

from ..MetricLearningAlgorithm import MetricLearningAlgorithm
from numba.decorators import jit, autojit
from numba import double
from utils import compute_distance_extremes, get_constraints
import numba
import numpy as np

class ItmlAlgorithm(MetricLearningAlgorithm):
    """
        Implementation of Information Theoretic Metric Learning
        Kulis et. al 
        #TODO: Add more reference materials
    """

    def set_default_parameters(self):
        self.parameters = {
            'gamma' : 1.0,
            'beta' : 1.0,
            'constant_factor' : 40.0,
            'type4_rank' : 5.0,
            'thresh' : 10e-5,
            'k' : 4,
            'max_iters' : 100000,
            'lower_percentile': 5,
            'upper_percentile': 95,
            'A0': np.eye(self.X.shape[1]),
            'verbose': True
        }

    def run_algorithm_specific_setup(self):
        self.l, self.u = compute_distance_extremes(self.X, self.parameters['lower_percentile'], 
                                        self.parameters['upper_percentile'], np.eye(self.X.shape[1]))
        num_constraints = int(self.parameters['constant_factor'] * (max(self.y.shape) * (max(self.y.shape))-1))
        self.constraints = get_constraints(self.y, num_constraints, self.l, self.u)
        # check to make sure that no pair of constrained vectors 
        # are identical. If they are, remove the constraint
        #TODO: Clean this up and make it pythonic
        #invalid = []
        #for i, c in enumerate(C):
        #    i1, i2 = C[i, :2]
        #    v = X[i1, :] - X[i2, :]
        #    if np.linalg.norm(v) < 10e-10:
        #        invalid.append(i)
        #C = np.delete(C, invalid, 0)
        #print self.constraints
        valid = np.array([np.linalg.norm(self.X[c[0],:] - self.X[c[1],:]) > 10e-10 for c in self.constraints])
        #print valid
        self.constraints = self.constraints[valid,:]
        self.A0 = self.parameters['A0']

    def learn_metric(self):
        tol, gamma, max_iters = self.parameters['thresh'], self.parameters['gamma'], self.parameters['max_iters']
        C = self.constraints
        X, y = self.X, self.y
        i = 0
        iteration = 0
        c = C.shape[0]
        lambdacurrent = np.zeros((c))
        bhat = np.array(C[:,3])
        lambdaold = np.array(lambdacurrent)
        converged = np.inf
        A = np.matrix(self.A0)
        verbose = self.parameters['verbose']

        while True:
            V = np.asmatrix(X[C[i, 0], :] - X[C[i, 1], :]).T # column vector x - y
            wtw = (V.T * A * V)[0, 0] # a scalar

            if np.abs(bhat[i]) < 10e-10:
                print('bhat should never be 0!')
                exit()

            if gamma == np.inf:
                gamma_proj = 1
            else:
                gamma_proj = gamma / (gamma + 1)
                
            if C[i, 2] == 1: # lower bound constraint
                alpha = min(lambdacurrent[i], gamma_proj * (1.0 / (wtw) - 1.0 / bhat[i]))
                lambdacurrent[i] = lambdacurrent[i] - alpha
                beta = alpha / (1 - alpha * wtw)        
                bhat[i] = 1.0 / ((1.0 / bhat[i]) + (alpha / gamma))        
            elif C[i, 2] == -1: # upper bound constraint
                alpha = min(lambdacurrent[i], gamma_proj * (1.0 / bhat[i] - 1.0 / wtw))
                lambdacurrent[i] = lambdacurrent[i] - alpha
                beta = -1 * alpha / (1 + alpha * wtw) 
                bhat[i] = 1.0 / ((1.0 / bhat[i]) - (alpha / gamma))
            
            A += beta * A * (V * V.T) * A # non-numba version
            # A = update(A, V, beta) # numba version not working
            if i == c - 1:
                normsum = np.linalg.norm(lambdacurrent) + np.linalg.norm(lambdaold)
                if normsum == 0:
                    break
                else:
                    converged = np.linalg.norm(lambdaold - lambdacurrent, ord = 1) / normsum
                    if (converged < tol) or (iteration > max_iters):
                        break
                lambdaold = np.array(lambdacurrent)
            i = ((i+1) % c)
            iteration += 1
            if iteration % 5000 == 0 and verbose:       
                print('itml iter: %d, converged = %f' % (iteration, converged))

        if verbose:
            print('itml converged to tol: %f, iteration: %d' % (converged, iteration))
        return np.asarray(A)

    """
    @autojit
    def update(A, V, beta):
        return A + beta * A * (V * V.T) * A    

    @jit(argtypes=(double[:,:],double[:,:],double))
    def update2(A, V, beta):
        return A + beta * A * (V * V.T) * A    
    """
