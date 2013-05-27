# -*- coding: utf-8 -*-
"""
Parameterized K Nearest Neighbors Classification
---------------------------------
This function is built especially for a learned metric
parameterized by the matrix A where this function takes
the matrix M such that A = M * M'
"""

# Author: John Collins <johnssocks@gmail.com>

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class ParameterizedKNeighborsClassifier(KNeighborsClassifier):
    
    def __init__(self, M=None, n_neighbors=5, 
                    weights='uniform',
                    algorithm='auto', leaf_size=30,
                    warn_on_equidistant=True, p=2):
        super(ParameterizedKNeighborsClassifier, self).__init__(n_neighbors=n_neighbors, 
                    weights=weights,
                    algorithm=algorithm, leaf_size=leaf_size,
                    warn_on_equidistant=warn_on_equidistant, p=p)
        self.M = np.array(M)
        if self.M.shape[0] != self.M.shape[1]:
            raise ValuError('Parameterizing matrix M with dimensions (%d, %d) is not square' % dim(self.M))

    def fit(self, X, y):
        """
            Before using KNeighborsClassifier.fit
            Convert training data X to X * M
        """
        X = np.array(X)
        if X.shape[1] != self.M.shape[0]:
            raise ValueError('Data matrix X with dimensions (%d, %d)' 
            'and paramaterizing matrix M with dimensions (%d, %d) are'
            'not compatible' % (X.shape[0], X.shape[1], self.M.shape[0], 
            self.M.shape[1]))
        X = np.dot(X, self.M)
        return super(ParameterizedKNeighborsClassifier, self).fit(X, y)

    def predict(self, X):
        """
            Before using KNeighborsClassifier.predict
            Convert new data X to X * M
        """
        X = np.array(X)
        if X.shape[1] != self.M.shape[0]:
            raise ValueError('Data matrix X with dimensions (%d, %d)'
            'and paramaterizing matrix M with dimensions (%d, %d) are'
            'not compatible' % (X.shape[0], X.shape[1], self.M.shape[0], 
            self.M.shape[1]))
        X = np.dot(X, self.M)
        return super(ParameterizedKNeighborsClassifier, self).predict(X)
