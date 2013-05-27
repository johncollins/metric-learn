"""
    @date: 5/27/2013
    @author: John Collins

    ParameterizedMetric
    -------------------
    Abstraction defining the interaction with a learned metric
"""

import scipy.linalg
import numpy as np

class ParameterizedMetric(object):
    """
        Abstraction to represent a learned metric
    """

    def __init__(self, A):
        self.A = A
        self.M = scipy.linalg.sqrtm(A).real

    def d(self, x, y):
        """
            Use to calculate the distance between two points
        """
        x = np.asmatrix(x)
        y = np.asmatrix(y)
        return np.sqrt(((x - y) * np.asmatrix(self.A) * (x - y).T) [0, 0])

    def get_A(self):
        """
            Get the matrix representation of the learned metric
        """
        return self.A

    def get_M(self):
        """
            Get the more useful learned-metric matrix root
        """
        return self.M

    def transform_space(self, X):
        """
            Transform an entire space by this learned metric
        """
        return np.dot(X, self.M)
