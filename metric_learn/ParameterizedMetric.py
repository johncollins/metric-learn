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
        self.M = scipy.linalg.sqrtm(A)

    def d(self, x, y):
        """
            Use to calculate the distance between two points
        """
        return np.real(np.sqrt(np.dot(np.dot((x - y), self.A), (x - y))) )

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
