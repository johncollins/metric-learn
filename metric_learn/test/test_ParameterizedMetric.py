"""
    @date: 5/27/2013
    @author: John Collins

    test_ParameterizedMetric
    -------------------
"""

import scipy.linalg
import numpy as np
import numpy.linalg
from nose.tools import assert_raises, assert_equal
from ..ParameterizedMetric import ParameterizedMetric

class TestParameterizedMetric(object):

    def __init__(self):
        self.PM1 = ParameterizedMetric(np.eye(2))
        self.PM2 = ParameterizedMetric(np.array([[1, 0], [0, 2]]))

    def test_d(self):
        x1 = np.array([1, 2])
        x2 = np.array([-1, 1])
        d1 = np.linalg.norm(x1 - x2)
        d2 = self.PM1.d(x1, x2)
        assert_equal(d1, d2)
        d2 = self.PM2.d(x1, x2)
        assert(d1 != d2)
    
    def test_get_A(self):
        assert((self.PM1.get_A() == np.array([[1, 0], [0, 1]])).all())
    
    def get_M(self):
        assert((self.PM1.get_M() == np.real(scipy.linalg.sqrtm(sqrtnp.array([[1, 0], [0, 1]]))).all()))

    def test_transform_space(self):
        M = self.PM2.get_M()
        X = np.array([[1, 2], [3, 4]])
        tX = np.dot(X, M)
        assert((tX == self.PM2.transform_space(X)).all())
