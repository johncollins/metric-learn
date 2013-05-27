from ..utils import compute_distance_extremes, get_constraints
import numpy as np
import random
from nose.tools import assert_equal

class TestUtils(object):
    
    def __init__(self):
        random.seed(0)
        self.X = np.array([[1, 10, 1], [3, 6, 7], [9, 11, 1], [1, 2, 1]])
        self.lower, self.upper = 20, 80
        self.M = np.eye(self.X.shape[1])

    def test_compute_distance_extremes(self):
        self.l, self.u = compute_distance_extremes(self.X, self.lower, self.upper, self.M)
        assert_equal(self.l, 28.275)
        assert_equal(self.u, 115.275)
