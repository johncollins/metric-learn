"""
    @date: 5/27/2013
    @author: John Collins

    test_EyeAlgorithm
"""

import numpy as np
from ..EyeAlgorithm import EyeAlgorithm
from scipy.spatial.distance import pdist
from nose.tools import assert_raises 

class TestEyeAlgorithm(object):

    def __init__(self):
        self.X = np.random.uniform(0, 1, (10, 15))
        eye_alg = EyeAlgorithm(self.X, None)
        self.metric = eye_alg.get_metric()
        self.tX = self.metric.transform_space(self.X)

    def test_learn_metric(self):
        assert((pdist(self.X)==pdist(self.tX)).all())
