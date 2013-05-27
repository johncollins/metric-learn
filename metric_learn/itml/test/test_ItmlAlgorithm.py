"""
    @date: 5/27/2013
    @author: John Collins

    test_ItmlAlgorithm
"""

import numpy as np
from ..ItmlAlgorithm import ItmlAlgorithm
from nose.tools import assert_raises 
from sklearn.metrics import silhouette_score

class TestItmlAlgorithm(object):

    def __init__(self):
        X1 = np.random.normal(0, 1, (10, 15))
        X2 = np.random.normal(1, 1, (10, 15))
        self.X = np.concatenate((X1, X2), axis=0)
        y1 = np.ones((1, 10))
        y2 = np.zeros((1, 10))
        self.y = np.concatenate((y1, y2), axis=1).squeeze()
        itml_alg = ItmlAlgorithm(self.X, self.y)
        self.metric = itml_alg.get_metric()
        self.tX = self.metric.transform_space(self.X)

    def test_learn_metric(self):
        from sklearn.metrics import silhouette_score
        S = silhouette_score(self.X, self.y)
        St = silhouette_score(self.tX, self.y)
        # using a metric learning alg should improve 
        assert(S <= St)
