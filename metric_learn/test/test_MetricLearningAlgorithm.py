"""
    @date: 5/27/2013
    @author: John Collins

    test_MetricLearningAlgorithm
"""

import scipy.linalg
from ..MetricLearningAlgorithm import MetricLearningAlgorithm
from nose.tools import assert_raises, assert_equal
import numpy as np

class TestMetricLearningAlgorithm(object):

    def __init__(self):
        self.X = np.array([[1, 2, 3], [4, 5, 6]])
        self.y = [1, 2]

    def test_instantiation(self):
        assert_raises(TypeError, MetricLearningAlgorithm, self.X, self.y)
        MLA = ConcreteMetricLearningAlgorithm(self.X, self.y, parameters = {'s': [3, 2, 1], 'tenor': 54})
        print MLA.parameters.keys()
        assert_equal(MLA.parameters['s'], [3, 2, 1]) 
        assert_equal(MLA.parameters['tenor'], 54) 
        assert_equal(MLA.foo, 'bar')
        MLA = ConcreteMetricLearningAlgorithm(self.X, self.y)
        print MLA.parameters
        assert_equal(MLA.parameters['s'], [1, 2, 3]) 
        assert_equal(MLA.parameters['tenor'], 45) 
        assert_equal(MLA.foo, 'bar')

class ConcreteMetricLearningAlgorithm(MetricLearningAlgorithm):
    """
        For testing the abstract MetricLearningAlgorithm class
    """

    def set_default_parameters(self):
        self.parameters = {'s': [1,2,3], 'tenor': 45, 'sax': 90}

    def run_algorithm_specific_setup(self):
        self.foo = 'bar' 

    def learn_metric(self):
        return np.eye(3)
