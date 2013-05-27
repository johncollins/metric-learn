"""
    @date: 5/27/2013
    @author: John Collins

    MetricLearningAlgorithm
    -----------------------
    An attempt to the define the abstracton, to which one will program
    when implementing a new metric learning algorithm.
"""

import scipy.linalg
from ParameterizedMetric import ParameterizedMetric

class MetricLearningAlgorithm(object):
    """
        Abstract class capturing everything common among metric learning algorithms.
    """

    def __init__(self, X, side_information, side_information_type='labels', parameters={}):
        self.X = X
        self.y = side_information
        self.run_setup(parameters=parameters)
        self.learned_metric = ParameterizedMetric(self.learn_metric())

    def run_setup(self, parameters={}):
        if parameters is None:
            self.set_default_parameters()
        else:
            self.set_parameters(parameters=parameters)
        self.run_algorithm_specific_setup()

    @property
    def set_default_parameters(self):
        """
            Sensible defaults for the algorithm
        """
        raise NotImplementedError('Not Implemented')

    @property
    def set_parameters(self, **kwargs):
        """
            Customize the parameters
        """
        raise NotImplementedError('Not Implemented')

    @property 
    def algorithm_specific_setup(self):
        """
            Create any other variables necessary for the algorithm
        """
        raise NotImplementedError('Not Implemented')

    @property
    def learn_metric(self):
        """
            Actual meat of the algorithm.
        """
        raise NotImplementedError('Not Implemented Error')

    def get_metric(self):
        return self.learned_metric
