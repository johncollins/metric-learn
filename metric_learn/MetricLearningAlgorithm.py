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
from abc import ABCMeta, abstractmethod
import numpy as np

class MetricLearningAlgorithm(object):
    """
        Abstract class capturing everything common among metric learning algorithms.
    """
    __metaclass__ = ABCMeta

    def __init__(self, X, side_information, side_information_type='labels', parameters={}):
        self.X = np.array(X)
        #TODO: If X is a data frame or some other useful type, extract its useful info
        self.y = np.array(side_information).squeeze() 
        #TODO: assuming labels for now. Change this. Maybe it's own abstraction
        self.run_setup(parameters=parameters)
        self.learned_metric = ParameterizedMetric(self.learn_metric())

    def run_setup(self, parameters={}):
        self.set_default_parameters()
        if parameters != {}:
            self.set_parameters(parameters=parameters)
        self.run_algorithm_specific_setup()

    @abstractmethod
    def set_default_parameters(self):
        """
            Sensible defaults for the algorithm
        """
        raise NotImplementedError('Not Implemented')

    def set_parameters(self, parameters):
        """
            Set any customizable parameters specified
        """
        for (key, value) in parameters.items():
            if key in self.parameters:
                self.parameters[key] = value

    @abstractmethod
    def run_algorithm_specific_setup(self):
        """
            Create any other variables necessary for the algorithm
        """
        raise NotImplementedError('Not Implemented')

    @abstractmethod
    def learn_metric(self):
        """
            Actual meat of the algorithm.
        """
        raise NotImplementedError('Not Implemented Error')

    def get_metric(self):
        return self.learned_metric
