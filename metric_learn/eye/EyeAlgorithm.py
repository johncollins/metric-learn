"""
    @date: 5/27/2013
    @author: John Collins

    EyeAlgorithm
    -------------------
    Learn the identity matrix. For testing and comparison.
"""
    
import numpy as np
from ..MetricLearningAlgorithm import MetricLearningAlgorithm

class EyeAlgorithm(MetricLearningAlgorithm):
    """
        Learn the identity matrix. For testing and comparison.
    """
    
    def set_default_parameters(self):
        pass

    def run_algorithm_specific_setup(self):
        pass

    def learn_metric(self):
        """
            "Learn" the identity matrix
        """
        return np.eye(np.array(self.X).shape[1])

