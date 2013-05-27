"""
@author: johnssocks
"""

from sklearn.neighbors import KNeighborsClassifier
from ..ParameterizedKNeighborsClassifier import ParameterizedKNeighborsClassifier
import numpy as np
from nose.tools import assert_equal, assert_raises

class TestParameterizedKNeighborsClassifier(object):

    def __init__(self):
        self.X = np.random.uniform(0, 1, (1000,100))
        self.y = map(int, np.floor(np.random.random_sample(size=1000) * 2))

    def test_with_identity(self):
        n1 = ParameterizedKNeighborsClassifier(M=np.eye(100), n_neighbors=10)
        n1.fit(self.X, self.y)
        X1 = n1.predict(self.X)
        n2 = KNeighborsClassifier(n_neighbors=10)
        n2.fit(self.X, self.y)
        X2 = n2.predict(self.X)
        assert((X1 == X2).all())

    def test_fit(self):
        X = [[1, 2], [3, 4]]
        y = [1, 0]
        M = np.eye(3)
        n = ParameterizedKNeighborsClassifier(M, n_neighbors=10)
        assert_raises(ValueError, n.fit, X, y)
    
    def test_predict(self):
        Xtr = [[1, 2, 3], [3, 4, 5]]
        Xte = [[1, 2], [3, 4]]
        y = [1, 0]
        M = np.eye(3)
        n = ParameterizedKNeighborsClassifier(M, n_neighbors=10)
        n.fit(Xtr, y)
        assert_raises(ValueError, n.predict, Xte)

if __name__ == "__main__":
    import nose
    nose.run(argv=[__file__, '-s', '-v'])
