def itml_on_en_reduced():

    print '... reading data'

    # local
    data_folder = os.path.join('..', 'xtaldata')
    # remote
    data_folder = '/ThesisData'

    X = np.load(os.path.join(data_folder, 'en_reduced_X_3class.pickle.npy'))
    y = np.load(os.path.join(data_folder, 'y.pickle.npy'))
    A0 = np.eye(X.shape[1])
    params = set_default_params()
    C = get_constraints(y, len(y), 10, 90)

    print '... learning metric'

    A = itml_alg(C, X, A0, params)

    np.save('/ThesisData/A_itml.pickle', A)

    return A

from get_constraints import get_constraints
from itml_alg import itml_alg

import numpy as np
import sys
import os

sys.path.append('..')
from set_default_params import set_default_params

# There seems to be some kind of memory leak with this algorithm
#import cloud
#jid = cloud.call(lambda: itml_on_en_reduced(), _vol = 'iscore', _env = 'scientific', _type = 'f2', _cores = 1)
