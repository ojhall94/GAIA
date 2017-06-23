#!/usr/bin/env/python3
#Oliver J. Hall

import numpy as np

class Prior:
    '''Simple class that returns 0 if parameters are within bounds, -inf if without
    '''
    def __init__(self, _bounds):
        self.bounds = _bounds

    def __call__(self, p):
        if not all(b[0] < v < b[1] for v, b in zip(p,self.bounds)):
            return - np.inf
        return 0
