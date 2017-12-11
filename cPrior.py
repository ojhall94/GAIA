#!/usr/bin/env/python3
#Oliver J. Hall

import numpy as np

class Prior:
    def __init__(self, _bounds):
        self.bounds = _bounds

    def lnprior(self, p):
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        return 0

    def __call__(self, p):
        prior = self.lnprior(p)
        return prior
