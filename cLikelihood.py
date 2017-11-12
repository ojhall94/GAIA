import numpy as np

# The "foreground" linear likelihood:
class Likelihood:
    def __init__(self,_x,_y,_xerr, _lnprior, _Model):
        self.x = _x
        self.y = _y
        self.xerr = _xerr
        self.lnprior = _lnprior
        self.Model = _Model

    def lnlike_fg(self, p):
        return self.Model.fg(p)

# The "background" outlier likelihood:
    def lnlike_bg(self, p):
        return self.Model.bg(p)

# Full probabilistic model.
    def lnprob(self, p):
        Q = p[-1]
        # b, sigrc, Q, o, sigo = p

        # First check the prior.
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf, None

        # Compute the vector of foreground likelihoods and include the q prior.
        ll_fg = self.lnlike_fg(p)
        arg1 = ll_fg + np.log(Q)

        # Compute the vector of background likelihoods and include the q prior.
        ll_bg = self.lnlike_bg(p)
        arg2 = ll_bg + np.log(1.0 - Q)

        # Combine these using log-add-exp for numerical stability.
        ll = np.nansum(np.logaddexp(arg1, arg2))

        # We're using emcee's "blobs" feature in order to keep track of the
        # foreground and background likelihoods for reasons that will become
        # clear soon.
        return lp + ll

    def __call__(self, p):
        logL = self.lnprob(p)
        return logL
