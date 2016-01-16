from __future__ import division, print_function
from itertools import izip

import scipy.optimize

import utils

class Param:
    """Model parameter."""

    def __init__(self, val, minval=-1e99, maxval=1e99, frozen=False):
        val = float(val)
        self.val = val
        self.defval = val
        self.minval = minval
        self.maxval = maxval
        self.frozen = frozen

    def __repr__(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.frozen)

class Fit:
    """Class to help fitting model, by keeping track of thawed parameters."""

    def __init__(self, pars, model, data, showfit=True):
        self.pars = pars
        self.model = model
        self.data = data
        self.showfit = showfit
        self.thawed = [
            name for name, par in sorted(pars.items()) if not par.frozen]
        self.ctr = 0

    def calcProfiles(self):
        """Predict model profiles for each band.

        Returns profiles, log-likelihood
        """

        ne_prof, T_prof, Z_prof = self.model.computeProfs(self.pars)

        profs = []
        for band in self.data.bands:
            modprof = band.calcProjProfile(
                self.data.annuli, ne_prof, T_prof, Z_prof,
                self.model.NH_1022pcm2)
            profs.append(modprof)

        return profs

    def profLikelihood(self, predprofs):
        """Given predicted profiles, calculate likelihood."""

        likelihood = 0.
        for band, predprof in izip(self.data.bands, predprofs):
            likelihood += utils.cashLogLikelihood(band.cts, predprof)
        return likelihood

    def thawedParVals(self):
        """Return values of thawed parameters."""
        return [self.pars[name].val for name in self.thawed]

    def updateThawed(self, vals):
        for val, name in izip(vals, self.thawed):
            self.pars[name].val = val

    def penaltyTrimBounds(self):
        """For each thawed parameter, compute a penalty index to subtract from
        likelihood, and trim parameter to range."""

        penalty = 0.
        for name in self.thawed:
            par = self.pars[name]
            if par.val > par.maxval:
                penalty += (par.val-par.maxval) / (par.maxval-par.minval)
                par.val = par.maxval
            elif par.val < par.minval:
                penalty += (par.minval-par.val) / (par.maxval-par.minval)
                par.val = par.minval
        return penalty

    def getLikelihood(self, vals):
        """Get likelihood for paramters given.  Parameters are trimmed to
        their allowed range, applying a penalty
        """

        self.updateThawed(vals)
        penalty = self.penaltyTrimBounds()
        profs = self.calcProfiles()
        like = self.profLikelihood(profs) - penalty*100

        if self.ctr % 1000 == 0 and self.showfit:
            print('%10i %10.1f' % (self.ctr, like))
        self.ctr += 1

        return float(like)

    def doFitting(self):
        """Optimize parameters to increase likelihood."""

        if self.showfit:
            print('Fitting')
        thawedpars = self.thawedParVals()
        fitpars = scipy.optimize.minimize(
            lambda pars: -self.getLikelihood(pars), thawedpars)
        if self.showfit:
            print('Fit Result:   %.1f' % -fitpars.fun)
        self.updateThawed(fitpars.x)
