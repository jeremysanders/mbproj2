from __future__ import division, print_function
from itertools import izip

import scipy.optimize
import numpy as N

import utils

try:
    import veusz.embed as veusz
except ImportError:
    veusz = None

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

    def __init__(self, pars, model, data):
        self.pars = pars
        self.model = model
        self.data = data
        self.refreshThawed()
        self.veuszembed = None

    def refreshThawed(self):
        """Call this if thawed changes."""
        self.thawed = [
            name for name, par in sorted(self.pars.items()) if not par.frozen]

    def calcProfiles(self):
        """Predict model profiles for each band.
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
        return float(like)

    def doFitting(self, silent=False):
        """Optimize parameters to increase likelihood.

        Returns likelihood
        """

        if not silent:
            print('Fitting')
        thawedpars = self.thawedParVals()

        ctr = [0]
        def minfunc(pars):
            like = self.getLikelihood(pars)
            if ctr[0] % 1000 == 0 and not silent:
                print('%10i %10.1f' % (ctr[0], like))
            ctr[0] += 1
            return -like

        fitpars = scipy.optimize.minimize(minfunc, thawedpars, method='Powell')
        fitpars = scipy.optimize.minimize(minfunc, fitpars.x, method='Nelder-Mead')
        if not silent:
            print('Fit Result:   %.1f' % -fitpars.fun)
        self.updateThawed(fitpars.x)
        return -fitpars.fun

    def plotProfiles(self):
        if veusz is None:
            raise RuntimeError('Veusz not found')

        if self.veuszembed is None:
            embed = self.veuszembed = veusz.Embedded()
            grid = self.veuszembed.Root.Add('page').Add('grid', columns=1)
            xaxis = grid.Add(
                'axis', name='x', direction='horizontal', log=True,
                autoRange='+2%')

            for i in range(len(self.data.bands)):
                graph = grid.Add('graph', name='graph%i' % i)
                xydata = graph.Add(
                    'xy', marker='none', name='data',
                    xData='radius', yData='data_%i' %i)
                xymodel = graph.Add(
                    'xy', marker='none', name='model', color='red',
                    xData='radius', yData='model_%i' %i)

            edges = self.data.annuli.edges_arcmin
            self.veuszembed.SetData(
                'radius', 0.5*(edges[1:]+edges[:-1]),
                symerr=0.5*(edges[1:]-edges[:-1]))
            grid.Action('zeroMargins')

        profs = self.calcProfiles()
        for i, (band, prof) in enumerate(zip(self.data.bands, profs)):
            self.veuszembed.SetData('data_%i' % i, band.cts)
            self.veuszembed.SetData('model_%i' % i, prof)
