from __future__ import division, print_function
from itertools import izip
import cPickle as pickle

import scipy.optimize
import numpy as N

import utils
from utils import uprint

try:
    import veusz.embed as veusz
except ImportError:
    veusz = None

debugfit = True

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

    def copy(self):
        p = Param(self.val, minval=self.minval, maxval=self.maxval, frozen=self.frozen)
        p.defval = self.defval
        return p

class Fit:
    """Class to help fitting model, by keeping track of thawed parameters."""

    def __init__(self, pars, model, data):
        """
        pars: dict of name->Param objects
        model: Model object
        data: Data object

        The parmaeters are for the model, but can include a parameter
        called backscale, which controls the scaling of the background
        """

        self.pars = pars
        self.model = model
        self.data = data
        self.refreshThawed()
        self.veuszembed = None
        self.bestlike = -1e99

    def refreshThawed(self):
        """Call this if thawed changes."""
        self.thawed = [
            name for name, par in sorted(self.pars.items()) if not par.frozen]

    def calcProfiles(self):
        """Predict model profiles for each band.
        """

        ne_prof, T_prof, Z_prof = self.model.computeProfs(self.pars)

        # optional background scaling parameter
        if 'backscale' in self.pars:
            backscale = self.pars['backscale'].val
        else:
            backscale = 1.

        profs = []
        for band in self.data.bands:
            modprof = band.calcProjProfile(
                self.data.annuli, ne_prof, T_prof, Z_prof,
                self.model.NH_1022pcm2,
                backscale=backscale)
            profs.append(modprof)

        return profs

    def profLikelihood(self, predprofs):
        """Given predicted profiles, calculate likelihood.
        (this doesn't include the prior)
        """

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

    def getLikelihood(self, vals=None):
        """Get likelihood for parameters given.  Parameters are trimmed to
        their allowed range, applying a penalty

        Also include are the priors from the various components
        """

        if vals is not None:
            self.updateThawed(vals)
        penalty = self.penaltyTrimBounds()
        profs = self.calcProfiles()
        like = self.profLikelihood(profs) - penalty*100
        prior = self.model.prior(self.pars)

        totlike = float(like+prior)

        if debugfit and (totlike-self.bestlike) > 0.1:
            self.bestlike = totlike
            #print("Better fit %.1f" % totlike)
            with utils.AtomicWriteFile("fit.dat") as fout:
                uprint(
                    "likelihood = %g + %g = %g" % (like, prior, totlike),
                    file=fout)
                for p in sorted(self.pars):
                    uprint("%s = %s" % (p, self.pars[p]), file=fout)

        return totlike

    def doFitting(self, silent=False, maxiter=10):
        """Optimize parameters to increase likelihood.

        Returns likelihood
        """

        if not silent:
            uprint('Fitting (Iteration 1)')

        ctr = [0]
        def minfunc(pars):
            like = self.getLikelihood(pars)
            if ctr[0] % 1000 == 0 and not silent:
                uprint('%10i %10.1f' % (ctr[0], like))
            ctr[0] += 1
            return -like

        thawedpars = self.thawedParVals()
        lastlike = self.getLikelihood(thawedpars)
        fpars = thawedpars
        for i in xrange(maxiter):
            fitpars = scipy.optimize.minimize(
                minfunc, fpars, method='Nelder-Mead')
            fpars = fitpars.x
            fitpars = scipy.optimize.minimize(
                minfunc, fpars, method='Powell')
            fpars = fitpars.x
            like = -fitpars.fun
            if abs(lastlike-like) < 0.1:
                break
            if not silent:
                uprint('Iteration %i' % (i+2))
            lastlike = like

        if not silent:
            uprint('Fit Result:   %.1f' % like)
        self.updateThawed(fpars)
        return like

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

    def save(self, filename):
        """Save fit."""

        # don't try to pickle veusz window
        embed = None
        if self.veuszembed is not None:
            embed = self.veuszembed
            self.veuszembed = None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        if embed is not None:
            self.veuszembed = embed
