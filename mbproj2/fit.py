# -*- coding: utf-8 -*-
# Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the Free
# Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
# MA 02111-1307, USA

"""Contains classes representing the model parameters and model fit.

"""

from __future__ import division, print_function, absolute_import

from six.moves import range, zip
import six.moves.cPickle as pickle
import scipy.optimize
import numpy as N

from . import utils
from .utils import uprint

try:
    import veusz.embed as veusz
except ImportError:
    veusz = None

debugfit = True

class Param:
    """Model parameter."""

    def __init__(self, val, minval=-1e99, maxval=1e99, frozen=False):
        """
        :param float val: value of parameter
        :param float minval: minimum allowed value
        :param float maxval: maximum allowed value
        :param bool frozen: whether parameter is allowed to vary
        """

        val = float(val)
        self.val = val
        self.defval = val
        self.minval = minval
        self.maxval = maxval
        self.frozen = frozen

    def copy(self):
        """Return a copy (remember to reimplement if overriding)."""
        p = Param(
            self._val,
            minval=self.minval, maxval=self.maxval, frozen=self.frozen)
        p.defval = self.defval
        return p

    def __repr__(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.frozen)

    def prior(self):
        """Return log prior for parameter."""
        if self.val < self.minval or self.val > self.maxval:
            return -N.inf
        return 0.

class Fit:
    """Class to help fitting model, by keeping track of thawed parameters."""

    def __init__(self, pars, model, data):
        """
        :param dict[str,Param] pars: parameters for model
        :param Model model: Model to fit
        :param Data data: Data to fit

        The parameters pars are for the model, but a parameter called
        backscale can be included, which controls the scaling of the
        background
        """

        self.pars = pars
        self.model = model
        self.data = data
        self.refreshThawed()
        self.veuszembed = None
        self.bestlike = -1e99

    def refreshThawed(self):
        """Call this after making changes to which parameters are thawed."""
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

    def likeFromProfs(self, predprofs):
        """Given predicted profiles, calculate log likelihood
        (excluding prior).

        :param list[numpy.array] predprofs: input profiles
        """
        likelihood = 0.
        for band, predprof in zip(self.data.bands, predprofs):
            likelihood += utils.cashLogLikelihood(band.cts, predprof)
        return likelihood

    def thawedParVals(self):
        """Return list of numeric values of thawed parameters."""
        return [self.pars[name].val for name in self.thawed]

    def updateThawed(self, vals):
        """Update values of parameter Param objects which are thawed.
        :param list[float] vals: numerical values of parameters
        """
        for val, name in zip(vals, self.thawed):
            self.pars[name].val = val

    def getLikelihood(self, vals=None):
        """Get likelihood for parameters given.

        Also include are the priors from the various components
        """

        if vals is not None:
            self.updateThawed(vals)

        # prior on parameters
        parprior = sum( (self.pars[p].prior() for p in self.pars) )
        if not N.isfinite(parprior):
            # don't want to evaluate profiles for invalid parameters
            return -N.inf

        profs = self.calcProfiles()
        like = self.likeFromProfs(profs)
        prior = self.model.prior(self.pars) + parprior

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
        """Optimize parameters to increase likelihood.  Uses scipy's
        Nelder-Mead and Powell optimizers, repeating if a new minimum
        is found.

        :param bool silent: print output during fitting
        :param int maxiter: maximum number of iterations
        :returns: log likelihood
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
        for i in range(maxiter):
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
        """Plot surface brightness profiles of model and data if Veusz is
        installed.
        """
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
        """Save fit in Python pickle format."""

        # don't try to pickle veusz window
        embed = None
        if self.veuszembed is not None:
            embed = self.veuszembed
            self.veuszembed = None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        if embed is not None:
            self.veuszembed = embed

def genericPopulationMinimizer(
    function, gennewparams, popnum=1000, keepfrac=0.8, maxiter=1000, sigmabreak=1e-3):
    """Minimize using a set of populations."""

    # generate initial list of function values and parameters
    uprint('Populating 0th generation')
    popfun = []
    poppar = []
    while len(poppar) < popnum:
        par = gennewparams()
        fun = function(par)
        if N.isfinite(fun):
            popfun.append(fun)
            poppar.append(par)

    popfun = N.array(popfun)
    poppar = N.array(poppar)

    # number of items to keep in each iteration
    keepnum = int(popnum*keepfrac)
    # number of new parameters to create
    newnum = popnum-keepnum

    # temporary locations for new function values and new parameters
    newfun = N.zeros(newnum)
    newpar = N.zeros((newnum, poppar.shape[1]))

    for gen in range(maxiter):

        # Sort into function order, keeping fraction. This could be a
        # partition, but this let's us keep track of what the best
        # item is.
        sortidxs = N.argsort(popfun)[:keepnum]
        popfun = popfun[sortidxs]
        poppar = poppar[sortidxs]

        bestpars = ' '.join(['%6.3f' % p for p in poppar[0]])
        bestfun = popfun[0]
        rmsfun = N.sqrt(N.mean(popfun**2))
        uprint('Gen %3i, best %7.4f, rms %7.4f, pars=%s' % (
            gen, bestfun, rmsfun, bestpars))

        if abs(bestfun-rmsfun) < sigmabreak:
            break

        # create new set of parameters
        i = 0
        while i < newnum:
            # choose random best parameter
            par1 = poppar[N.random.randint(keepnum)]
            par2 = poppar[N.random.randint(keepnum)]
            # adjust by standard deviations
            movpar = N.random.normal()*(par2-par1) + par1

            # if it is valid, keep
            fun = function(movpar)
            if N.isfinite(fun):
                newfun[i] = fun
                newpar[i, :] = movpar
                i += 1

        # merge list
        popfun = N.concatenate( (popfun, newfun) )
        poppar = N.vstack( (poppar, newpar) )

    # find lowest value and return value and parameters
    bestidx = N.argmin(popfun)
    return popfun[bestidx], poppar[bestidx]

def populationMinimiser(fit, popnum=1000, keepfrac=0.8, maxiter=1000, sigmabreak=1e-3):

    # get list of parameters min and max values
    minvals = []
    maxvals = []
    for name, par in sorted(fit.pars.items()):
        if not par.frozen:
            minvals.append(par.minval)
            maxvals.append(par.maxval)
    minvals = N.array(minvals)
    maxvals = N.array(maxvals)

    def gennewparams():
        r = N.random.rand(len(minvals))
        pars = minvals + r*(maxvals-minvals)
        return pars

    def minfunc(pars):
        like = fit.getLikelihood(pars)
        return -like

    bestfun, bestpar = genericPopulationMinimizer(
        minfunc, gennewparams, popnum=popnum, keepfrac=keepfrac,
        maxiter=maxiter, sigmabreak=sigmabreak)

    uprint('Best likelihood after minimization:', -bestfun)
    fit.updateThawed(bestpar)
