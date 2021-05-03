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

"""A fit to the model.
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

class Fit:
    """Class to help fitting model, by keeping track of thawed parameters."""

    def __init__(self, pars, model, data):
        """
        :param dict[str,ParamBase] pars: parameters for model
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
        self._veuszembed = []
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
            backscale = self.pars['backscale'].v
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
            # ignore bins where there is no exposure or no area
            validbins = (band.exposures>0) & (band.areascales>0)

            likelihood += utils.cashLogLikelihood(
                band.cts[validbins], predprof[validbins])
        return likelihood

    def thawedParVals(self):
        """Return list of numeric values of thawed parameters."""
        return [self.pars[name].v for name in self.thawed]

    def updateThawed(self, vals):
        """Update values of parameter ParamBase objects which are thawed.
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

    def plotProfiles(self,
                     logx=True,
                     logy=False,
                     showback=True,
                     rates=False,
                     title='mbproj2 profiles',
                     mode='update',
                     savefilename=None,
                     exportfilename=None):
        """Plot surface brightness profiles of model and data if Veusz is
        installed.

        :param logx: Use log x axis
        :param logy: Use log y axis
        :param showback: Whether to show background curves
        :param rates: Show rates (cts/s/arcmin2) instead of cts/annulus
        :param title: Veusz window title
        :param mode: "hidden"=hidden window for export, "new"=new window, "update"=update last window
        :param savefilename: Save Veusz document to file
        :param exportfilename: Export Veusz document to graphics file
        """
        if veusz is None:
            raise RuntimeError('Veusz not found')

        update = False
        if mode == 'hidden':
            embed = veusz.Embedded(hidden=True)
        elif mode == 'new' or len(self._veuszembed)==0:
            embed = veusz.Embedded(title)
            self._veuszembed.append(embed)
        elif mode == 'update':
            embed = self._veuszembed[-1]
            update = True
        else:
            raise ValueError("Invalid mode")

        if not update:
            root = embed.Root
            root.colorTheme.val = 'default-latest'
            page = root.Add('page')
            page.height.val = '%gcm' % (len(self.data.bands)*5)
            grid = page.Add('grid', columns=1)
            xaxis = grid.Add(
                'axis', name='x', direction='horizontal',
                autoRange='+2%')
            if logx:
                xaxis.log.val = True

            for i in range(len(self.data.bands)):
                graph = grid.Add('graph', name='graph%i' % i)
                if logy:
                    graph.y.log.val = True

                xydata = graph.Add(
                    'xy', marker='none', name='data',
                    xData='radius', yData='data_%i' % i)
                xymodel = graph.Add(
                    'xy', marker='none', name='model',
                    xData='radius', yData='model_%i' % i)
                if showback:
                    xyback = graph.Add(
                        'xy', marker='none', name='back',
                        xData='radius', yData='back_%i' % i)

            edges = self.data.annuli.edges_arcmin
            embed.SetData(
                'radius', 0.5*(edges[1:]+edges[:-1]),
                symerr=0.5*(edges[1:]-edges[:-1]))
            grid.Action('zeroMargins')

        if 'backscale' in self.pars:
            backscale = self.pars['backscale'].v
        else:
            backscale = 1.

        profs = self.calcProfiles()
        for i, (band, prof) in enumerate(zip(self.data.bands, profs)):
            factor = (
                1/(self.data.annuli.geomarea_arcmin2*band.areascales*band.exposures)
                if rates else 1 )
            embed.SetData('data_%i' % i, band.cts*factor)
            embed.SetData('model_%i' % i, prof*factor)
            embed.SetData(
                'back_%i' % i, band.backrates*self.data.annuli.geomarea_arcmin2*
                band.areascales*band.exposures*factor*backscale)

        if savefilename:
            embed.Save(savefilename)
        if exportfilename:
            embed.Export(exportfilename)

        if mode == 'hidden':
            embed.Close()

    def save(self, filename):
        """Save fit in Python pickle format."""

        # don't try to pickle veusz window
        embed = None
        if self._veuszembed:
            embed = self._veuszembed
            self._veuszembed = None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        if embed:
            self._veuszembed = embed

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
