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

"""For doing MCMC given Fit object."""

from __future__ import division, print_function, absolute_import

import os
import six
import h5py
import emcee
import numpy as N

from . import utils
from . import fit
from .utils import uprint
from . import forkparallel

class _MultiProcessPool:
    """Internal object to use ForkQueue to evaluate multiple profiles
    simultaneously."""

    def __init__(self, func, processes):
        self.queue = forkparallel.ForkQueue(func, processes)

    def map(self, func, args):
        """Note func is ignored here."""
        return self.queue.execute(args)

class MCMC:
    """For running Markov Chain Monte Carlo."""

    def __init__(self, fit,
                 walkers=100, processes=1, initspread=0.01):
        """
        :param Fit fit: Fit object to use for mcmc
        :param int walkers: number of emcee walkers to use
        :param int processes: number of simultaneous processes to compute likelihoods
        :param float initspread: random Gaussian width added to create initial parameters
        """

        self.fit = fit
        self.walkers = walkers
        self.numpars = len(fit.thawed)
        self.initspread = initspread

        # function for getting likelihood
        likefunc = lambda pars: fit.getLikelihood(pars)

        # pool object for returning result for multiple processes
        pool = (
            None if processes <= 1 else _MultiProcessPool(likefunc, processes))

        # for doing the mcmc sampling
        self.sampler = emcee.EnsembleSampler(
            walkers, len(fit.thawed), likefunc, pool=pool)
        # starting point
        self.pos0 = None

        # header items to write to output file
        self.header = {
            'burn': 0,
            }

    def _generateInitPars(self):
        """Generate initial set of parameters from fit."""

        thawedpars = N.array(self.fit.thawedParVals())
        assert N.all(N.isfinite(thawedpars))

        # create enough parameters with finite likelihoods
        p0 = []
        while len(p0) < self.walkers:
            p = N.random.normal(0, self.initspread, size=self.numpars) + thawedpars
            if N.isfinite(self.fit.getLikelihood(p)):
                p0.append(p)
        return p0

    def burnIn(self, length, autorefit=True, minfrac=0.2, minimprove=0.01):
        """Burn in, restarting fit and burn if necessary.

        :param bool autorefit: refit position if new minimum is found during burn in
        :param float minfrac: minimum fraction of burn in to do if new minimum found
        :param float minimprove: minimum improvement in fit statistic to do a new fit
        """

        def innerburn():
            """Returns False if new minimum found and autorefit is set"""

            bestfit = None
            bestprob = initprob = self.fit.getLikelihood(self.fit.thawedParVals())
            p0 = self._generateInitPars()

            # record period
            self.header['burn'] = length

            # iterate over burn-in period
            for i, result in enumerate(self.sampler.sample(
                    p0, iterations=length, store=False)):

                if i % 10 == 0:
                    uprint(' Burn %i / %i (%.1f%%)' % (
                            i, length, i*100/length))

                self.pos0 = result.coords
                lnprob = result.log_prob

                # new better fit
                if lnprob.max()-bestprob > minimprove:
                    bestprob = lnprob.max()
                    maxidx = lnprob.argmax()
                    bestfit = self.pos0[maxidx]

                # abort if new minimum found
                if ( autorefit and i>length*minfrac and
                     bestfit is not None ):

                    uprint('  Restarting burn as new best fit has been found '
                           ' (%g > %g)' % (bestprob, initprob) )
                    self.fit.updateThawed(bestfit)
                    self.sampler.reset()
                    return False

            self.sampler.reset()
            return True

        uprint('Burning in')
        while not innerburn():
            uprint('Restarting, as new mininimum found')
            self.fit.doFitting()

    def run(self, length):
        """Run chain.

        :param int length: length of chain
        """

        uprint('Sampling')
        self.header['length'] = length

        # initial parameters
        if self.pos0 is None:
            uprint(' Generating initial parameters')
            p0 = self._generateInitPars()
        else:
            uprint(' Starting from end of burn-in position')
            p0 = self.pos0

        # do sampling
        for i, result in enumerate(self.sampler.sample(
                p0, iterations=length)):

            if i % 10 == 0:
                uprint(' Step %i / %i (%.1f%%)' % (i, length, i*100/length))

        uprint('Done')

    def save(self, outfilename, thin=1):
        """Save chain to HDF5 file.

        :param str outfilename: output hdf5 filename
        :param int thin: save every N samples from chain
        """

        self.header['thin'] = thin

        uprint('Saving chain to', outfilename)
        with h5py.File(outfilename, 'w') as f:
            # write header entries
            for h in sorted(self.header):
                f.attrs[h] = self.header[h]

            # write list of parameters which are thawed
            f['thawed_params'] = [x.encode('utf-8') for x in self.fit.thawed]

            # output chain
            f.create_dataset(
                'chain',
                data=self.sampler.chain[:, ::thin, :].astype(N.float32),
                compression=True, shuffle=True)
            # likelihoods for each walker, iteration
            f.create_dataset(
                'likelihood',
                data=self.sampler.lnprobability[:, ::thin].astype(N.float32),
                compression=True, shuffle=True)
            # acceptance fraction
            f['acceptfrac'] = self.sampler.acceptance_fraction.astype(N.float32)
            # last position in chain
            f['lastpos'] = self.sampler.chain[:, -1, :].astype(N.float32)

        uprint('Done')

def replayChainSB(
        chainfilename, data, model, pars, burn=0, thin=10,
        confint=68.269, randsample=False):

    """Replay chain, computing surface brightness files

    :param chainfilename: input physical chain filename
    :param Data data: input data
    :param Model model: input model
    :type pars: dict[str, ParamBase]
    :param pars: parameters used in model
    :param confint: total confidence interval (percentage)
    :param burn: skip initial N items in chain
    :param thin: skip every N iterations in chain
    :param randsample: randomly sample chain when thinning

    :returns: medians and confidence interval percentiles
    """

    uprint('Computing surface brightness profiles from chain', chainfilename)
    with h5py.File(chainfilename, 'r') as f:
        fakefit = fit.Fit(pars, model, None)
        filethawed = [x.decode('utf-8') for x in f['thawed_params']]
        if fakefit.thawed != filethawed:
            raise RuntimeError('Parameters do not match those in chain')

        if randsample:
            #print('Geting random sample')
            chain = f['chain'][:, burn:, :]
            chain = chain.reshape(-1, chain.shape[2])
            rows = N.arange(chain.shape[0])
            N.random.shuffle(rows)
            chain = chain[rows[:len(rows)//thin], :]
        else:
            chain = f['chain'][:, burn::thin, :]
            chain = chain.reshape(-1, chain.shape[2])

    # iterate over input
    length = len(chain)
    annuli = data.annuli

    # keep copy of profiles
    totprofs = [[] for i in range(len(data.bands))]
    clustprofs = [[] for i in range(len(data.bands))]
    backprofs = [[] for i in range(len(data.bands))]

    for i, parvals in enumerate(chain):
        if i % 1000 == 0:
            uprint(' Step %i / %i (%.1f%%)' % (i, length, i*100/length))

        fakefit.updateThawed(parvals)

        # optional background scaling parameter
        if 'backscale' in fakefit.pars:
            backscale = fakefit.pars['backscale'].val
        else:
            backscale = 1.

        ne_prof, T_prof, Z_prof = model.computeProfs(fakefit.pars)
        for i, band in enumerate(data.bands):
            clustprof, backprof = band.calcProjProfileCmpts(
                annuli, ne_prof, T_prof, Z_prof,
                model.NH_1022pcm2,
                backscale=backscale)

            # convert to rates / s / arcmin2
            scale = 1/(annuli.geomarea_arcmin2 * band.areascales * band.exposures)
            clustprof *= scale
            backprof *= scale

            totprofs[i].append(clustprof+backprof)
            clustprofs[i].append(clustprof)
            backprofs[i].append(backprof)

    def getrange(profs):
        median, posrange, negrange = N.percentile(
            profs, [50, 50+confint/2, 50-confint/2], axis=0)
        return N.column_stack((median, posrange-median, negrange-median))

    # compute medians and errors
    uprint(' Computing medians')
    outprofs = {}
    for i in range(len(data.bands)):
        outprofs['band%i_tot' % i] = getrange(totprofs[i])
        outprofs['band%i_clust' % i] = getrange(clustprofs[i])
        outprofs['band%i_back' % i] = getrange(backprofs[i])

    # write radii
    rmid = annuli.midpt_kpc
    rpos = annuli.edges_kpc[1:] - rmid
    rneg = annuli.edges_kpc[:-1] - rmid
    outprofs['radius_kpc'] = N.column_stack((rmid, rpos, rneg))

    rmid = 0.5*(annuli.edges_arcmin[1:]+annuli.edges_arcmin[:-1])
    rpos = annuli.edges_arcmin[1:] - rmid
    rneg = annuli.edges_arcmin[:-1] - rmid
    outprofs['radius_arcmin'] = N.column_stack((rmid, rpos, rneg))

    # write counts and rate
    for i, band in enumerate(data.bands):
        cts = band.cts
        perr = 1 + N.sqrt(band.cts + 0.75)
        nerr = -N.sqrt(N.clip(band.cts-0.25, 0, None))
        scale = 1/(annuli.geomarea_arcmin2 * band.areascales * band.exposures)

        outprofs['band%i_cts' % i] = N.column_stack((cts, perr, nerr))
        outprofs['band%i_rate' % i] = N.column_stack((cts*scale, perr*scale, nerr*scale))

    uprint('Done median computation')
    return outprofs

def saveSBProfilesHDF5(outfilename, profiles):
    """Given median profiles from replayChainSB, save output profiles to
    hdf5.
    """
    try:
        os.unlink(outfilename)
    except OSError:
        pass
    uprint('Writing', outfilename)
    with h5py.File(outfilename, 'w') as f:
        for name in profiles:
            f[name] = profiles[name]
            f[name].attrs['vsz_twod_as_oned'] = 1
