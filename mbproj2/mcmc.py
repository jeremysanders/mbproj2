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

import h5py
import emcee
import numpy as N

from . import utils
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
                    p0, iterations=length, storechain=False)):

                if i % 10 == 0:
                    uprint(' Burn %i / %i (%.1f%%)' % (
                            i, length, i*100/length))

                self.pos0, lnprob, rstate0 = result[:3]

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
