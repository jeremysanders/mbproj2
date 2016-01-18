from __future__ import division, print_function
import os

import h5py
import emcee
import numpy as N

import utils
import forkparallel

class MultiProcessPool:
    """Use ForkQueue to evaluate multiple profiles simultaneously."""
    def __init__(self, func, processes):
        self.queue = forkparallel.ForkQueue(func, processes)

    def map(self, func, args):
        """Note func is ignored here."""
        return self.queue.execute(args)

class MCMC:
    """For running MCMC."""
    def __init__(self, outfilename, fit, walkers=100, processes=1):

        self.fit = fit
        self.walkers = walkers
        self.numpars = len(fit.thawed)

        # function for getting likelihood
        likefunc = lambda pars: fit.getLikelihood(pars)

        # pool object for returning result for multiple processes
        pool = (
            None if processes <= 1 else MultiProcessPool(likefunc, processes))

        # for doing the mcmc sampling
        self.sampler = emcee.EnsembleSampler(
            walkers, len(fit.thawed), likefunc, pool=pool)
        # starting point
        self.pos0 = None

        # the output file
        if os.path.exists(outfilename):
            raise RuntimeError('Output file %s already exists' % outfilename)
        self.outfile = h5py.File(outfilename)
        self.initFile()

    def initFile(self):
        """Write headers to output file."""

        outfile = self.outfile

        outfile.attrs['walkers'] = self.walkers
        outfile.attrs['length'] = 0
        outfile.attrs['thin'] = -1
        outfile.attrs['burn'] = 0
        outfile['thawed_params'] = self.fit.thawed

    def generateInitPars(self):
        """Generate initial set of parameters from fit."""

        p0 = []
        while len(p0) < self.walkers:
            p = N.random.normal(
                1, 0.001, size=self.numpars)*self.fit.thawedParVals()
            if N.isfinite(self.fit.getLikelihood(p)):
                p0.append(p)
        return p0

    def innerburnin(self, length, autorefit):
        """Burn in chain.

        Returns False if new minimum found
        """

        bestfit = None
        bestprob = initprob = self.fit.getLikelihood(self.fit.thawedParVals())
        p0 = self.generateInitPars()

        # record period
        self.outfile.attrs['burn'] = length

        # iterate over burn-in period
        for i, result in enumerate(self.sampler.sample(
                p0, iterations=length, storechain=False)):

            if i % 10 == 0:
                print(' Burn %i / %i (%.1f%%)' % (
                        i, length, i*100/length))

            self.pos0, lnprob, rstate0 = result[:3]

            # new better fit
            if lnprob.max()-bestprob > 0.01:
                bestprob = lnprob.max()
                maxidx = lnprob.argmax()
                bestfit = self.pos0[maxidx]

            # abort if new minimum found
            if ( autorefit and i>length*0.2 and
                 bestfit is not None ):

                print('  Restarting burn as new best fit has been found '
                      ' (%g > %g)' % (bestprob, initprob) )
                self.fit.updateThawed(bestfit)
                self.sampler.reset()
                return False

        self.sampler.reset()
        return True

    def burnIn(self, length, autorefit=True):
        """Burn in, restarting if necessary."""

        print('Burning in')
        while not self.innerburnin(length, autorefit):
            print('Restarting, as new mininimum found')
            self.fit.doFitting()

    def run(self, length=1000, thin=1):
        """Run main chain."""

        # create output

        if self.pos0 is None:
            print('Generating initial parameters')
            p0 = self.generateInitPars()
        else:
            print('Using burn in parameters')
            p0 = self.pos0

        # start up
        print('Sampling')

        for i, result in enumerate(self.sampler.sample(
                p0, iterations=length)):

            if i % 10 == 0:
                print(' Step %i / %i (%.1f%%)' % (i, length, i*100/length))

        # write output
        self.outfile.attrs['length'] = length
        self.outfile.attrs['thin'] = thin
        self.outfile.create_dataset(
            'chain',
            data=self.sampler.chain[:, ::thin, :].astype(N.float32),
            compression=True, shuffle=True)
        self.outfile.create_dataset(
            'prob',
            data=self.sampler.lnprobability[:, ::thin].astype(N.float32),
            compression=True, shuffle=True)
        self.outfile['acceptfrac'] = self.sampler.acceptance_fraction.astype(N.float32)
        self.outfile['lastpos'] = self.sampler.chain[:, -1, :].astype(N.float32)

        print('Done!')
