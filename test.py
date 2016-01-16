from __future__ import division, print_function

import os
import numpy as N
import scipy.optimize
import emcee
import h5py

import mbproj2 as mb

indir='../jsproj/examples/pks0745'

cosmology = mb.Cosmology(0.1028)
annuli = mb.loadAnnuli(
    os.path.join(indir, 'sb_profile_1200_2500.dat.rebin'), cosmology)

nfw = mb.CmptMassNFW(annuli)

ne_cmpt = mb.CmptBinned('ne', annuli, log=True, defval=-2)
Z_cmpt = mb.CmptFlat('Z', annuli, defval=0.3, log=True)

model = mb.ModelHydro(annuli, nfw, ne_cmpt, Z_cmpt, NH_1022pcm2=0.378)
pars = model.defPars()

# pars['Z'].frozen = True

band1 = mb.loadBand(
    os.path.join(indir, 'sb_profile_500_1200.dat.rebin'),
    0.5, 1.2,
    os.path.join(indir, 'ann_0_total_rmf.fits'),
    os.path.join(indir, 'ann_0_total_arf.fits'))
band1.backrates = N.loadtxt(
    os.path.join(indir, 'sb_bgcomb_500_1200.dat.rebin.norm'))[:,2]

band2 = mb.loadBand(
    os.path.join(indir, 'sb_profile_1200_2500.dat.rebin'),
    1.2, 2.5,
    os.path.join(indir, 'ann_0_total_rmf.fits'),
    os.path.join(indir, 'ann_0_total_arf.fits'))
band2.backrates = N.loadtxt(
    os.path.join(indir, 'sb_bgcomb_1200_2500.dat.rebin.norm'))[:,2]

band3 = mb.loadBand(
    os.path.join(indir, 'sb_profile_2500_6000.dat.rebin'),
    2.5, 6.0,
    os.path.join(indir, 'ann_0_total_rmf.fits'),
    os.path.join(indir, 'ann_0_total_arf.fits'))
band3.backrates = N.loadtxt(
    os.path.join(indir, 'sb_bgcomb_2500_6000.dat.rebin.norm'))[:,2]

data = mb.Data([band1, band2, band3], annuli)

fit = mb.Fit(pars, model, data)
fit.doFitting()


ndim = len(fit.thawed)
nwalkers = 200

bestfit = N.array(fit.thawedParVals())
p0 = []
for i in xrange(nwalkers):
    p0.append(N.random.normal(loc=bestfit, scale=N.abs(bestfit)*1e-3))

import mbproj2.forkparallel
class Pool:
    def __init__(self, func, instances):
        self.queue = mbproj2.forkparallel.ForkQueue(func, instances)

    def map(self, func, parlist):
        """func is ignored here."""
        results = self.queue.execute(parlist)
        return results

func = lambda par: fit.getLikelihood(par)
pool = Pool(func, 4)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, func, pool=pool)

pos, prob, state = sampler.run_mcmc(p0, 1000)
sampler.reset()

sampler.run_mcmc(pos, 1000)

with h5py.File('test.h5') as f:
    f['chain'] = sampler.flatchain

print(sorted(fit.thawed))



# timing: queue, 2 instances, no batch: 3m22.088s
# timing: queue, 2 instances, batch in half: 2m41.44s
# timing: queue, 4 instances, batch in 1/4: 2m24.45s
# timing: direct: 3m34s
