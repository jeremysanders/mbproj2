from __future__ import division, print_function

import sys
import os
import numpy as N

import mbproj2 as mb

indir='../jsproj/examples/pks0745'

cosmology = mb.Cosmology(0.1028)
annuli = mb.loadAnnuli(
    os.path.join(indir, 'sb_profile_1200_2500.dat.rebin'), cosmology)

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

nfw = mb.CmptMassNFW(annuli)
ne_cmpt = mb.CmptBinned('ne', annuli, log=True, defval=-2, minval=-6., maxval=1.)
Z_cmpt = mb.CmptFlat('Z', annuli, defval=N.log10(0.3), log=True, minval=-2., maxval=1.)
model = mb.ModelHydro(annuli, nfw, ne_cmpt, Z_cmpt, NH_1022pcm2=0.378)
pars = model.defPars()

pars['Z'].val = N.log10(0.4)
pars['Z'].frozen = True

mb.estimateDensityProfile(model, data, pars)

fit = mb.Fit(pars, model, data)
fit.doFitting()

mcmc = mb.MCMC(fit, walkers=200, processes=4)
mcmc.burnIn(1000)
mcmc.run(1000)
mcmc.save('chain.hdf5')

profs = mb.replayChainPhys('chain.hdf5', model, pars)
mb.savePhysProfilesHDF5('medians.hdf5', profs)
