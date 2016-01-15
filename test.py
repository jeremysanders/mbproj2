from __future__ import division, print_function

import os
import numpy as N

import mbproj2 as MB

indir='../jsproj/examples/pks0745'

cosmology = MB.Cosmology(0.1028)
annuli = MB.loadAnnuli(
    os.path.join(indir, 'sb_profile_1200_2500.dat.rebin'), cosmology,
    0.378)

nfw = MB.CmptMassNFW(annuli)

ne_cmpt = MB.CmptBinned('ne', annuli, binning=5, log=True, defval=-2)
Z_cmpt = MB.CmptFlat('Z', annuli, defval=0.3, log=True)

m2 = MB.ModelHydro(annuli, nfw, ne_cmpt, Z_cmpt)
pars = m2.defPars()

profs = m2.computeProfs(pars)

band1 = MB.loadBand(
    os.path.join(indir, 'sb_profile_500_1200.dat.rebin'),
    0.5, 1.2,
    'ann_0_total_rmf.fits',
    'ann_0_total_arf.fits')
band1.backrates = N.loadtxt(
    os.path.join(indir, 'sb_bgcomb_500_1200.dat.rebin.norm'))[:,2]

band2 = MB.loadBand(
    os.path.join(indir, 'sb_profile_1200_2500.dat.rebin'),
    1.2, 2.5,
    'ann_0_total_rmf.fits',
    'ann_0_total_arf.fits')
band1.backrates = N.loadtxt(
    os.path.join(indir, 'sb_bgcomb_1200_2500.dat.rebin.norm'))[:,2]

band3 = MB.loadBand(
    os.path.join(indir, 'sb_profile_2500_6000.dat.rebin'),
    2.5, 6.0,
    'ann_0_total_rmf.fits',
    'ann_0_total_arf.fits')
band1.backrates = N.loadtxt(
    os.path.join(indir, 'sb_bgcomb_2500_6000.dat.rebin.norm'))[:,2]

