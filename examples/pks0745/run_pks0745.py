from __future__ import division, print_function

import sys
import os
import numpy as N

import mbproj2

def main():
    indir = '.'

    cosmology = mbproj2.Cosmology(0.1028)

    # define annuli using 1st two columns in this file
    annuli = mbproj2.loadAnnuli(
        os.path.join(indir, 'sb_profile_1200_2500.dat.rebin'), cosmology)

    # load count profiles for each band
    bands = []
    for emin, emax in ((0.5, 1.2), (1.2, 2.5), (2.5, 6.0)):
        band =  mbproj2.loadBand(
            os.path.join(indir, 'sb_profile_%i_%i.dat.rebin' % (
                emin*1000, emax*1000)),
            emin, emax,
            os.path.join(indir, 'ann_0_total_rmf.fits'),
            os.path.join(indir, 'ann_0_total_arf.fits'))
        band.backrates = N.loadtxt(
            os.path.join(indir, 'sb_bgcomb_%i_%i.dat.rebin.norm' % (
                emin*1000, emax*1000)))[:,2]
        bands.append(band)

    data = mbproj2.Data(bands, annuli)

    # this is the mass component to use
    nfw = mbproj2.CmptMassNFW(annuli)
    # density component
    ne_cmpt = mbproj2.CmptBinned(
        'ne', annuli, log=True, defval=-2, minval=-6., maxval=1.)
    # metallicity component
    Z_cmpt = mbproj2.CmptFlat(
        'Z', annuli, defval=N.log10(0.3), log=True, minval=-2., maxval=1.)

    # combine components to create model
    model = mbproj2.ModelHydro(
        annuli, nfw, ne_cmpt, Z_cmpt, NH_1022pcm2=0.378)

    # get default parameters
    pars = model.defPars()

    # example change in parameter
    pars['Z'].val = N.log10(0.4)
    pars['Z'].frozen = True

    # estimate densities from surface brightness to avoid fits
    mbproj2.estimateDensityProfile(model, data, pars)

    # fit parameters to data
    fit = mbproj2.Fit(pars, model, data)
    fit.doFitting()

    # now start MCMC using 4 processes
    mcmc = mbproj2.MCMC(fit, walkers=200, processes=4)
    mcmc.burnIn(1000)
    mcmc.run(1000)
    mcmc.save('chain.hdf5')

    # compute physical profiles and save as medians
    profs = mbproj2.replayChainPhys('chain.hdf5', model, pars)
    mbproj2.savePhysProfilesHDF5('medians.hdf5', profs)

if __name__ == '__main__':
    main()
