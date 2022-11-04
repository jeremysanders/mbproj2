#!/usr/bin/env python3

# non-hydrostatic version of profile fit assuming parametric
# modified-beta density model and McDonald temperature profile

# I've not really tested whether this temperature model is working
# properly, but it's just an example

# note xspec will run for a while on first invocation to create grids,
# so don't worry if it hangs on Fitting

import sys
import pickle

import numpy as N
import h5py

import mbproj2 as mb

# MCMC parameters
#################
# number to burn
nburn = 1000
# length of chain
nlength = 1000
# number of walkers
nwalkers = 200
# number of processes/threads
nthreads = 4
#################

# energy bands in eV
bandEs = [
    [ 500, 750], [ 750,1000], [1000,1250],
    [1250,1500], [1500,2000], [2000,3000],
    [3000,4000], [4000,5000], [5000,6000],
    [6000,7000]
    ]

# response
rmf = 'centre_merge.rmf'
arf = 'centre_merge.arf'

# filename template (to fill in energies)
infgtempl = 'total_fg_prof_%04i_%04i.dat.rebinopttwo20.s5'
inbgtempl = 'total_bg_prof_%04i_%04i.dat.match.rebinopttwo20.s5'

# name for outputs
name = 'fit_modified_beta_nonhydro'

def loadBand(bandE):
    """Load foreground and background profiles from file and construct
    Band object."""

    data = N.loadtxt(infgtempl % (bandE[0], bandE[1]))

    # radii of centres of annuli in arcmin
    radii = data[:,0]
    # half-width of annuli in arcmin
    hws = data[:,1]
    # number of counts (integer)
    cts = data[:,2]
    # areas of annuli, taking account of pixelization (arcmin^2)
    areas = data[:,3]
    # exposures (s)
    exps = data[:,4]
    # note: vignetting can be input into exposure or areas, but
    # background needs to be consistent

    # geometric area factor
    geomareas = N.pi*((radii+hws)**2-(radii-hws)**2)
    # ratio between real pixel area and geometric area
    areascales = areas/geomareas

    # this is the band object fitted to the data
    band = mb.Band(
        bandE[0]/1000, bandE[1]/1000,
        cts, rmf, arf, exps, areascales=areascales)

    # this is the background profile
    # load rates in cts/s/arcmin^2
    backd = N.loadtxt(inbgtempl % (bandE[0], bandE[1]))
    band.backrates = backd[:,5]

    return band

def getEdges():
    """Get edges of annuli in arcmin.
    There should be one more than the number of annuli.
    """
    data = N.loadtxt(infgtempl % (bandEs[0][0], bandEs[0][1]))
    return N.hstack(( data[0,0]-data[0,1], data[:,0]+data[:,1] ))

def main():
    # cluster parameters
    redshift = 0.7019
    NH_1022pcm2 = 0.0137
    Z_solar = 0.3

    # for calculating distances, etc.
    cosmology = mb.Cosmology(redshift)

    # annuli object contains edges of annuli
    annuli = mb.Annuli(getEdges(), cosmology)

    # load each band, chopping outer radius
    bands = []
    for bandE in bandEs:
        bands.append(loadBand(bandE))

    # Data object represents annuli and bands
    data = mb.Data(bands, annuli)

    # this is the modified beta model density described in Sanders+17
    # (used in McDonald+12) (single mode means only one beta model, as
    # described in Vikhlinin+06)
    ne_cmpt = mb.CmptVikhDensity('ne', annuli, mode='single')
    # this is the parametric temperature model from McDonald+14, eqn 1
    T_cmpt = mb.CmptMcDonaldTemperature('T', annuli)
    # flat metallicity profile
    Z_cmpt = mb.CmptFlat('Z', annuli, defval=Z_solar)

    # nfw mass model
    nfw = mb.CmptMassNFW(annuli)

    # non-hydrostatic model combining density, temperature and metallicity
    model = mb.ModelNullPot(
        annuli, ne_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)

    # get default parameters
    pars = model.defPars()

    # add parameter which allows variation of background with a
    # Gaussian prior with sigma=0.1
    pars['backscale'] = mb.ParamGaussian(1., prior_mu=1, prior_sigma=0.1)

    # freeze metallicity at fixed value
    pars['Z'].frozen = True
   
    # stop radii going beyond edge of data
    pars['ne_logrc_1'].maxval = annuli.edges_logkpc[-2]
    pars['ne_logr_s'].maxval = annuli.edges_logkpc[-2]

    # some ranges of parameters to allow for the density model
    pars['ne_gamma'].val = 3.
    pars['ne_gamma'].frozen = True
    pars['ne_logrc_1'].val = 2.
    pars['ne_alpha'].val = 0.1
    pars['ne_alpha'].maxval = 4
    pars['ne_alpha'].minval = 0.
    pars['ne_epsilon'].maxval = 10

    # do fitting of data with model
    fit = mb.Fit(pars, model, data)
    # refreshThawed is required if frozen is changed after Fit is
    # constructed before doFitting (it's not required here)
    fit.refreshThawed()
    fit.doFitting()

    # construct MCMC object and do burn in
    mcmc = mb.MCMC(fit, walkers=nwalkers, processes=nthreads)
    mcmc.burnIn(nburn)

    # save best fit
    with open('%s_fit.pickle' % name, 'wb') as f:
        pickle.dump(fit, f, -1)

    # run mcmc proper and save to an output chain file
    # (note chain parameters are stored alphabetically - see hdf_view)
    chainfilename = '%s_chain.hdf5' % name
    mcmc.run(nlength)
    mcmc.save(chainfilename)

    # construct a set of physical median profiles from the chain and
    # save
    profs = mb.replayChainPhys(
        chainfilename, model, pars, thin=10, confint=68.269)
    mb.savePhysProfilesHDF5('%s_medians.hdf5' % name, profs)
    mb.savePhysProfilesText('%s_medians.txt' % name, profs)

if __name__ == '__main__':
    main()
