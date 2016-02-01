from __future__ import division, print_function
import numpy as N

import cmpt
import model
import fit
from physconstants import kpc_cm

def estimateDensityProfile(inmodel, data, modelpars):
    """Do a simplified fit to get the initial density profile.

    This assumes an isothermal cluster with constant 0.3 solar
    metallicity and does not use hydrostatic equilibrium.
    """

    print('Estimating densities')

    annuli = inmodel.annuli
    # temporary model with flat temperature and metallicity (fixed)
    tempmodel = model.ModelNullPot(
        annuli,
        inmodel.ne_cmpt,
        cmpt.CmptFlat('T', annuli, defval=3., minval=0.1, maxval=12),
        cmpt.CmptFlat('Z', annuli, defval=0.3, log=False),
        NH_1022pcm2=inmodel.NH_1022pcm2)

    temppars = tempmodel.defPars()
    temppars['Z'].frozen = True
    tempfit = fit.Fit(temppars, tempmodel, data)

    tempfit.doFitting()

    # update parameters in original model
    txt = []
    for par in sorted(temppars):
        if par[:2] == 'ne':
            modelpars[par] = temppars[par]
            txt.append('%4.1f' % temppars[par].val)

    print('Done estimating densities:', ' '.join(txt))

def fitBeta(annuli, data, NH_1022pcm2, Z_solar, T_keV):
    """Fit beta density model with isothermal cluster.

    Return ne_cmpt, T_cmpt, Z_cmpt, pars
    """

    ne_beta_cmpt = cmpt.CmptBeta('ne', annuli)
    T_cmpt = cmpt.CmptFlat(
        'T', annuli, defval=T_keV, minval=0.01, maxval=50.)
    Z_cmpt = cmpt.CmptFlat(
        'Z', annuli, defval=Z_solar, minval=-2., maxval=1.)
    betamodel = model.ModelNullPot(
        annuli, ne_beta_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)

    betapars = betamodel.defPars()
    betapars['Z'].fixed = True

    betafit = fit.Fit(betapars, betamodel, data)
    betafit.doFitting(silent=True)
    like = betafit.doFitting(silent=True)
    print(' Log likelihood (beta): %.1f' % like)

    return ne_beta_cmpt, T_cmpt, Z_cmpt, betapars

def initialNeCmptBinnedFromBeta(
    annuli, data, NH_1022pcm2=0.01, Z_solar=0.3, T_keV=3.):
    """Return ne component and initial parameters."""

    print('Estimating densities using beta model')

    ne_beta_cmpt, T_cmpt, Z_cmpt, betapars = fitBeta(
        annuli, data, NH_1022pcm2, Z_solar, T_keV)

    # then switch to a binned profile
    ne_binned_cmpt = cmpt.CmptBinned(
        'ne', annuli, log=True, defval=-2, minval=-6., maxval=1.)
    binnedmodel = model.ModelNullPot(
        annuli, ne_binned_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)
    binnedpars = binnedmodel.defPars()
    binnedpars['Z'].fixed = True
    binnedpars['T'].val = betapars['T'].val

    # copy in beta profile
    ne_prof = N.log10(ne_beta_cmpt.computeProf(betapars))
    for i, v in enumerate(ne_prof):
        binnedpars['ne_%03i' % i].val = v

    binnedfit = fit.Fit(binnedpars, binnedmodel, data)
    binnedfit.doFitting(silent=True)
    like = binnedfit.doFitting(silent=True)
    print(' Log likelihood (full): %.1f' % like)

    print('Estimated profile:', N.log10(ne_binned_cmpt.computeProf(binnedpars)))

    outpars = {par: val for par, val in binnedpars.iteritems()
               if par[:3] == 'ne_'}
    return ne_binned_cmpt, outpars

def initialNeCmptInterpolMoveRadFromBeta(
    annuli, data, nradbins, NH_1022pcm2=0.01, Z_solar=0.3, T_keV=3.):
    """Create a density profile with the possibility to move
    interpolated bin radii based on an isothermal beta model initial
    fit.
    """

    print('Estimating densities using beta model')

    ne_beta_cmpt, T_cmpt, Z_cmpt, betapars = fitBeta(
        annuli, data, NH_1022pcm2, Z_solar, T_keV)

    print('Switching to interpolation model')
    ne_moving_cmpt = cmpt.CmptInterpolMoveRad(
        'ne', annuli, defval=-3., minval=-6, maxval=1.,
        log=True, nradbins=nradbins)

    movingmodel = model.ModelNullPot(
        annuli, ne_moving_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)

    # setup parameters
    movingpars = movingmodel.defPars()
    movingpars['Z'].fixed = True
    movingpars['T'].val = betapars['T'].val

    # create radial bins and calculate densities at points from beta
    # model
    rlogannuli = N.log10(annuli.midpt_cm / kpc_cm)
    rlog = N.linspace(rlogannuli[0], rlogannuli[-1], nradbins)
    betane = ne_beta_cmpt.computeProf(betapars)
    nepars = N.interp(rlog, rlogannuli, betane)

    # update parameters
    for i in xrange(nradbins):
        movingpars['ne_%03i' % i].val = nepars[i]
        movingpars['ne_r_%03i' % i].val = rlog[i]

    # do fitting of new model
    movingfit = fit.Fit(movingpars, movingmodel, data)
    movingfit.doFitting(silent=True)
    like = movingfit.doFitting(silent=True)
    print(' Log likelihood (full): %.1f' % like)

    print('Estimated profile:', N.log10(ne_moving_cmpt.computeProf(movingpars)))

    outpars = {par: val for par, val in movingpars.iteritems()
               if par[:3] == 'ne_'}
    return ne_moving_cmpt, outpars
