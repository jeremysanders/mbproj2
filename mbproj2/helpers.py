from __future__ import division, print_function
import numpy as N

import cmpt
import model
import fit

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

def initialNeCmptBinnedFromBeta(annuli, data, NH_1022pcm2=0.01, Z_solar=0.3, T_keV=3.):
    """Return ne component and initial parameters."""

    print('Estimating densities using beta model')

    # initially fit a beta model
    Z_cmpt = cmpt.CmptFlat(
        'Z', annuli, defval=Z_solar, minval=-2., maxval=1.)
    T_cmpt = cmpt.CmptFlat(
        'T', annuli, defval=T_keV, minval=0.01, maxval=50.)
    ne_beta_cmpt = cmpt.CmptBeta('ne', annuli)
    betamodel = model.ModelNullPot(
        annuli, ne_beta_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)

    betapars = betamodel.defPars()
    betapars['Z'].fixed = True

    betafit = fit.Fit(betapars, betamodel, data)
    betafit.doFitting(silent=True)
    like = betafit.doFitting(silent=True)
    print(' Log likelihood (beta): %.1f' % like)

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

    outpars = {par: val for par, val in binnedpars.iteritems() if par[:3] == 'ne_'}
    return ne_binned_cmpt, outpars
