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

"""Various useful functions for the user.
"""

from __future__ import division, print_function, absolute_import
import numpy as N

from . import cmpt
from . import model
from . import fit
from . import utils
from .utils import uprint
from .physconstants import kpc_cm

def estimateDensityProfile(inmodel, data, modelpars):
    """Do a simplified fit to get the initial density profile.

    This assumes an isothermal cluster with constant 0.3 solar
    metallicity and does not use hydrostatic equilibrium.
    """

    uprint('Estimating densities')

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

    uprint('Done estimating densities:', ' '.join(txt))

def fitBeta(annuli, data, NH_1022pcm2, Z_solar, T_keV, silent=True):
    """Fit beta density model with isothermal cluster.

    Return ne_cmpt, T_cmpt, Z_cmpt, pars
    """

    ne_beta_cmpt = cmpt.CmptBeta('ne', annuli)
    T_cmpt = cmpt.CmptFlat(
        'T', annuli, defval=T_keV, minval=0.1, maxval=50.)
    Z_cmpt = cmpt.CmptFlat(
        'Z', annuli, defval=Z_solar, minval=-2., maxval=1.)
    betamodel = model.ModelNullPot(
        annuli, ne_beta_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)

    betapars = betamodel.defPars()
    betapars['Z'].fixed = True

    betafit = fit.Fit(betapars, betamodel, data)
    betafit.doFitting(silent=silent)
    like = betafit.doFitting(silent=silent)
    uprint(' Log likelihood (beta): %.1f' % like)

    return ne_beta_cmpt, T_cmpt, Z_cmpt, betapars

def initialNeCmptBinnedFromBeta(
    annuli, data, NH_1022pcm2=0.01, Z_solar=0.3, T_keV=3.):
    """Return ne component and initial parameters."""

    uprint('Estimating densities using beta model')

    ne_beta_cmpt, T_cmpt, Z_cmpt, betapars = fitBeta(
        annuli, data, NH_1022pcm2, Z_solar, T_keV)

    # then switch to a binned profile
    ne_binned_cmpt = cmpt.CmptBinnedJumpPrior(
        'ne', annuli, log=True, defval=-2, minval=-6., maxval=1.)
    binnedmodel = model.ModelNullPot(
        annuli, ne_binned_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)
    binnedpars = binnedmodel.defPars()
    binnedpars['Z'].frozen = True
    binnedpars['T'].val = betapars['T'].val

    # copy in beta profile
    ne_prof = N.log10(ne_beta_cmpt.computeProf(betapars))
    for i, v in enumerate(ne_prof):
        binnedpars['ne_%03i' % i].val = v

    binnedfit = fit.Fit(binnedpars, binnedmodel, data)
    binnedfit.doFitting(silent=True)
    like = binnedfit.doFitting(silent=True)
    uprint(' Log likelihood (full): %.1f' % like)

    uprint('Estimated profile:', N.log10(ne_binned_cmpt.computeProf(binnedpars)))

    outpars = {par: val for par, val in binnedpars.items()
               if par[:3] == 'ne_'}
    return ne_binned_cmpt, outpars

def autoRadialBins(annuli, data, minsn, minbins=2, maxbins=100):
    """Take radial count profiles and choose bins using number of
    projected counts."""

    ninbins = len(annuli.massav_cm)

    lastchange = 0
    while True:
        radii = [annuli.massav_cm[0]]
        lastidx = 0
        idx = 1
        while idx < ninbins:
            fgcts = N.sum([b.cts[lastidx:idx] for b in data.bands])
            bgcts = N.sum([
                    (b.backrates*b.exposures*annuli.geomarea_arcmin2*b.areascales)
                    [lastidx:idx] for b in data.bands])

            sn = (fgcts-bgcts) / utils.gehrels(fgcts)
            if sn > minsn:
                radii.append(annuli.massav_cm[idx])
                lastidx = idx

            idx += 1

        if radii[-1] < annuli.massav_cm[-1]:
            radii.append(annuli.massav_cm[-1])

        nbins = len(radii)-1
        if nbins < minbins:
            if lastchange > 0:
                raise ValueError('Loop detected in S/N binning')
            lastchange = -1
            minsn /= 1.1
        elif nbins > maxbins:
            if lastchange < 0:
                raise ValueError('Loop detected in S/N binning')
            lastchange = 1
            minsn *= 1.1
        else:
            radii_log_kpc = N.log10(N.array(radii) / kpc_cm)
            uprint('Chosen radial interpolation points using S/N %.1f' % minsn)
            uprint('Radii:', radii_log_kpc)
            return radii_log_kpc

def initialNeCmptInterpolMoveRadFromBeta(
    annuli, data, mode, NH_1022pcm2=0.01, Z_solar=0.3, T_keV=3.,
        nradbins=10, minsn=30, minbins=2, maxbins=100, silent=True):
    """Create a density profile with the possibility to move
    interpolated bin radii based on an isothermal beta model initial
    fit.

    returns: density component, default density parameters

    mode should be: 'lognbins', 'minsn'
    """

    uprint('Estimating densities using beta model')

    ne_beta_cmpt, T_cmpt, Z_cmpt, betapars = fitBeta(
        annuli, data, NH_1022pcm2, Z_solar, T_keV, silent=silent)

    uprint('Switching to interpolation model')

    # create radial bins
    rlogannuli = N.log10(annuli.midpt_cm / kpc_cm)
    betane = N.log10(ne_beta_cmpt.computeProf(betapars))

    if mode == 'lognbins':
        rlog = N.linspace(rlogannuli[0], rlogannuli[-1], nradbins)
        nbins = nradbins
    elif mode == 'minsn':
        rlog = autoRadialBins(annuli, data, minsn, minbins=minbins, maxbins=maxbins)
        nbins = len(rlog)
    else:
        raise ValueError('Invalid mode')

    ne_moving_cmpt = cmpt.CmptInterpolMoveRad(
        'ne', annuli, defval=-3., minval=-6, maxval=1.,
        log=True, nradbins=nbins)

    movingmodel = model.ModelNullPot(
        annuli, ne_moving_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)

    # setup parameters
    movingpars = movingmodel.defPars()
    movingpars['Z'].frozen = True
    movingpars['T'].val = betapars['T'].val
    # calculate densities at bins
    nepars = N.interp(rlog, rlogannuli, betane)

    # update parameters
    for i in range(nbins):
        movingpars['ne_%03i' % i].val = nepars[i]
        movingpars['ne_r_%03i' % i].val = rlog[i]

    # do fitting of new model
    movingfit = fit.Fit(movingpars, movingmodel, data)
    movingfit.doFitting(silent=silent)
    like = movingfit.doFitting(silent=silent)
    uprint(' Log likelihood (full): %.1f' % like)

    uprint('Estimated profile:', N.log10(ne_moving_cmpt.computeProf(movingpars)))

    outpars = {par: val for par, val in movingpars.items()
               if par[:3] == 'ne_'}
    return ne_moving_cmpt, outpars
