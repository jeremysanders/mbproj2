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

"""Components which make up a profile. Each component has a set of
parameters (of type ParamXXX).

"""

from __future__ import division, print_function, absolute_import
import math

import numpy as N
from scipy.special import hyp2f1

from .param import Param
from .physconstants import kpc_cm

class Cmpt:
    """Parametrize a profile."""

    def __init__(self, name, annuli):
        """
        :param name: prepended to each model parameter name
        :param Annuli annuli: annuli to analyse with component
        """
        self.name = name
        self.annuli = annuli

    def defPars(self):
        """
        :rtype: dict[str,Param]
        :return: default dict of parameters (names to Param objects).
        """

    def computeProf(self, pars):
        """Compute profile given model parameters.

        :type pars: dict[str, Param]
        :param pars: parameter values
        :returns: output profile
        """
        pass

    def prior(self, pars):
        """Given parameters, compute prior.

        :type pars: dict[str, Param]
        :param pars: parameter values
        :returns: log prior
        """
        return 0.

class CmptFlat(Cmpt):
    """A flat profile."""

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99, log=False):
        """
        :param name: name (used as parameter name)
        :param annuli: Annuli object
        :param defval: default value
        :param minval: minimum value
        :param maxval: maximum value
        :param log: use 10**value to convert to physical quantity
        """
        Cmpt.__init__(self, name, annuli)
        self.defval = defval
        self.minval = minval
        self.maxval = maxval
        self.log = log

    def defPars(self):
        return {self.name: Param(
            self.defval, minval=self.minval, maxval=self.maxval)}

    def computeProf(self, pars):
        v = pars[self.name].val
        if self.log:
            v = 10**v

        return N.full(self.annuli.nshells, float(v))

class CmptBinned(Cmpt):
    """A profile made of bins with a parameter for every N bin."""

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99,
        binning=1, interpolate=False, log=False):
        """
        name: name (used as start of parameter names)
        annuli: Annuli object
        defval: default value
        minval: minimum value
        maxval: maximum value
        binning: factor to bin annuli (how many bins per parameter)
        interpolate: interpolate values in intermediate bins
        log: use 10**values to convert to physical quantity
        """

        Cmpt.__init__(self, name, annuli)
        self.defval = defval
        self.minval = minval
        self.maxval = maxval
        self.binning = binning
        self.interpolate = interpolate
        self.log = log

        # rounded up division
        self.npars = -(-annuli.nshells // binning)
        # list of all the parameter names for the annuli
        self.parnames = ['%s_%03i' % (self.name, i) for i in range(self.npars)]

    def defPars(self):
        return {
            n: Param(self.defval, minval=self.minval, maxval=self.maxval)
            for n in self.parnames
            }

    def computeProf(self, pars):

        # extract radial parameters for model
        pvals = N.array([pars[n].val for n in self.parnames])

        if self.binning == 1:
            profile = pvals
        else:
            if self.interpolate:
                annidx = N.arange(self.annuli.nshells) / self.binning
                profile = N.interp(annidx, N.arange(self.npars), pvals)
            else:
                annidx = N.arange(self.annuli.nshells) // self.binning
                profile = pvals[annidx]

        if self.log:
            profile = 10**profile

        return profile

class CmptBinnedJumpPrior(CmptBinned):
    """A binned component using a prior that the values shouldn't jump
    by more than the factor given.
    """

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99,
        binning=1, interpolate=False, log=False, priorjump=0.):
        """
        :param name: name (used as start of parameter names)
        :param annuli: Annuli object
        :param defval: default value
        :param minval: minimum value
        :param maxval: maximum value
        :param binning: factor to bin annuli (how many bins per parameter)
        :param interpolate: interpolate values in intermediate bins
        :param log: use 10**values to convert to physical quantity
        :param priorjump: fractional difference allowed to jump between bins, implemented as a prior
        """

        CmptBinned.__init__(
            self, name, annuli, defval=defval, minval=minval,
            maxval=maxval, binning=binning, interpolate=interpolate, log=log)
        self.priorjump = priorjump

    def prior(self, pars):
        if self.priorjump <= 0:
            return 0.

        # this is a hacky prior to ensure that the values in the
        # profile do not jump by more than a factor of jumpprior
        pvals = N.array([pars[n].val for n in self.parnames])
        if self.log:
            pvals = 10**pvals

        priorval = 0
        lastp = pvals[0]
        for p in pvals[1:]:
            if abs(p/lastp-1) > self.priorjump:
                priorval -= abs(p/lastp-1)/self.priorjump
            elif abs(lastp/p-1) > self.priorjump:
                priorval -= abs(lastp/p-1)/self.priorjump
            lastp = p
        return 100*priorval

class CmptMoveRadBase(Cmpt):
    """Base class for components with bins which can move."""

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99,
        nradbins=5, log=False):
        """

        :param name: name (used as start of parameter names)
        :param Annuli annuli: Annuli object
        :param defval: default value
        :param minval: minimum value
        :param maxval: maximum value
        :param nradbins: number of control points ("bins") to use
        :param log: use 10**values to convert to physical quantity

        Model parameters: 'XX_YYY' and 'XX_r_YYY' where YYY goes from
        000...999, based on the number of radial bins.

        """

        Cmpt.__init__(self, name, annuli)
        self.defval = defval
        self.minval = minval
        self.maxval = maxval
        self.log = log
        self.nradbins = nradbins

        # list of all the parameter names for the annuli
        self.valparnames = ['%s_%03i' % (self.name, i) for i in range(nradbins)]
        self.radparnames = ['%s_r_%03i' % (self.name, i) for i in range(nradbins)]

    def defPars(self):
        valspars = {
            n: Param(self.defval, minval=self.minval, maxval=self.maxval)
            for n in self.valparnames
            }

        # log spacing in radius (with radial range fixed)
        rlogannuli = self.annuli.midpt_logkpc
        rlog = N.linspace(rlogannuli[0], rlogannuli[-1], self.nradbins)
        rpars = {
            n: Param(r, minval=rlogannuli[0], maxval=rlogannuli[-1], frozen=True)
            for n, r in zip(self.radparnames, rlog)
            }

        # combined parameters
        valspars.update(rpars)
        return valspars

class CmptInterpolMoveRad(CmptMoveRadBase):
    """A profile with control points, using interpolation to find the
    values in between.

    The radii of the control points are parameters (XX_r_999 in log kpc)
    """

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99,
            nradbins=5, log=False, intbeyond=False):

        """
        :param name: used as start of parameter names
        :param Annuli annuli: Annuli object
        :param defval: default value
        :param minval: minimum value
        :param maxval: maximum value
        :param nradbins: number of control points ("bins") to use
        :param interpolate: interpolate values in intermediate bins
        :param log: use 10**values to convert to physical quantity
        :param intbeyond: powerlaw interpolate inside and outside radii (assumes constant values if False)
        """

        CmptMoveRadBase.__init__(
            self, name, annuli, defval=defval, minval=minval, maxval=maxval,
            nradbins=nradbins, log=log)
        self.intbeyond = intbeyond

    def computeProf(self, pars):
        rvals = N.array([pars[n].val for n in self.radparnames])
        vvals = N.array([pars[n].val for n in self.valparnames])

        # radii might be in wrong order
        sortidxs = N.argsort(rvals)
        rvals = rvals[sortidxs]
        vvals = vvals[sortidxs]

        logannkpc = self.annuli.massav_logkpc
        if not self.intbeyond:
            # do interpolation, truncating at bounds
            prof = N.interp(logannkpc, rvals, vvals)
        else:
            # do interpolating, extending beyond
            # this is the gradient between each points
            grads = (vvals[1:]-vvals[:-1]) / (rvals[1:]-rvals[:-1])
            # index to point below this one (truncating if necessary)
            idx = N.searchsorted(rvals, logannkpc)-1
            idx = N.clip(idx, 0, len(grads)-1)
            # calculate line from point using gradient to next point
            dr = logannkpc - rvals[idx]
            prof = vvals[idx] + dr*grads[idx]

        if self.log:
            prof = 10**prof
        return prof

class CmptInterpolMoveRadIncr(CmptInterpolMoveRad):
    """Radial profile which has to rise inwards."""

    def prior(self, pars):
        rvals = N.array([pars[n].val for n in self.radparnames])
        vvals = N.array([pars[n].val for n in self.valparnames])

        # radii might be in wrong order
        sortidxs = N.argsort(rvals)
        rvals = rvals[sortidxs]
        vvals = vvals[sortidxs]

        if N.any( vvals[1:] > vvals[:-1] ):
            return -N.inf
        return 0

class CmptBinnedMoveRad(CmptMoveRadBase):
    """Binned data with movable radii."""

    def computeProf(self, pars):
        rvals = N.array([pars[n].val for n in self.radparnames])
        vvals = N.array([pars[n].val for n in self.valparnames])

        # radii might be in wrong order
        sortidxs = N.argsort(rvals)
        rvals = rvals[sortidxs]
        vvals = vvals[sortidxs]
        if self.log:
            vvals = 10**vvals

        # do binning
        idxs = N.searchsorted(rvals, self.annuli.massav_logkpc)
        idxsclip = N.clip(idxs, 0, len(vvals)-1)
        prof = vvals[idxsclip]
        return prof

class CmptBinWidthIncr(Cmpt):
    """Component where bins have increasing widths.
    Not sure this is very useful.

    Adds _dw_* parameters which are delta-widths
    """

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99,
        nradbins=5, log=False):

        Cmpt.__init__(self, name, annuli)
        self.defval = defval
        self.minval = minval
        self.maxval = maxval
        self.log = log
        self.nradbins = nradbins

        # list of all the parameter names for the annuli
        self.valparnames = ['%s_%03i' % (self.name, i) for i in range(nradbins)]
        self.radparnames = ['%s_dw_%03i' % (self.name, i) for i in range(nradbins)]

    def defPars(self):
        valspars = {
            n: Param(self.defval, minval=self.minval, maxval=self.maxval)
            for n in self.valparnames
            }

        # widths of bins
        maxrad_kpc = self.annuli.massav_kpc[-1]
        rvals_kpc = (N.linspace(0, N.sqrt(maxrad_kpc), self.nradbins+1))**2
        logwidths_kpc = rvals_kpc[1:-1] - rvals_kpc[:-2]
        wdelta = N.log10( N.hstack((logwidths_kpc, logwidths_kpc[1:]-logwidths_kpc[:-1])) )

        rpars = {
            n: Param(v, minval=-4., maxval=4., frozen=True)
            for n, v in zip(self.radparnames, wdelta)
            }

        # combined parameters
        valspars.update(rpars)

        valspars['%s_r_outer' % self.name] = Param(
            N.log10(self.annuli.edges_cm[-1] / kpc_cm),
            minval=-1, maxval=4., frozen=True)

        return valspars

    def computeProf(self, pars):
        wvals = N.array([pars[n].val for n in self.radparnames])
        vvals = N.array([pars[n].val for n in self.valparnames])
        if self.log:
            vvals = 10**vvals
        
        outer_kpc = 10**pars['%s_r_outer' % self.name].val

        bwincr = N.cumsum(10**wvals)
        rvals = N.cumsum(bwincr)
        rvals_kpc = rvals * (outer_kpc / rvals[-1])
        #print(rvals_kpc)

        idxs = N.searchsorted(rvals_kpc, self.annuli.massav_kpc)
        idxsclip = N.clip(idxs, 0, len(vvals)-1)
        prof = vvals[idxsclip]
        return prof

class CmptIncr(Cmpt):
    """An increasing-inward log component."""

    def __init__(
        self, name, annuli, defval=0., minval=-5, maxval=5):
        Cmpt.__init__(self, name, annuli)
        self.defval = defval
        self.minval = minval
        self.maxval = maxval

        self.npars = annuli.nshells
        # list of all the parameter names for the annuli
        self.parnames = ['%s_%03i' % (self.name, i) for i in range(self.npars)]

    def defPars(self):
        return {
            n: Param(self.defval, minval=self.minval, maxval=self.maxval)
            for n in self.parnames
            }

    def computeProf(self, pars):
        pvals = N.array([pars[n].val for n in self.parnames])
        pvals = 10**pvals

        pvals = N.cumsum(pvals[::-1])[::-1]
        return pvals

class CmptIncrMoveRad(Cmpt):
    """A profile with control points, using interpolation to find the
    values in between.

    The radii of the control points are model parameters XX_r_YYY (in
    log kpc), where YYY is the index of the annulus 000...999.  The y
    values (XX_YYY) are gradients in log (cm^-3 log kpc^-1).

    This model forces the density profile to increase inwards.

    """

    def __init__(
        self, name, annuli, defval=0., defouter=0., minval=-5., maxval=5.,
        nradbins=5, log=False):

        Cmpt.__init__(self, name, annuli)
        self.defval = defval
        self.defouter = defouter
        self.minval = minval
        self.maxval = maxval
        self.log = log
        self.nradbins = nradbins

        # list of all the parameter names for the annuli
        self.valparnames = ['%s_%03i' % (self.name, i) for i in range(nradbins)]
        self.radparnames = ['%s_r_%03i' % (self.name, i) for i in range(nradbins)]

    def defPars(self):
        valspars = {
            n: Param(self.defval, minval=self.minval, maxval=self.maxval)
            for n in self.valparnames
            }

        # log spacing in radius (with radial range fixed)
        rlogannuli = N.log10(self.annuli.midpt_cm / kpc_cm)
        rlog = N.linspace(rlogannuli[0], rlogannuli[-1], self.nradbins)
        rpars = {
            n: Param(r, minval=rlogannuli[0], maxval=rlogannuli[-1], frozen=True)
            for n, r in zip(self.radparnames, rlog)
            }

        # combined parameters
        valspars.update(rpars)

        valspars['%s_outer' % self.name] = Param(
            self.defouter, minval=self.minval, maxval=self.maxval)

        return valspars

    def computeProf(self, pars):
        rvals = N.array([pars[n].val for n in self.radparnames])
        vvals = N.array([pars[n].val for n in self.valparnames])

        # radii might be in wrong order
        sortidxs = N.argsort(rvals)
        rvals = rvals[sortidxs]
        vvals = vvals[sortidxs]

        loggradprof = N.interp(self.annuli.massav_logkpc, rvals, vvals)
        gradprof = 10**loggradprof

        # work out deltas
        logekpc = self.annuli.edges_logkpc
        logwidthkpc = logekpc[1:] - logekpc[:-1]

        if not N.isfinite(logwidthkpc[0]):
            ekpc = self.annuli.edges_kpc
            logwidthkpc[0] = logekpc[1] - 0.5*N.log10(ekpc[0]+ekpc[1])

        deltas = gradprof * logwidthkpc
        outer = 10**pars['%s_outer' % self.name].val
        prof = N.cumsum(deltas[::-1])[::-1] + outer

        return prof

def betaprof(rin_cm, rout_cm, n0, beta, rc):
    """Return beta function density profile

    Calculates average density in each shell.
    """

    # this is the average density in each shell
    # i.e.
    # Integrate[n0*(1 + (r/rc)^2)^(-3*beta/2)*4*Pi*r^2, r]
    # between r1 and r2
    def intfn(r):
        return ( 4/3 * n0 * math.pi * r**3 *
                 hyp2f1(3/2, 3/2*beta, 5/2, -(r/rc)**2)
                 )

    r1 = rin_cm * (1/kpc_cm)
    r2 = rout_cm * (1/kpc_cm)
    nav = (intfn(r2) - intfn(r1)) / (4/3*math.pi * (r2**3 - r1**3))
    return nav

class CmptBeta(Cmpt):
    """Beta model.

    Model parameters are XX_n0 (log base 10), XX_beta and XX_rc (log10 kpc)
    """

    def defPars(self):
        return {
            '%s_n0' % self.name: Param(-2., minval=-7., maxval=2.),
            '%s_beta' % self.name: Param(2/3, minval=0., maxval=4.),
            '%s_rc' % self.name: Param(1.7, minval=-1, maxval=3.7),
            }

    def computeProf(self, pars):
        n0 = 10**pars['%s_n0' % self.name].val
        beta = pars['%s_beta' % self.name].val
        rc = 10**pars['%s_rc' % self.name].val
        return betaprof(self.annuli.rin_cm, self.annuli.rout_cm, n0, beta, rc)

class CmptDoubleBeta(Cmpt):
    """Double beta model.

    Model parameters are XX_n0_N (log base 10), XX_beta_N and XX_rc_N (log10
    kpc), where N is 1 and 2
    """

    def defPars(self):
        return {
            '%s_n0_1' % self.name: Param(-2., minval=-7., maxval=2.),
            '%s_beta_1' % self.name: Param(2/3, minval=0., maxval=4.),
            '%s_rc_1' % self.name: Param(1.3, minval=-1., maxval=3.7),
            '%s_n0_2' % self.name: Param(-3., minval=-7., maxval=2.),
            '%s_beta_2' % self.name: Param(0.5, minval=0., maxval=4.),
            '%s_rc_2' % self.name: Param(2, minval=-1., maxval=3.7),
            }

    def computeProf(self, pars):
        return (
            betaprof(
                self.annuli.rin_cm, self.annuli.rout_cm,
                10**pars['%s_n0_1' % self.name].val,
                pars['%s_beta_1' % self.name].val,
                10**pars['%s_rc_1' % self.name].val) +
            betaprof(
                self.annuli.rin_cm, self.annuli.rout_cm,
                10**pars['%s_n0_2' % self.name].val,
                pars['%s_beta_2' % self.name].val,
                10**pars['%s_rc_2' % self.name].val))

class CmptVikhDensity(Cmpt):
    """Density model from Vikhlinin+06, Eqn 3.

    Modes:
    'double': all components
    'single': only first component
    'betacore': only first two terms of first component

    Densities and radii are are log base 10
    """

    def __init__(self, name, annuli, mode='double'):
        Cmpt.__init__(self, name, annuli)
        self.mode = mode

    def defPars(self):
        pars = {
            '%s_n0_1' % self.name: Param(-3., minval=-7., maxval=2.),
            '%s_beta_1' % self.name: Param(2/3., minval=0., maxval=4.),
            '%s_logrc_1' % self.name: Param(2.3, minval=-1., maxval=3.7),
            '%s_alpha' % self.name: Param(0., minval=-1, maxval=2.),
            }

        if self.mode in ('single', 'double'):
            pars.update({
                '%s_epsilon' % self.name: Param(3., minval=0., maxval=5.),
                '%s_gamma' % self.name: Param(3., minval=0., maxval=10, frozen=True),
                '%s_logr_s' % self.name: Param(2.7, minval=0, maxval=3.7),
                })

        if self.mode == 'double':
            pars.update({
                    '%s_n0_2' % self.name: Param(-1., minval=-7., maxval=2.),
                    '%s_beta_2' % self.name: Param(0.5, minval=0., maxval=4.),
                    '%s_logrc_2' % self.name: Param(1.7, minval=-1., maxval=3.7),
                    })
        return pars

    def vikhFunction(self, pars, radii_kpc):
        n0_1 = 10**pars['%s_n0_1' % self.name].val
        beta_1 = pars['%s_beta_1' % self.name].val
        rc_1 = 10**pars['%s_logrc_1' % self.name].val
        alpha = pars['%s_alpha' % self.name].val

        r = radii_kpc
        retn_sqd = (
            n0_1**2 *
            (r/rc_1)**(-alpha) / (
                (1+r**2/rc_1**2)**(3*beta_1-0.5*alpha))
            )

        if self.mode in ('single', 'double'):
            r_s = 10**pars['%s_logr_s' % self.name].val
            epsilon = pars['%s_epsilon' % self.name].val
            gamma = pars['%s_gamma' % self.name].val

            retn_sqd /= (1+(r/r_s)**gamma)**(epsilon/gamma)

        if self.mode == 'double':
            n0_2 = 10**pars['%s_n0_2' % self.name].val
            rc_2 = 10**pars['%s_logrc_2' % self.name].val
            beta_2 = pars['%s_beta_2' % self.name].val

            retn_sqd += n0_2**2 / (1 + r**2/rc_2**2)**(3*beta_2)

        return N.sqrt(retn_sqd)

    def computeProf(self, pars):
        return self.vikhFunction(pars, self.annuli.midpt_kpc)

    def prior(self, pars):
        rc_1 = 10**pars['%s_logrc_1' % self.name].val
        try:
            r_s = 10**pars['%s_logr_s' % self.name].val
        except KeyError:
            return 0
        if rc_1 > r_s:
            return -N.inf
        return 0

class CmptMcDonaldTemperature(Cmpt):
    """Temperature model from McDonald+14, equation 1

    Radii are are log base 10
    """

    def defPars(self):
        n = self.name
        pars = {
            '%s_logT0' % n: Param(0.5, minval=-1, maxval=1.7),
            '%s_logTmin' % n: Param(0.5, minval=-1, maxval=1.7),
            '%s_logrc' % n: Param(2.3, minval=-1., maxval=3.7),
            '%s_logrt' % n: Param(2.7, minval=0, maxval=3.7),
            '%s_acool' % n: Param(2., minval=0, maxval=4.),
            '%s_a' % n: Param(0., minval=-4, maxval=4.),
            '%s_b' % n: Param(1., minval=0.001, maxval=4.),
            '%s_c' % n: Param(1., minval=0, maxval=4.),
            }
        return pars

    def computeProf(self, pars):
        n = self.name
        T0 = 10**pars['%s_logT0' % n].val
        Tmin = 10**pars['%s_logTmin' % n].val
        rc = 10**pars['%s_logrc' % n].val
        rt = 10**pars['%s_logrt' % n].val
        acool = pars['%s_acool' % n].val
        a = pars['%s_a' % n].val
        b = pars['%s_b' % n].val
        c = pars['%s_c' % n].val

        x = self.annuli.midpt_kpc
        T = (
            T0
            * ((x/rc)**acool + (Tmin/T0))
            / (1 + (x/rc)**acool)
            * (x/rt)**-a
            / (1 + (x/rt)**b)**(c/b)
            )
        return T
