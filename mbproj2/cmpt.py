# Components which make up a profile. Each component has a set of
# parameters (of type Param).

from __future__ import division, print_function
import math

import numpy as N
from scipy.special import hyp2f1

from fit import Param
from physconstants import kpc_cm

class Cmpt:
    """Parametrise a profile."""

    def __init__(self, name, annuli):
        """annuli is an Annuli object."""
        self.name = name
        self.annuli = annuli

    def defPars(self):
        """Return default parameters (dict of Param)."""

    def computeProf(self, pars):
        """Return profile for annuli given."""
        pass

class CmptFlat(Cmpt):
    """A flat profile."""

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99, log=False):

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
    """A profile made of bins."""

    def __init__(
        self, name, annuli, defval=0., minval=-1e99, maxval=1e99,
        binning=1, interpolate=False, log=False):

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
        self.parnames = ['%s_%03i' % (self.name, i) for i in xrange(self.npars)]

    def defPars(self):
        return {
            n: Param(self.defval, minval=self.minval, maxval=self.maxval)
            for n in self.parnames
            }

    def computeProf(self, pars):

        # extract radial parameters for model
        pvals = N.array([pars[n].val for n in self.parnames])
        if self.log:
            pvals = 10**pvals

        if self.binning == 1:
            return pvals
        else:
            if self.interpolate:
                annidx = N.arange(self.annuli.nshells) / self.binning
                return N.interp(annidx, N.arange(self.npars), pvals)
            else:
                annidx = N.arange(self.annuli.nshells) // self.binning
                return pvals[annidx]

class CmptIncr(Cmpt):
    """An increasing-inward component."""

    def __init__(
        self, name, annuli, defval=0., minval=-5, maxval=5):
        Cmpt.__init__(self, name, annuli)
        self.defval = defval
        self.minval = minval
        self.maxval = maxval

        self.npars = annuli.nshells
        # list of all the parameter names for the annuli
        self.parnames = ['%s_%03i' % (self.name, i) for i in xrange(self.npars)]

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

def betaprof(rin_cm, rout_cm, n0, beta, rc):
    """Return beta function density profile."""

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
    """Beta model."""

    def defPars(self):
        return {
            'n0': Param(-2., minval=-7., maxval=2.),
            'beta': Param(2/3, minval=0., maxval=4.),
            'rc': Param(50., minval=0., maxval=5000.),
            }

    def computeProf(self, pars):
        n0 = 10**pars['n0'].val
        beta = pars['beta'].val
        rc = pars['rc'].val
        return betaprof(self.annuli.rin_cm, self.annuli.rout_cm, n0, beta, rc)

class CmptDoubleBeta(Cmpt):
    """Double beta model."""

    def defPars(self):
        return {
            'n0_1': Param(-2., minval=-7., maxval=2.),
            'beta_1': Param(2/3, minval=0., maxval=4.),
            'rc_1': Param(20., minval=0., maxval=5000.),
            'n0_2': Param(-3., minval=-7., maxval=2.),
            'beta_2': Param(0.5, minval=0., maxval=4.),
            'rc_2': Param(100., minval=0., maxval=5000.),
            }

    def computeProf(self, pars):
        return (
            betaprof(
                self.annuli.rin_cm, self.annuli.rout_cm,
                10**pars['n0_1'].val, pars['beta_1'].val, pars['rc_1'].val) +
            betaprof(
                self.annuli.rin_cm, self.annuli.rout_cm,
                10**pars['n0_2'].val, pars['beta_2'].val, pars['rc_2'].val))
