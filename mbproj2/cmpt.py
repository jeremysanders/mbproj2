from __future__ import division, print_function
from param import Param
import numpy as N
 
class Cmpt:
    """Parametrise a profile."""

    def __init__(self, name, annuli):
        """annuli is an Annuli object."""
        self.name = name
        self.annuli = annuli

    def defPars(self):
        """Return default parameters."""

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

        return N.full(self.annuli.nshells, v)

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
