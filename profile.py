from __future__ import division, print_function
import numpy as N

class Profile:
    """Parametrise a profile."""

    def __init__(self, name, annuli):
        """annuli is an Annuli object."""
        self.name = name
        self.annuli = annuli

    def numPars(self):
        """Return number of parameters."""

    def defPars(self):
        """Return default parameters."""

    def computeProf(self, pars):
        """Return profile for annuli given."""
        pass

class ProfileFlat(Profile):
    """A flat profile."""

    def __init__(self, name, annuli, defval=0.):
        Profile.__init__(self, name, annuli)
        self.defval = defval
        self.freeze = freeze

    def numPars(self):
        return 0 if self.freeze else 1

    def defPars(self):
        return [] if self.freeze else [('val', self.defval)]

    def computeProf(self, pars):
        return N.full(self.annuli.nshells, self.defval if self.freeze else pars[0])

class ProfileBinned(Profile):
    """A profile made of bins."""

    def __init__(
        self, name, annuli, defval=0., binning=1, interpolate=False):

        Profile.__init__(self, name, annuli)
        self.defval = defval
        self.binning = binning
        self.interpolate = interpolate
        self.freeze = freeze

        # rounded up division
        self.npars = 0 if freeze else -(-annuli.nshells // binning)

    def numPars(self):
        return self.npars

    def defPars(self):
        if N.isscalar(self.defval):
            return [('val%03i' % i, self.defval) for i in xrange(self.npars)]
        else:
            return [] if self.freeze else [
                ('val%03i' % i, v) for i, v in enumerate(self.defval)]

    def computeProf(self, pars):
        if self.freeze:
            if N.isscalar(self.defval):
                return N.full(self.annuli.nshells, self.defval)
            else:
                annidx = N.arange(self.annuli.nshells) // self.binfactor
                return N.array(self.defval)[annidx]

        if self.binfactor == 1:
            return N.array(pars)
        else:
            if self.interpolate:
                annidx = N.arange(self.annuli.nshells) / self.binfactor
                return N.interp(annidx, N.arange(self.npars), pars)
            else:
                annidx = N.arange(self.annuli.nshells) // self.binfactor
                return N.array(pars)[annidx]
