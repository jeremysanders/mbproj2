from __future__ import division, print_function

from itertools import izip
import utils

class Fit:
    """Class to help fitting model, by keeping track of thawed parameters."""

    def __init__(self, pars, model, data):
        self.pars = pars
        self.model = model
        self.data = data
        self.thawed = [
            name for name, par in sorted(pars.items()) if not par.frozen]

    def calcProfiles(self):
        """Predict model profiles for each band.

        Returns profiles, log-likelihood
        """

        ne_prof, T_prof, Z_prof = self.model.computeProfs(self.pars)

        profs = []
        for band in self.data.bands:
            modprof = band.calcProjProfile(
                self.data.annuli, ne_prof, T_prof, Z_prof,
                self.model.NH_1022pcm2)
            profs.append(modprof)

        return profs

    def calcLikelihood(self, predprofs):
        """Given predicted profiles, calculate likelihood."""

        likelihood = 0.
        for band, predprof in izip(self.data.bands, predprofs):
            likelihood += utils.cashLogLikelihood(band.cts, predprof)
        return likelihood

    def thawedVals(self):
        """Return values of thawed parameters."""
        return [self.pars[name].val for name in self.thawed]

    def updateThawed(self, vals):
        for val, name in izip(vals, self.thawed):
            self.pars[name].val = val
