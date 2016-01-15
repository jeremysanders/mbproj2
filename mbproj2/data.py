# -*- coding: utf-8 -*-

from __future__ import division, print_function
from itertools import izip
import numpy as N
from physconstants import kpc_cm
import utils
import countrate

class Annuli:
    """Store information about the annuli."""

    def __init__(self, edges_arcmin, cosmology):
        self.cosmology = cosmology

        self.edges_arcmin = edges_arcmin
        self.geomarea_arcmin2 = N.pi * (edges_arcmin[1:]**2 - edges_arcmin[:-1]**2)
        self.nshells = len(edges_arcmin) - 1

        # radii of shells
        e = cosmology.kpc_per_arcsec * edges_arcmin * 60 * kpc_cm
        self.edges_cm = e
        rout = self.rout_cm = e[1:]
        rin = self.rin_cm = e[:-1]

        # this is the average radius, assuming constant mass in the shell
        self.massav_cm = 0.75 * (rout**4 - rin**4) / (rout**3 - rin**3)

        # mid point of shell
        self.midpt_cm = 0.5 * (rout + rin)

        # shell widths
        self.widths_cm = rout - rin

        # geometric area
        self.geomarea_cm2 = N.pi * (rout**2-rin**2)

        # volume of shells
        self.vols_cm3 = 4/3 * N.pi * (rout**3-rin**3)

        # projected volumes
        self.projvols_cm3 = utils.projectionVolumeMatrix(e)

        # count rate helper (associated with cosmology)
        self.ctrate = countrate.CountRate(cosmology)

def loadAnnuli(filename, cosmology, centrecol=0, hwcol=1):
    """Load annuli from data file, if radius of centre (arcmin) is
    given by centrecol and bin half-width is given by hwcol
    column. cosmology is a Cosmology object."""

    data = N.loadtxt(filename)
    centre = data[:,centrecol]
    hw = data[:,hwcol]

    edges = N.concatenate([[centre[0]-hw[0]], centre+hw])
    return Annuli(edges, cosmology)

def expandlist(x, length):
    """If x is a list, check it has the length length.
    Otherwise, expand item to be a list with length given."""

    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) != length:
            raise RuntimeError('Length not same')
        return list(x)
    else:
        return [x]*length

class Band:
    """Count profile in a band."""

    def __init__(
        self, emin_keV, emax_keV, cts, rmf, arf, exposures,
        backrates=None, areascales=None):

        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.cts = cts
        self.rmf = rmf
        self.arf = arf
        self.exposures = N.array(expandlist(exposures, len(cts)))

        if backrates is None:
            self.backrates = N.zeros(len(cts))
        else:
            self.backrates = N.array(expandlist(backrates, len(cts)))

        if areascales is None:
            self.areascales = N.ones(len(cts))
        else:
            self.areascales = N.array(expandlist(areascales, len(cts)))

    def calcProjProfile(self, annuli, ne_prof, T_prof, Z_prof, NH_1022pcm2):
        """Predict profile given cluster profiles."""

        rates = annuli.ctrate.getCountRate(
            self.rmf, self.arf, self.emin_keV, self.emax_keV,
            NH_1022pcm2, T_prof, Z_prof, ne_prof)

        projrates = N.dot(rates, annuli.projvols_cm3) * self.areascales
        projrates += self.backrates * (annuli.geomarea_arcmin2*self.areascales)
        projrates *= self.exposures

        return projrates

def loadBand(
    filename, emin_keV, emax_keV, rmf, arf,
    radiuscol=0, hwcol=1, ctcol=2, areacol=3, expcol=4):
    """Load a band using standard data format."""

    data = N.loadtxt(filename)
    radii = data[:,radiuscol]
    hws = data[:,hwcol]
    cts = data[:,ctcol]
    areas = data[:,areacol]
    exps = data[:,expcol]

    geomareas = N.pi*4*radii*hws
    areascales = areas/geomareas

    return Band(emin_keV, emax_keV, cts, rmf, arf, exps, areascales=areascales)

class Data:
    """Dataset class."""

    def __init__(self, bands, annuli):
        self.bands = bands
        self.annuli = annuli

    def calcProfiles(self, model, pars):
        """Predict model profiles for each band.

        Returns profiles, log-likelihood
        """

        ne_prof, T_prof, Z_prof = model.computeProfs(pars)

        profs = []
        for band in self.bands:
            modprof = band.calcProjProfile(
                self.annuli, ne_prof, T_prof, Z_prof, model.NH_1022pcm2)
            profs.append(modprof)

        return profs

    def calcLikelihood(self, predprofs):
        """Given predicted profiles, calculate likelihood."""

        likelihood = 0.
        for band, predprof in izip(self.bands, predprofs):
            likelihood += utils.cashLogLikelihood(band.cts, predprof)
        return likelihood
