from __future__ import division, print_function

import numpy as N
from physconstants import kpc_cm
import utils

class Annuli:
    """Store information about the annuli."""

    def __init__(self, edges_arcmin, cosmology, NH_1022pcm2):
        self.cosmology = cosmology
        self.NH_1022pcm2 = NH_1022pcm2

        self.edges_arcmin = edges_arcmin
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

        # projected volumes (make a copy for speed)
        self.projvols_cm3 = N.ascontiguousarray(
            utils.projectionVolumeMatrix(e).transpose())

def loadAnnuli(filename, cosmology, NH, centrecol=0, hwcol=1):
    """Load annuli from data file, if radius of centre (arcmin) is
    given by centrecol and bin half-width is given by hwcol
    column. cosmology is a Cosmology object."""

    data = N.loadtxt(filename)
    centre = data[:,centrecol]
    hw = data[:,hwcol]

    edges = N.concatenate([[centre[0]-hw[0]], centre+hw])
    return Annuli(edges, cosmology, NH)

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


