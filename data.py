from __future__ import division, print_function

import numpy as N
from physconstants import kpc_cm
import utils

class Annuli:
    """Store information about the annuli."""

    def __init__(self, edges_arcmin, cosmology):
        self.cosmology = cosmology
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

        # volume of shells
        self.vols_cm3 = 4/3 * N.pi * (e[1:]**3 - e[:-1]**3)

        # projected volumes (make a copy for speed)
        self.projvols_cm3 = N.ascontiguousarray(
            utils.projectionVolumeMatrix(e).transpose())

def loadAnnuli(filename, centrecol, hwcol, cosmology):
    data = N.loadtxt(filename)
    centre = data[:,centrecol]
    hw = data[:,hwcol]

    edges = N.concatenate([[centre[0]-hw[0]], centre+hw])
    return Annuli(edges, cosmology)

class Data:
    pass
