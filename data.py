from utils import projectionVolume
from physconstants import kpc_cm
import numpy as N

class Annuli:
    """Store information about the annuli."""

    def __init__(self, edges_arcmin, cosmology):
        self.cosmology = cosmology
        self.edges_arcmin = edges_arcmin
        self.nshells = len(edges_arcmin) - 1

        e = cosmology.kpc_per_arcsec * edges_arcmin * 60 * kpc_cm

        # radii of shells
        self.edges_cm = e
        self.massav_cm = 0.75*(e[1:]**4 - e[:-1]**4) / (
            e[1:]**3 - e[:-1]**3)
        self.midpt_cm = 0.5*(e[1:] + e[:-1])
        self.widths_cm = e[1:] - e[:-1]

        # volume of shells
        self.vols_cm3 = 4./3. * N.pi * (e[1:]**3 - e[:-1]**3)

        # projected volumes
        self.projvols_cm3 = N.ascontiguousarray(
            utils.projectionVolumeMatrix(e).transpose())

class Data:
    pass
