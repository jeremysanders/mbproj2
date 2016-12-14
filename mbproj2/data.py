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

"""Contains classes:

Annuli: geometry of annuli and cosmology.
Band: information on a surface brightness profile.
Data: set of Band objects.
"""

from __future__ import division, print_function
import numpy as N

from .physconstants import kpc_cm
from . import utils
from .utils import uprint
from . import countrate

class Annuli:
    """Store information about the annuli."""

    def __init__(self, edges_arcmin, cosmology):
        """
        edges_arcmin: numpy array of edges of annuli in arcmin
        cosmology: Cosmology object
        """

        self.update(edges_arcmin, cosmology)

    def __getstate__(self):
        """Don't save derived quantities when pickling."""
        return {
            'edges_arcmin': self.edges_arcmin,
            'cosmology': self.cosmology,
            }
    def __setstate__(self, state):
        """Recalculate derived quantities when unpickling."""
        self.update(state['edges_arcmin'], state['cosmology'])

    def update(self, edges_arcmin, cosmology):
        """Change the annuli.

        Useful for recalculating models with new grid.
        """

        self.edges_arcmin = edges_arcmin
        self.cosmology = cosmology

        self.geomarea_arcmin2 = N.pi * (edges_arcmin[1:]**2 - edges_arcmin[:-1]**2)
        self.nshells = len(edges_arcmin) - 1

        # edges of shells
        e = cosmology.kpc_per_arcsec * edges_arcmin * 60 * kpc_cm
        self.edges_cm = e
        self.edges_kpc = e / kpc_cm
        self.edges_logkpc = N.log10(self.edges_kpc)

        # inner and outer radii
        rout = self.rout_cm = e[1:]
        rin = self.rin_cm = e[:-1]

        # mid point of shell
        self.midpt_cm = 0.5 * (rout + rin)
        self.midpt_kpc = self.midpt_cm / kpc_cm
        self.midpt_logkpc = N.log10(self.midpt_kpc)

        # this is the average radius, assuming constant mass in the shell
        self.massav_cm = 0.75 * (rout**4 - rin**4) / (rout**3 - rin**3)
        self.massav_kpc = self.massav_cm / kpc_cm
        self.massav_logkpc = N.log10(self.massav_kpc)

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

    uprint('Loading annuli from', filename)
    data = N.loadtxt(filename)
    centre = data[:,centrecol]
    hw = data[:,hwcol]

    edges = N.concatenate([[centre[0]-hw[0]], centre+hw])
    return Annuli(edges, cosmology)

def expandlist(x, length):
    """If x is a list, check it has the length length.
    Otherwise, expand item to be a list with length given."""

    if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, N.ndarray):
        if len(x) != length:
            raise RuntimeError('Length not same')
        return N.array(x, dtype=N.float64)
    else:
        return N.full(length, float(x))

class Band:
    """Count profile in a band."""

    def __init__(
        self, emin_keV, emax_keV, cts, rmf, arf, exposures,
        backrates=None, areascales=None, psfmatrix=None):
        """
        emin_keV: minimum energy of band in keV
        emax_keV: maximum energy of band in keV
        cts: numpy array of counts in each annulus
        rmf: response matrix filename
        arf: ancillary response matrix filename
        exposures: numpy array of exposures in each annulus

        optionally:
        backrates: numpy array of rates of cts/s/arcmin^2 in each annulus
        areascales: numpy array of scaling factors to convert from geometric
          area in annulus to real area (including pixels)
        psfmatrix: matrix to convolve to account for PSF, usually calculated
          using functions in psfconvolve submodule
        """

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
            self.areascales = N.array(areascales)

        self.psfmatrix = psfmatrix

    def calcProjProfile(self, annuli, ne_prof, T_prof, Z_prof, NH_1022pcm2, backscale=1.):
        """Predict profile given cluster profiles."""

        rates = annuli.ctrate.getCountRate(
            self.rmf, self.arf, self.emin_keV, self.emax_keV,
            NH_1022pcm2, T_prof, Z_prof, ne_prof)

        projrates = annuli.projvols_cm3.dot(rates)

        if self.psfmatrix is not None:
            projrates = self.psfmatrix.dot(projrates)

        projrates *= self.areascales

        projrates += self.backrates * backscale * annuli.geomarea_arcmin2 * self.areascales
        projrates *= self.exposures

        return projrates

def loadBand(
    filename, emin_keV, emax_keV, rmf, arf,
    radiuscol=0, hwcol=1, ctcol=2, areacol=3, expcol=4):
    """Load a band using standard data format."""

    uprint('Loading band %g to %g keV from %s' % (emin_keV, emax_keV, filename))

    data = N.loadtxt(filename)
    radii = data[:,radiuscol]
    hws = data[:,hwcol]
    cts = data[:,ctcol]
    areas = data[:,areacol]
    exps = data[:,expcol]

    geomareas = N.pi*((radii+hws)**2-(radii-hws)**2)
    areascales = areas/geomareas

    return Band(emin_keV, emax_keV, cts, rmf, arf, exps, areascales=areascales)

class Data:
    """Dataset class."""

    def __init__(self, bands, annuli):
        """
        bands: list of Band objects
        annuli: Annuli object
        """
        
        self.bands = bands
        self.annuli = annuli
