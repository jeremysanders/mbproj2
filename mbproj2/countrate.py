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

"""Module to get count rates for temperatures, densities and
metallicities.

Results are taken from xspec, interpolating between results at fixed
temperatures and metallicities
"""

from __future__ import division, print_function, absolute_import

import os.path
import h5py

import numpy as N
import scipy.interpolate

from . import utils
from .xspechelper import XSpecHelper

class CountRate:
    """Object caches count rates for temperatures, densities and
    metallicities."""

    Tmin = 0.06
    Tmax = 60.
    Tsteps = 100
    Tlogvals = N.linspace(N.log(Tmin), N.log(Tmax), Tsteps)

    def __init__(self, cosmo):
        """Initialise with cosmology."""
        self.cosmo = cosmo
        self.ctcache = {}
        self.fluxcache = {}

    def getCountRate(self, rmf, arf, minenergy_keV, maxenergy_keV,
                     NH_1022, T_keV, Z_solar, ne_cm3):
        """get count rate in counts per cm3 for parcel of gas between energies
        given."""

        key = (
            minenergy_keV, maxenergy_keV, self.cosmo.z, NH_1022, rmf, arf,
            XSpecHelper.abun, XSpecHelper.absorb, XSpecHelper.apecroot
        )

        if key not in self.ctcache:
            self.addCountCache(key)

        # evaluate interpolation functions for temperature given
        a0, a1 = self.ctcache[key]
        T_keV = N.clip(T_keV, self.Tmin, self.Tmax)
        logT = N.log(T_keV)
        Z0_ctrate = N.exp(N.interp(logT, self.Tlogvals, a0))
        Z1_ctrate = N.exp(N.interp(logT, self.Tlogvals, a1))

        # use Z=0 and Z=1 count rates to evaluate at Z given
        return (Z0_ctrate + (Z1_ctrate-Z0_ctrate)*Z_solar)*ne_cm3**2

    def addCountCache(self, key):
        """Work out the counts for the temperature values for the key
        given.
        """

        minenergy_keV, maxenergy_keV, z, NH_1022, rmf, arf, abun, absorb, apecroot = key

        if not os.path.exists(rmf):
            raise RuntimeError('RMF %s does not exist' % rmf)
        #if not os.path.exists(arf):
        #    raise RuntimeError('ARF %s does not exist' % arf)

        hdffile = 'countrate_cache.hdf5'

        # nasty hack to stop concurrent access breaking
        with utils.WithLock(hdffile + '.lockdir') as lock:
            textkey = '_'.join(str(x) for x in key).replace('/', '@')
            with h5py.File(hdffile, 'a') as f:
                if textkey not in f:
                    xspec = XSpecHelper()
                    xspec.changeResponse(rmf, arf, minenergy_keV, maxenergy_keV)
                    allZresults = []
                    for Z_solar in (0., 1.):
                        Zresults = []
                        for Tlog in CountRate.Tlogvals:
                            countrate = xspec.getCountsPerSec(
                                NH_1022, N.exp(Tlog), Z_solar, self.cosmo, 1.)
                            Zresults.append(countrate)
                        Zresults = N.array(Zresults)
                        Zresults[Zresults < 1e-300] = 1e-300
                        allZresults.append(Zresults)
                    xspec.finish()
                    allZresults = N.array(allZresults)
                    f[textkey] = allZresults
                allZresults = N.array(f[textkey])

        results = (N.log(allZresults[0]), N.log(allZresults[1]))
        self.ctcache[key] = results

    def getFlux(self, T_keV, Z_solar, ne_cm3, emin_keV=0.01, emax_keV=100., NH_1022pcm2=0.):
        """Get flux per cm3 in erg/cm2/s.

        emin_keV and emax_keV are the energy range
        """

        key = (
            emin_keV, emax_keV, NH_1022pcm2,
            XSpecHelper.abun, XSpecHelper.absorb, XSpecHelper.apecroot)
        if key not in self.fluxcache:
            self.makeFluxCache(*key)
        fluxcache = self.fluxcache[key]

        logT = N.log( N.clip(T_keV, self.Tmin, self.Tmax) )

        # evaluate interpolation functions for temperature given
        Z0_flux, Z1_flux = [f(logT) for f in fluxcache]

        # use Z=0 and Z=1 count rates to evaluate at Z given
        return (Z0_flux + (Z1_flux-Z0_flux)*Z_solar)*ne_cm3**2

    def makeFluxCache(self, emin_keV, emax_keV, NH_1022pcm2, abun, absorb, apecroot):
        """Work out fluxes for the temperature grid points and response."""

        xspec = XSpecHelper()
        xspec.dummyResponse()
        results = []
        # we can work out the counts at other metallicities from two values
        # we also work at a density of 1 cm^-3
        for Z_solar in (0., 1.):
            Zresults = []
            for Tlog in CountRate.Tlogvals:
                flux = xspec.getFlux(
                    N.exp(Tlog), Z_solar, self.cosmo, 1.,
                    NH_1022pcm2=NH_1022pcm2,
                    emin_keV=emin_keV, emax_keV=emax_keV)
                Zresults.append(flux)

            # store functions which interpolate the results from above
            results.append( scipy.interpolate.interpolate.interp1d(
                CountRate.Tlogvals, N.array(Zresults), kind='cubic' ) )

        xspec.finish()
        self.fluxcache[(
            emin_keV, emax_keV, NH_1022pcm2,
            XSpecHelper.abun, XSpecHelper.absorb, XSpecHelper.apecroot
        )] = tuple(results)
