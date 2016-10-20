# -*- coding: utf-8 -*-

"""Module to get count rates for temperatures, densities and metallicities.

Results are taken from xspec, interpolating between results at fixed temperatures
and metallicities
"""

from __future__ import division, print_function

import os.path
import h5py

import numpy as N
import scipy.interpolate

import utils
from xspechelper import XSpecHelper

class CountRate(object):
    """Object caches count rates for temperatures, densities and metallicites."""

    Tmin = 0.06
    Tmax = 60.
    Tsteps = 100
    Tlogvals = N.linspace(N.log(Tmin), N.log(Tmax), Tsteps)

    def __init__(self, cosmo):
        """Initialise with cosmology."""
        self.cosmo = cosmo
        self.ctcache = {}
        self.fluxcache = None

    def getCountRate(self, rmf, arf, minenergy_keV, maxenergy_keV,
                     NH_1022, T_keV, Z_solar, ne_cm3):
        """get count rate in counts per cm3 for parcel of gas between energies given."""

        key = (minenergy_keV, maxenergy_keV, self.cosmo.z, NH_1022, rmf, arf)

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
        """Work out the counts for the temperature values for the key given."""
        minenergy_keV, maxenergy_keV, z, NH_1022, rmf, arf = key

        if not os.path.exists(rmf):
            raise RuntimeError('RMF %s does not exist' % rmf)
        if not os.path.exists(arf):
            raise RuntimeError('ARF %s does not exist' % arf)

        hdffile = 'countrate_cache.hdf5'

        # nasty hack to stop concurrent access breaking
        with utils.WithLock(hdffile + '.lockdir') as lock:
            textkey = '_'.join(str(x) for x in key).replace('/', '@')
            with h5py.File(hdffile) as f:
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

    def getBolometricFlux(self, T_keV, Z_solar, ne_cm3):
        """Get bolometric flux per cm3 in erg/cm2/s."""

        if not self.fluxcache:
            self.makeFluxCache()

        logT = N.log( N.clip(T_keV, self.Tmin, self.Tmax) )

        # evaluate interpolation functions for temperature given
        Z0_flux, Z1_flux = [f(logT) for f in self.fluxcache]

        # use Z=0 and Z=1 count rates to evaluate at Z given
        return (Z0_flux + (Z1_flux-Z0_flux)*Z_solar)*ne_cm3**2

    def makeFluxCache(self):
        """Work out fluxes for the temperature grid points and response."""

        xspec = XSpecHelper()
        xspec.dummyResponse()
        results = []
        # we can work out the counts at other metallicities from two values
        # we also work at a density of 1 cm^-3
        for Z_solar in (0., 1.):
            Zresults = []
            for Tlog in CountRate.Tlogvals:
                flux = xspec.getFlux(N.exp(Tlog), Z_solar, self.cosmo, 1.)
                Zresults.append(flux)

            # store functions which interpolate the results from above
            results.append( scipy.interpolate.interpolate.interp1d(
                CountRate.Tlogvals, N.array(Zresults), kind='cubic' ) )

        xspec.finish()
        self.fluxcache = tuple(results)
