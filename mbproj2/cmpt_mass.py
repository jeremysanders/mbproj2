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

"""CmptMass objects define dark matter potentials."""

from __future__ import division, print_function, absolute_import

import math
import numpy as N
import scipy.special

from .param import Param
from .cmpt import Cmpt
from .physconstants import Mpc_km, G_cgs, Mpc_cm, km_cm, kpc_cm, solar_mass_g

class CmptMass(Cmpt):
    """Base mass component."""
    def __init__(self, name, annuli, suffix=''):
        """
        :param name: prefix for parameters
        :param Annuli annuli: annuli to examine
        :param suffix: suffix to add to name if set
        """
        if suffix:
            name = '%s_%s' % (name, suffix)
        Cmpt.__init__(self, name, annuli)

    def computeProf(self, pars):
        """Compute g_cmps2 and potential_ergpg profiles."""

class CmptMassNFW(CmptMass):
    """NFW profile.
    Useful detals here:
    http://nedwww.ipac.caltech.edu/level5/Sept04/Brainerd/Brainerd5.html
    and Lisa Voigt's thesis

    Model parameters are nfw_logconc (log10 concentration) and
    nfw_r200_logMpc (log10 r200 in Mpc)

    """

    def __init__(self, annuli, suffix=None):
        """
        :param Annuli annuli: Annuli object
        :param suffix: suffix to append to name nfw in parameters
        """
        CmptMass.__init__(self, 'nfw', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_logconc' % self.name: Param(0.3, minval=-2., maxval=2.),
            '%s_r200_logMpc' % self.name: Param(0., minval=-1., maxval=1.)
            }

    def computeProf(self, pars):
        c = 10**(pars['%s_logconc' % self.name].v)
        r200 = 10**(pars['%s_r200_logMpc' % self.name].v)
        radius_cm = self.annuli.massav_cm
        #radius_cm = self.annuli.midpt_cm

        # relationship between r200 and scale radius
        rs_Mpc = r200 / c

        # calculate characteristic overdensity of halo (using 200
        # times critical mass density)
        delta_c = (200/3) * c**3 / (math.log(1.+c) - c/(1+c))
        # Hubble's constant at z (km/s/Mpc)
        cosmo = self.annuli.cosmology
        Hz_km_s_Mpc = cosmo.H0 * math.sqrt(
            cosmo.WM*(1.+cosmo.z)**3 + cosmo.WV )
        # critical density at redshift of halo
        rho_c = 3. * ( Hz_km_s_Mpc / Mpc_km )**2 / (8 * math.pi * G_cgs)
        rho_0 = delta_c * rho_c

        # radius relative to scale radius
        x = radius_cm * (1/(rs_Mpc * Mpc_cm))

        # temporary quantities
        r_cube = (rs_Mpc * Mpc_cm)**3
        log_1x = N.log(1.+x)

        # mass enclosed within x
        mass = (4 * math.pi * rho_0) * r_cube * (log_1x - x/(1.+x))

        # gravitational acceleration
        g = G_cgs * mass / radius_cm**2

        # potential
        Phi = (-4 * math.pi * rho_0 * G_cgs) * r_cube * log_1x / radius_cm

        return g, Phi



class CmptMassNFWfnM(CmptMass):
    """NFW profile which is a function of mass at some overdensity

    Useful detals here:
    http://nedwww.ipac.caltech.edu/level5/Sept04/Brainerd/Brainerd5.html
    and Lisa Voigt's thesis

    This version parameterizes M (for some given overdensity) instead of R200

    Model parameters are nfw_logconc (log10 concentration) and
    nfw_M_logMsun (log10 Msun at overdensity)
    """

    def __init__(self, annuli, suffix=None, overdensity=500):
        """
        :param Annuli annuli: Annuli object
        :param suffix: suffix to append to name nfw in parameters
        """
        CmptMass.__init__(self, 'nfw', annuli, suffix=suffix)
        self.overdensity = overdensity

    def defPars(self):
        return {
            '%s_logconc' % self.name: Param(2., minval=-2., maxval=2.),
            '%s_M_logMsun' % self.name: Param(14., minval=13., maxval=16.)
        }

    def computeProf(self, pars):
        c = 10**(pars['%s_logconc' % self.name].v)
        M_X00_g = 10**(pars['%s_M_logMsun' % self.name].v) * solar_mass_g

        delta_c = (200/3) * c**3 / (math.log(1.+c) - c/(1+c))
        cosmo = self.annuli.cosmology
        Hz_km_s_Mpc = cosmo.H0 * math.sqrt(
            cosmo.WM*(1.+cosmo.z)**3 + cosmo.WV )
        # critical density at redshift of halo
        rho_c = 3. * ( Hz_km_s_Mpc / Mpc_km )**2 / (8 * math.pi * G_cgs)
        rho_0 = delta_c * rho_c

        # get enclosed mass given radius and rs
        def calc_mass_g(r_cm, rs_Mpc):
            rs_cm = rs_Mpc * Mpc_cm
            x = r_cm/rs_cm
            xp1 = x+1
            M_g = (4*math.pi*rho_0) * rs_cm**3 * (N.log(xp1) - x/xp1)
            return M_g

        # solve M_500=(4/3)*pi*rho*R_500**3 to get rs
        R_X00_cm = ( M_X00_g / (4/3*math.pi*rho_c*self.overdensity) )**(1/3)
        rs_Mpc = scipy.optimize.brentq(
            lambda r_Mpc: calc_mass_g(R_X00_cm, r_Mpc)-M_X00_g,
            0.0001, 1000)

        r_cm = self.annuli.massav_cm
        mass_g = calc_mass_g(r_cm, rs_Mpc)

        # gravitational acceleration
        g = G_cgs * mass_g / r_cm**2

        # potential
        x = r_cm / (rs_Mpc*Mpc_cm)
        Phi = (
            (-4 * math.pi * rho_0 * G_cgs) * (rs_Mpc*Mpc_cm)**3 *
            N.log(1+x) / r_cm
        )

        return g, Phi

class CmptMassGNFW(CmptMass):
    """Generalised NFW.

    This is an NFW with a free inner slope (alpha).

    rho(r) = rho0 / ( (r/rs)**alpha * (1+r/rs)**(3-alpha) )

    For details see Schmidt & Allen (2007)
    http://adsabs.harvard.edu/doi/10.1111/j.1365-2966.2007.11928.x

    Model parameters are gnfw_logconc (log10 concentration),
    gnfw_r200_logMpc (log10 r200 in Mpc) and gnfw_alpha (alpha
    parameter; 1 is standard NFW).

    """

    def __init__(self, annuli, suffix=None):
        """
        :param Annuli annuli: Annuli object
        :param suffix: suffix to append to name gnfw in parameters
        """
        CmptMass.__init__(self, 'gnfw', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_logconc' % self.name: Param(2., minval=-2., maxval=2.),
            '%s_r200_logMpc' % self.name: Param(0., minval=-1., maxval=1.),
            '%s_alpha' % self.name: Param(1., minval=0., maxval=2.5),
            }

    def computeProf(self, pars):
        # get parameter values
        c = 10**(pars['%s_logconc' % self.name].v)
        r200_Mpc = 10**(pars['%s_r200_logMpc' % self.name].v)
        alpha = pars['%s_alpha' % self.name].v

        # check to make sure funny things don't happen
        alpha = max(min(alpha, 2.999), 0.)

        # overdensity relative to critical density
        phi = c**(3-alpha) / (3-alpha) * scipy.special.hyp2f1(
            3-alpha, 3-alpha, 4-alpha, -c)
        delta_c = (200/3) * c**3 / phi

        # Hubble's constant at z (km/s/Mpc)
        cosmo = self.annuli.cosmology
        Hz_km_s_Mpc = cosmo.H0 * math.sqrt(
            cosmo.WM*(1.+cosmo.z)**3 + cosmo.WV )

        # critical density at redshift of halo
        rho_c = 3 * ( Hz_km_s_Mpc / Mpc_km )**2 / (8 * math.pi * G_cgs)
        rho_0 = delta_c * rho_c

        # scale radius
        rs_cm = r200_Mpc * Mpc_cm / c

        # radius of shells relative to scale radius
        x = self.annuli.massav_cm * (1/rs_cm)

        # gravitational acceleration
        g = (
            4 * math.pi * rho_0 * G_cgs / (3-alpha) * rs_cm * x**(1-alpha) *
            scipy.special.hyp2f1(3-alpha, 3-alpha, 4-alpha, -x)
            )

        # potential
        Phi = (
            4 * math.pi * rho_0 * G_cgs / (alpha-2) * rs_cm**2 * (
                1 + -x**(2-alpha) / (3-alpha) *
                scipy.special.hyp2f1(3-alpha, 2-alpha, 4-alpha, -x) )
            )

        return g, Phi

class CmptMassKing(CmptMass):
    """King potential.

    This is the modified Hubble potential, where
    rho = rho0 / (1+(r/r0)**2)**1.5

    r0 = sqrt(9*sigma**2/(4*Pi*G*rho0))

    We define rho in terms of r0 and sigma

    Model parameters are king_sigma_logkmps (sigma in log10 km/s)
    and king_rcore_logkpc (rcore in log10 kpc).

    """

    def __init__(self, annuli, suffix=None):
        """
        :param Annuli annuli: Annuli object
        :param suffix: suffix to append to name king in parameters
        """
        CmptMass.__init__(self, 'king', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_sigma_logkmps' % self.name: Param(2.5, minval=0., maxval=3.7),
            '%s_rcore_logkpc' % self.name: Param(1.3, minval=-1., maxval=3.4)
            }

    def computeProf(self, pars):
        sigma_cmps = 10**(pars['%s_sigma_logkmps' % self.name].v) * km_cm
        r0 = 10**(pars['%s_rcore_logkpc' % self.name].v) * kpc_cm

        # calculate central density from r0 and sigma
        rho0 = 9*sigma_cmps**2 / (4 * math.pi * G_cgs * r0**2)
        r = self.annuli.massav_cm

        # this often occurs below, so precalculate
        rsqrtfac = N.sqrt(r**2 + r0**2)

        g = (G_cgs/r**2)*(4*math.pi*r0**3*rho0) * (
            -r / rsqrtfac +
             N.arcsinh(r/r0))

        # taken from isothermal.nb
        phi = ( -8 * G_cgs * math.pi * (r0/r)**3 * (
                (r*N.sqrt(((r**2 + r0**2)*(-r0 + rsqrtfac))/
                          (r0 + rsqrtfac)) +
                 r0*N.sqrt(r**2 + 2*r0*(r0 - rsqrtfac)))* rho0 *
                N.arcsinh(N.sqrt(-1./2 + 0.5*N.sqrt(1 + r**2/r0**2))) ))

        return g, phi

class CmptMassPoint(CmptMass):
    """Point mass.

    Model parameter is pt_M_logMsun, which is point mass in log solar
    masses.

    """

    def __init__(self, annuli, suffix=None):
        """
        annuli: Annuli object
        suffix: suffix to append to name pt in parameters
        """
        CmptMass.__init__(self, 'pt', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_M_logMsun' % self.name: Param(12., minval=9., maxval=14.),
            }

    def computeProf(self, pars):
        mass_g = 10**(pars['%s_M_logMsun' % self.name].v) * solar_mass_g

        r = self.annuli.massav_cm
        g = G_cgs * mass_g / r**2
        phi = -G_cgs * mass_g / r
        return g, phi

class CmptMassArb(CmptMass):
    """Parametrise mass density using interpolation.

    Model parameters are arb_rho_YYY (mass density in log10 g cm^-3)
    and arb_r_YYY (radii in log10 kpc) for annuli YYY.

    """

    def __init__(self, annuli, nradbins, suffix=None):
        """
        :param Annuli annuli: annuli used
        :param suffix: suffix to append to name arb in parameters
        """
        CmptMass.__init__(self, 'arb', annuli, suffix=suffix)
        self.nradbins = nradbins

        # list of all the parameter names for the annuli
        self.valparnames = ['%s_rho_%03i' % (self.name, i) for i in range(nradbins)]
        self.radparnames = ['%s_r_%03i' % (self.name, i) for i in range(nradbins)]

    def defPars(self):
        rlogannuli = self.annuli.midpt_logkpc
        rlog = N.linspace(rlogannuli[0], rlogannuli[-1], self.nradbins)
        rpars = {
            n: Param(r, minval=rlogannuli[0], maxval=rlogannuli[-1], frozen=True)
            for n, r in zip(self.radparnames, rlog)
            }

        valspars = {
            n: Param(self.defval, minval=-15., maxval=4.)
            for n in self.valparnames
            }

        # combined parameters
        valspars.update(rpars)
        return valspars

    def computeProf(self, pars):
        rvals = N.array([pars[n].v for n in self.radparnames])
        rhovals = N.array([pars[n].v for n in self.valparnames])

        # radii might be in wrong order
        sortidxs = N.argsort(rvals)
        rvals = rvals[sortidxs]
        vvals = vvals[sortidxs]

        # rgrid spanning over range of annuli (and a little inside)
        logannkpc = self.annuli.massav_logkpc
        rgrid_logkpc = N.linspace(logannkpc[0]-0.7, logannkpc[-1], 256)
        rgrid_cent_logkpc = 0.5*(rgrid_logkpc[1:] + rgrid_logkpc[:-1])
        rgrid_cm = 10**rgrid_logkpc * kpc_cm

        # do interpolating, extending beyond
        # this is the gradient between each points
        grads = (vvals[1:]-vvals[:-1]) / (rvals[1:]-rvals[:-1])
        # index to point below this one (truncating if necessary)
        idx = N.searchsorted(rvals, rgrid_cent_logkpc)-1
        idx = N.clip(idx, 0, len(grads)-1)
        # calculate line from point using gradient to next point
        dr = rgrid_cent_logkpc - rvals[idx]
        rho = vvals[idx] + dr*grads[idx]
        rho = 10**rho

        # compute mass in shells
        vols_cm3 = (4./3) * N.pi * N.ediff1d(rgrid_cm**3)
        Mshell_g = rho * vols_cm3
        # cumulative log mass in shells
        Mcuml_logg = N.log(N.cumsum(Mshell_g))
        # do interpolation in log space to get total mass
        mass_g = N.exp(M.interp(logannkpc, rgrid_cent_logkpc, Mcuml_logg))

        r = self.annuli.massav_cm
        g = G_cgs * mass_g / r**2
        phi = -G_cgs * mass_g / r
        return g, phi

class CmptMassMulti(CmptMass):
    """Multi-component mass profile made up CmptMass objects."""

    def __init__(self, name, annuli, cmpts, suffix=None):
        """
        :param name: name of component
        :param Annuli annuli: annuli to use
        :param list[CmptMass] cmpts: components to sum
        """
        CmptMass.__init__(self, name, annuli, suffix=suffix)
        self.cmpts = cmpts

    def defPars(self):
        retn = {}
        for cmpt in self.cmpts:
            retn.update(cmpt.defPars())
        return retn

    def computeProf(self, pars):
        tot_g, tot_pot = 0., 0.
        for cmpt in self.cmpts:
            g, pot = cmpt.computeProf(pars)
            tot_g += g
            tot_pot += pot
        return tot_g, tot_pot

    def prior(self, pars):
        tot = 0.
        for cmpt in self.cmpts:
            tot += cmpt.prior(pars)
        return tot

class CmptMassEinasto(CmptMass):
    """Einasto profile.

    Following: https://doi.org/10.1051/0004-6361/201118543
    Retana-Montenegro1, Van Hese, Gentile, Baes and F. Frutos-Alfaro
    2012, A&A, 540, A70
    """

    def __init__(self, annuli, suffix=None):
        CmptMass.__init__(self, 'einasto', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_Mtot_logMsun' % self.name: Param(14., minval=12., maxval=16),
            '%s_n' % self.name: Param(4.35, minval=0, maxval=20),
            '%s_rs_logMpc' % self.name: Param(0, minval=-1.3, maxval=0.7),
        }

    def computeProf(self, pars):

        n = pars['%s_n' % self.name].v
        Mtot = 10**(pars['%s_Mtot_logMsun' % self.name].v) * solar_mass_g
        rs = 10**(pars['%s_rs_logMpc' % self.name].v) * Mpc_cm
        r = self.annuli.massav_cm

        # constant to ensure rs is radius containing half total mass
        # (solves 2*Gamma(3n,d_n)/Gamma(3n) = 1)
        d_n = (
            - 17557576/(1242974068875*n**4)
            + 1048/(31000725*n**3)
            + 184/(229635*n*n)
            + 8/(1215*n)
            - 1/3
            + 3*n
        )

        # scale length
        h = rs / d_n**n

        # central density (not required)
        # rho0 = Mtot/(4*math.pi * h**3 * n * scipy.special.gamma(3*n))

        # reduced radius
        s = r * (d_n**n/rs)

        # precompute values used for M_r and Phi_r
        s_pow_inv_n = s**(1/n)
        Gamma_P_3n = scipy.special.gammainc(3*n, s_pow_inv_n)
        Gamma_Q_2n = scipy.special.gammaincc(2*n, s_pow_inv_n)
        # Gamma(2n)/Gamma(3n)
        Gamma_2n_3n = math.exp(
            scipy.special.gammaln(2*n)-scipy.special.gammaln(3*n))
        inv_r = 1/r

        # cumulative mass as a function of radius (note Gamma_P=1-Gamma_Q)
        M_r = Mtot * Gamma_P_3n

        # acceleration
        g_r = G_cgs * M_r * inv_r**2

        # potential (note sign changed from eqn 19 in above paper)
        Phi_r = -G_cgs * Mtot * inv_r * (
            Gamma_P_3n +
            s * Gamma_Q_2n * Gamma_2n_3n
        )

        return g_r, Phi_r
