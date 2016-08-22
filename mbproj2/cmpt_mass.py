from __future__ import division, print_function

import math
import numpy as N
import scipy.special

from fit import Param
from cmpt import Cmpt
from physconstants import Mpc_km, G_cgs, Mpc_cm, km_cm, kpc_cm, solar_mass_g

class CmptMass(Cmpt):
    def __init__(self, name, annuli, suffix=''):
        if suffix:
            name = '%s_%s' % (name, suffix)
        Cmpt.__init__(self, name, annuli)

class CmptMassNFW(CmptMass):
    """NFW profile.
    Useful detals here:
    http://nedwww.ipac.caltech.edu/level5/Sept04/Brainerd/Brainerd5.html
    and Lisa Voigt's thesis

    Parameters:
    nfw_logconc: log10 concentration
    nfw_r200_logMpc: log10 r200 in Mpc
    """

    def __init__(self, annuli, suffix=None):
        CmptMass.__init__(self, 'nfw', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_logconc' % self.name: Param(2., minval=-2., maxval=2.),
            '%s_r200_logMpc' % self.name: Param(0., minval=-1., maxval=1.)
            }

    def computeProf(self, pars):
        """Compute g_cmps2 and potential_ergpg profiles."""

        c = 10**(pars['%s_logconc' % self.name].val)
        r200 = 10**(pars['%s_r200_logMpc' % self.name].val)
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

class CmptMassGNFW(CmptMass):
    """Generalised NFW.

    This is an NFW with a free inner slope (alpha).

    rho(r) = rho0 / ( (r/rs)**alpha * (1+r/rs)**(3-alpha) )

    For details see Schmidt & Allen (2007)
    http://adsabs.harvard.edu/doi/10.1111/j.1365-2966.2007.11928.x

    Parameters:
    gnfw_logconc: log10 concentration
    gnfw_r200_logMpc: log10 r200 in Mpc
    gnfw_alpha: alpha parameter (1 is standard NFW)
    """

    def __init__(self, annuli, suffix=None):
        CmptMass.__init__(self, 'gnfw', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_logconc' % self.name: Param(2., minval=-2., maxval=2.),
            '%s_r200_logMpc' % self.name: Param(0., minval=-1., maxval=1.),
            '%s_alpha' % self.name: Param(1., minval=0., maxval=2.5),
            }

    def computeProf(self, pars):
        """Compute g_cmps2 and potential_ergpg profiles."""

        # get parameter values
        c = 10**(pars['%s_logconc' % self.name].val)
        r200_Mpc = 10**(pars['%s_r200_logMpc' % self.name].val)
        alpha = pars['%s_alpha' % self.name].val

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

    Parameters:
    king_sigma_logkmps: sigma in log10 km/s
    king_rcore_logkpc: rcore in log10 kpc
    """

    def __init__(self, annuli, suffix=None):
        CmptMass.__init__(self, 'king', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_sigma_logkmps' % self.name: Param(2.5, minval=0., maxval=3.7),
            '%s_rcore_logkpc' % self.name: Param(1.3, minval=-1., maxval=3.4)
            }

    def computeProf(self, pars):
        sigma_cmps = 10**(pars['%s_sigma_logkmps' % self.name].val) * km_cm
        r0 = 10**(pars['%s_rcore_logkpc' % self.name].val) * kpc_cm

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
    """Point mass."""

    def __init__(self, annuli, suffix=None):
        CmptMass.__init__(self, 'pt', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_M_logMsun' % self.name: Param(12., minval=10., maxval=14.),
            }

    def computeProf(self, pars):
        mass_g = 10**(pars['%s_M_logMsun' % self.name].val) * solar_mass_g

        r = self.annuli.massav_cm
        g = G_cgs * mass_g / r**2
        phi = -G_cgs * mass_g / r
        return g, phi

class CmptMassMulti(CmptMass):
    """Multi-component mass profile."""

    def __init__(self, name, annuli, cmpts, suffix=None):
        CmptMass.__init__(self, name, annuli, suffix=suffix)
        self.cmpts = cmpts

    def defPars(self):
        retn = {}
        for cmpt in self.cmpts:
            retn.update(cmpt.defPars())
        return retn

    def computeProf(self, pars):
        tot_g, tot_pot = 0, 0
        for cmpt in self.cmpts:
            g, pot = cmpt.computeProf(pars)
            tot_g += g
            tot_pot += pot
        return tot_g, tot_pot
