from __future__ import division, print_function

import math
import numpy as N
from fit import Param
from cmpt import Cmpt
from physconstants import Mpc_km, G_cgs, Mpc_cm, km_cm, kpc_cm

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
    """

    def __init__(self, annuli, suffix=None):
        CmptMass.__init__(self, 'nfw', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_conc' % self.name: Param(2., minval=0.01, maxval=100.),
            '%s_r200_Mpc' % self.name: Param(2., minval=0.1, maxval=10.)
            }

    def computeProf(self, pars):
        """Compute g_cmps2 and potential_ergpg profiles."""

        c = pars['%s_conc' % self.name].val
        r200 = pars['%s_r200_Mpc' % self.name].val
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

class CmptMassKing(CmptMass):
    """King potential.

    This is the modified Hubble potential, where
    rho = rho0 / (1+(r/r0)**2)**1.5

    r0 = sqrt(9*sigma**2/(4*Pi*G*rho0))

    We define rho in terms of r0 and sigma
    """

    def __init__(self, annuli, suffix=None):
        CmptMass.__init__(self, 'king', annuli, suffix=suffix)

    def defPars(self):
        return {
            '%s_sigma_kmps' % self.name: Param(600., minval=1, maxval=5000.),
            '%s_rcore_kpc' % self.name: Param(20., minval=0.1, maxval=5000.)
            }

    def computeProf(self, pars):
        sigma_cmps = pars['%s_sigma_kmps' % self.name].val * km_cm
        r0 = pars['%s_rcore_kpc' % self.name].val * kpc_cm

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

    def computeProfs(self, pars):
        tot_g, tot_pot = 0, 0
        for cmpt in self.cmpts:
            g, pot = cmpt.computeProfs(pars)
            tot_g += g
            tot_pot += pot
        return tot_g, tot_pot
