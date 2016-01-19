from __future__ import division, print_function

from fit import Param
import math
import numpy as N
from itertools import izip
from physconstants import ne_nH, mu_g, mu_e, P_ne_to_T, G_cgs

class Model:
    def __init__(self, annuli, NH_1022pcm2=None):
        """Initialise Model. annuli is an Annuli object."""
        self.annuli = annuli
        assert NH_1022pcm2 is not None
        self.NH_1022pcm2 = NH_1022pcm2

    def defPars(self):
        """Return dict of parameters (Param objects)."""

    def computeProfs(self, pars):
        """Take the dict of parameter values and return
        arrays of ne, T and Z.
        """

    def computeMassProf(self, pars):
        """Compute mass profile, given pars.
        Returns: g, potential (both can be 0)
        """

class ModelNullPot(Model):
    """This is a form of the model without any gravitational
    potential parameters.

    Density, temperature and abunance are all separately fit.
    """

    def __init__(self, annuli, ne_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=None):
        Model.__init__(self, annuli, NH_1022pcm2=NH_1022pcm2)
        self.ne_cmpt = ne_cmpt
        self.T_cmpt = T_cmpt
        self.Z_cmpt = Z_cmpt

    def defPars(self):
        pars = self.ne_cmpt.defPars()
        pars.update(self.T_cmpt.defPars())
        pars.update(self.Z_cmpt.defPars())
        return pars

    def computeProfs(self, pars):
        ne_prof = self.ne_cmpt.computeProf(pars)
        T_prof = self.T_cmpt.computeProf(pars)
        Z_prof = self.Z_cmpt.computeProf(pars)
        return ne_prof, T_prof, Z_prof

    def computeMassProf(self, pars):
        return 0*self.annuli.midpt_cm, 0*self.annuli.midpt_cm

class ModelHydro(Model):
    """This is a form of the model assuming hydrostatic
    equilibrium.

    Temperature is calculated assuming hydrostatic equilibrium.
    Included parameter is the outer pressure Pout
    """

    def __init__(self, annuli, mass_cmpt, ne_cmpt, Z_cmpt, NH_1022pcm2=None):
        Model.__init__(self, annuli, NH_1022pcm2=NH_1022pcm2)
        self.mass_cmpt = mass_cmpt
        self.ne_cmpt = ne_cmpt
        self.Z_cmpt = Z_cmpt

    def defPars(self):
        pars = {'Pout_logergpcm3': Param(-15., minval=-30., maxval=0.)}
        pars.update(self.mass_cmpt.defPars())
        pars.update(self.ne_cmpt.defPars())
        pars.update(self.Z_cmpt.defPars())
        return pars

    def computeGasAccn(self, ne_prof):
        """Compute acceleration due to gas mass for density profile given.
        """

        # mass in each shell
        masses_g = ne_prof * self.annuli.vols_cm3 * (mu_e * mu_g)

        # cumulative mass interior to each shell
        Minterior_g = N.cumsum( N.hstack( ([0.], masses_g[:-1]) ) )

        # this is the mean acceleration on the shell, computed as total
        # force from interior mass divided by the total mass:
        #   ( Int_{r=R1}^{R2} (G/r**2) *
        #                     (M + Int_{R=R1}^{R} 4*pi*R^2*rho*dR) *
        #                     4*pi*r^2*rho*dR ) / (
        #   (4./3.*pi*(R2**3-R1**3)*rho)
        rout, rin = self.annuli.rout_cm, self.annuli.rin_cm
        gmean = G_cgs*(
            3*Minterior_g +
            ne_prof*(mu_e*mu_g*math.pi)*(
                (rout-rin)*((rout+rin)**2 + 2*rin**2)))  / (
            rin**2 + rin*rout + rout**2 )

        return gmean

    def computeProfs(self, pars):
        # this is the outer pressure
        P0_ergpcm3 = 10**pars['Pout_logergpcm3'].val

        # this is acceleration and potential from mass model
        g_prof, pot_prof = self.mass_cmpt.computeProf(pars)

        # input density and abundance profiles
        ne_prof = self.ne_cmpt.computeProf(pars)
        # avoid hydrostatic equilibrium blowing up below
        ne_prof = N.clip(ne_prof, 1e-99, 1e99)
        Z_prof = self.Z_cmpt.computeProf(pars)
        # metallicity cannot be -ve
        Z_prof = N.clip(Z_prof, 0, 1e99)

        # add to total acceleration
        g_prof += self.computeGasAccn(ne_prof)

        # now compute temperatures from hydrostatic equilibrium by
        # iterating in reverse over the shells
        T_prof = []
        P_ergpcm3 = P0_ergpcm3
        for ne_pcm3, width_cm, g_cmps2 in izip(
            ne_prof[::-1], self.annuli.widths_cm[::-1], g_prof[::-1]):

            T_keV = P_ergpcm3 / (P_ne_to_T * ne_pcm3)
            T_prof.insert(0, T_keV)
            P_ergpcm3 += width_cm * g_cmps2 * ne_pcm3 * (mu_e * mu_g)

        T_prof = N.array(T_prof)

        return ne_prof, T_prof, Z_prof

    def computeMassProf(self, pars):
        """Compute g and potential given parameters."""

        g_prof, pot_prof = self.mass_cmpt.computeProf(pars)

        # add accn due to gas to total acceleration
        ne_prof = self.ne_cmpt.computeProf(pars)
        g_prof += self.computeGasAccn(ne_prof)

        return g_prof, pot_prof
