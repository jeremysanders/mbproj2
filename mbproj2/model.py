# Clusters models built from components. Includes models which assume
# hydrostatic equilibrium and those that don't

from __future__ import division, print_function

from fit import Param
import math
import numpy as N
from physconstants import ne_nH, mu_g, mu_e, P_keV_to_erg, G_cgs

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

def computeGasAccn(annuli, ne_prof):
    """Compute acceleration due to gas mass for density profile
    given."""

    # mass in each shell
    masses_g = ne_prof * annuli.vols_cm3 * (mu_e * mu_g)

    # cumulative mass interior to each shell
    Minterior_g = N.cumsum( N.hstack( ([0.], masses_g[:-1]) ) )

    # this is the mean acceleration on the shell, computed as total
    # force from interior mass divided by the total mass:
    #   ( Int_{r=R1}^{R2} (G/r**2) *
    #                     (M + Int_{R=R1}^{R} 4*pi*R^2*rho*dR) *
    #                     4*pi*r^2*rho*dR ) / (
    #   (4./3.*pi*(R2**3-R1**3)*rho)
    rout, rin = annuli.rout_cm, annuli.rin_cm
    gmean = G_cgs*(
        3*Minterior_g +
        ne_prof*(mu_e*mu_g*math.pi)*(
            (rout-rin)*((rout+rin)**2 + 2*rin**2)))  / (
        rin**2 + rin*rout + rout**2 )

    return gmean

class ModelHydro(Model):
    """This is a form of the model assuming hydrostatic
    equilibrium. The temperature is calculated from the density and
    pressure computed from the mass model.

    Temperature is calculated assuming hydrostatic equilibrium.
    Included parameter is the outer pressure Pout
    """

    def __init__(self, annuli, mass_cmpt, ne_cmpt, Z_cmpt, NH_1022pcm2=None):
        Model.__init__(self, annuli, NH_1022pcm2=NH_1022pcm2)
        self.mass_cmpt = mass_cmpt
        self.ne_cmpt = ne_cmpt
        self.Z_cmpt = Z_cmpt

    def defPars(self):
        """Return default model parameters dict."""
        pars = {'Pout_logergpcm3': Param(-15., minval=-30., maxval=0.)}
        pars.update(self.mass_cmpt.defPars())
        pars.update(self.ne_cmpt.defPars())
        pars.update(self.Z_cmpt.defPars())
        return pars

    def computeProfs(self, pars):
        """Calculate profiles assuming hydrostatic equilibrium.

        Returns ne_pcm3, T_keV, Z_solar
        """

        # this is the outer pressure
        P0_ergpcm3 = 10**pars['Pout_logergpcm3'].val

        # this is acceleration and potential from mass model
        g_cmps2, pot_ergpg = self.mass_cmpt.computeProf(pars)

        # input density and abundance profiles
        ne_pcm3 = self.ne_cmpt.computeProf(pars)
        # avoid hydrostatic equilibrium blowing up below
        ne_pcm3 = N.clip(ne_pcm3, 1e-99, 1e99)

        # clipped metallicity
        Z_solar = self.Z_cmpt.computeProf(pars)
        Z_solar = N.clip(Z_solar, 0, 1e99)

        # add (small) gas contribution to total acceleration
        g_cmps2 += computeGasAccn(self.annuli, ne_pcm3)

        # changes in pressure in each bin due to hydrostatic eqbm (h*rho*g)
        deltaP_bins = self.annuli.widths_cm * g_cmps2 * ne_pcm3 * (mu_e * mu_g)
        deltaP_ergpcm3 = N.concatenate((deltaP_bins[1:], [P0_ergpcm3]))

        # add up contributions inwards to get total pressure
        P_ergpcm3 = N.cumsum(deltaP_ergpcm3[::-1])[::-1]

        # calculate temperatures given pressures ad densities
        T_keV = P_ergpcm3 / (P_keV_to_erg * ne_pcm3)

        return ne_pcm3, T_keV, Z_solar

    def computeMassProf(self, pars):
        """Compute g and potential given parameters."""

        g_prof, pot_prof = self.mass_cmpt.computeProf(pars)

        # add accn due to gas to total acceleration
        ne_prof = self.ne_cmpt.computeProf(pars)
        g_prof += computeGasAccn(self.annuli, ne_prof)

        return g_prof, pot_prof

class ModelHydroEntropy(Model):
    """This is a form of the model assuming hydrostatic equilibrium,
    but parameterising using the entropy (K), not density.

    Temperature is calculated assuming hydrostatic equilibrium.
    Included parameter is the outer pressure Pout
    """

    def __init__(
        self, annuli, mass_cmpt, K_cmpt, Z_cmpt, NH_1022pcm2=None, self_gravity=True):

        Model.__init__(self, annuli, NH_1022pcm2=NH_1022pcm2)
        self.mass_cmpt = mass_cmpt
        self.K_cmpt = K_cmpt
        self.Z_cmpt = Z_cmpt
        self.self_gravity = self_gravity

    def defPars(self):
        pars = {'Pout_logergpcm3': Param(-15., minval=-30., maxval=0.)}
        pars.update(self.mass_cmpt.defPars())
        pars.update(self.K_cmpt.defPars())
        pars.update(self.Z_cmpt.defPars())
        return pars

    def iterateComputeProfs(self, pars):
        """Iteratively compute output profiles.

        This is iterative because if we want to include gas self
        gravity, this changes g."""

        # this is the outer pressure
        P0_ergpcm3 = 10**pars['Pout_logergpcm3'].val

        # compute the entropy and clip to avoid numerical issues
        Ke_keVcm2 = self.K_cmpt.computeProf(pars)
        Ke_keVcm2 = N.clip(Ke_keVcm2, 1e-99, 1e99)

        # this is acceleration and potential from mass model
        g_cmps2, pot_ergpg = self.mass_cmpt.computeProf(pars)

        # do we bother working out the effects of self-gravity?  we
        # loop round several times if we want to - TODO: check whether
        # 4 is a reasonable number
        iters = 4 if self.self_gravity else 1

        # initial density for iteration
        ne_pcm3 = N.zeros(self.annuli.nshells)

        # repeatedly calculate density, then include self gravity
        for i in xrange(iters):
            # add (small) gas contribution to total acceleration
            tot_g_cmps2 = g_cmps2 + computeGasAccn(self.annuli, ne_pcm3)

            ne_pcm3 = []
            P_ergpcm3 = P0_ergpcm3
            for i in xrange(self.annuli.nshells-1, -1, -1):
                ne = (P_ergpcm3 / P_keV_to_erg / Ke_keVcm2[i])**(3./5.)
                P_ergpcm3 += self.annuli.widths_cm[i] * tot_g_cmps2[i] * ne * (mu_e * mu_g)
                ne_pcm3.insert(0, ne)
            ne_pcm3 = N.array(ne_pcm3)

        T_keV = ne_pcm3**(2./3.) * Ke_keVcm2

        return ne_pcm3, T_keV, tot_g_cmps2, pot_ergpg

    def computeProfs(self, pars):
        """Calculate profiles assuming hydrostatic equilibrium."""

        ne_prof, T_prof, g_prof, pot_prof = self.iterateComputeProfs(pars)

        # clipped metallicity
        Z_prof = N.clip(self.Z_cmpt.computeProf(pars), 0., 1e99)

        return ne_prof, T_prof, Z_prof

    def computeMassProf(self, pars):
        """Compute g and potential given parameters."""
        ne_prof, T_prof, g_prof, pot_prof = self.iterateComputeProfs(pars)
        return g_prof, pot_prof
