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

"""Define hydrostatic and non-hydrostatic models.

ModelNullPot: non-hydrostatic model, equivalent to spectral fitting
ModelHydro: hydrostatic model, parameterized by density profile
"""

from __future__ import division, print_function, absolute_import
import math

from six.moves import range
import numpy as N

from .fit import Param
from .physconstants import ne_nH, mu_g, mu_e, P_keV_to_erg, G_cgs

class Model:
    """Base class for different models.

    The parameter to the models are provided by a dict mapping the
    parameter name to a Param object.

    """

    def __init__(self, annuli, NH_1022pcm2=None):
        """
        :param Annuli annuli: annuli on sky
        :param float NH_1022pcm2: absorbing column density in 10^22 cm^-2
        """
        self.annuli = annuli
        assert NH_1022pcm2 is not None
        self.NH_1022pcm2 = NH_1022pcm2

    def defPars(self):
        """
        :rtype: dict[str,Param]
        :return: default dict of parameters (names to Param objects).
        """

    def computeProfs(self, pars):
        """Compute profiles of physical parameters.

        :type pars: dict[str, Param]
        :param pars: parameter values
        :returns: arrays of ne, T and Z
        """

    def computeMassProf(self, pars):
        """Compute mass profile.

        :type pars: dict[str, Param]
        :param pars: input parameters
        :returns: g and potential arrays (CGS; both can be 0)
        """

    def prior(self, pars):
        """Compute prior, given parameters.

        Returns log likelihood."""
        return 0.

class ModelNullPot(Model):
    """This is a form of the model without any gravitational
    potential parameters.

    Density, temperature and abunance are all separately fit.
    """

    def __init__(self, annuli, ne_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=None):
        """
        :param Annuli annuli: annuli to analyse
        :param Cmpt ne_cmpt: density component
        :param Cmpt T_cmpt: temperature component
        :param Cmpt Z_cmpt: metallicity component
        :param float NH_1022pcm2: absorbing column density in 10^22 cm^-2
        """
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

    def prior(self, pars):
        return (
            self.T_cmpt.prior(pars) +
            self.ne_cmpt.prior(pars) +
            self.Z_cmpt.prior(pars)
            )

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

    The Pout_logergpcm3 parameter is the outer pressure in log10 erg/cm^3.
    """

    def __init__(self, annuli, mass_cmpt, ne_cmpt, Z_cmpt, NH_1022pcm2=None):
        """
        :param Annuli annuli: annuli to analyse
        :param CmptMass mass_cmpt: dark matter mass component
        :param Cmpt ne_cmpt: density component
        :param Cmpt Z_cmpt: metallicity component
        :param float NH_1022pcm2: absorbing column density in 10^22 cm^-2
        """
        Model.__init__(self, annuli, NH_1022pcm2=NH_1022pcm2)
        self.mass_cmpt = mass_cmpt
        self.ne_cmpt = ne_cmpt
        self.Z_cmpt = Z_cmpt

    def defPars(self):
        pars = {'Pout_logergpcm3': Param(-15., minval=-20., maxval=-8.)}
        pars.update(self.mass_cmpt.defPars())
        pars.update(self.ne_cmpt.defPars())
        pars.update(self.Z_cmpt.defPars())
        return pars

    def computeProfs(self, pars):
        """Calculate profiles assuming hydrostatic equilibrium.

        :returns: ne_pcm3, T_keV, Z_solar
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

        # changes in pressure in outer and inner halves of bin (around massav)
        ptmp = g_cmps2 * ne_pcm3 * (mu_e*mu_g)
        deltah_out = self.annuli.edges_cm[1:] - self.annuli.massav_cm
        deltaP_out = deltah_out * ptmp
        deltah_in = self.annuli.massav_cm - self.annuli.edges_cm[:-1]
        deltaP_in = deltah_in * ptmp

        # combine halves and include outer pressure
        deltaP_halves = N.ravel( N.column_stack((deltaP_in, deltaP_out)) )
        deltaP_ergpcm3 = N.concatenate((deltaP_halves[1:], [P0_ergpcm3]))

        # add up contributions inwards to get total pressure,
        # discarding pressure between shells
        P_ergpcm3 = N.cumsum(deltaP_ergpcm3[::-1])[::-2]

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

    def prior(self, pars):
        return (
            self.mass_cmpt.prior(pars) +
            self.ne_cmpt.prior(pars) +
            self.Z_cmpt.prior(pars)
            )

class ModelHydroEntropy(Model):
    """This is a form of the model assuming hydrostatic equilibrium,
    but parameterising using the entropy (K), not density.

    As the gas mass can't be calculated directly, the routine iterates
    a number of times to get the density, updating the potential each
    time.

    The Pout_logergpcm3 parameter is the outer pressure in log10 erg/cm^3.

    """

    def __init__(
        self, annuli, mass_cmpt, K_cmpt, Z_cmpt, NH_1022pcm2=None, self_gravity=True):
        """
        :param Annuli annuli: annuli to analyse
        :param CmptMass mass_cmpt: dark matter mass component
        :param Cmpt K_cmpt: density component
        :param Cmpt Z_cmpt: metallicity component
        :param float NH_1022pcm2: absorbing column density in 10^22 cm^-2
        :param float self_gravity: include loop to iteratively calculate self-gravity of baryonic mass
        """

        Model.__init__(self, annuli, NH_1022pcm2=NH_1022pcm2)
        self.mass_cmpt = mass_cmpt
        self.K_cmpt = K_cmpt
        self.Z_cmpt = Z_cmpt
        self.self_gravity = self_gravity

    def defPars(self):
        pars = {'Pout_logergpcm3': Param(-13., minval=-16., maxval=0.)}
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
        for i in range(iters):
            # add (small) gas contribution to total acceleration
            tot_g_cmps2 = g_cmps2 + computeGasAccn(self.annuli, ne_pcm3)

            ne_pcm3 = []
            P_ergpcm3 = P0_ergpcm3
            for i in range(self.annuli.nshells-1, -1, -1):
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

    def prior(self, pars):
        return (
            self.mass_cmpt.prior(pars) +
            self.K_cmpt.prior(pars) +
            self.Z_cmpt.prior(pars)
            )
