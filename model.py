import math
from itertools import izip
from physconstants import ne_nH, mu_g, mu_e, boltzmann_erg_K, keV_K, G_cgs

class Model:
    def defPars(self):
        """Return list of default parameters, in form
        [('name1', val1), ('name2', val2)]
        """

    def computeProfs(self, annuli, pars):
        """Take the array of parameter values and return
        arrays of ne, T and Z.

        annuli is an Annuli object
        """

    def numPars(self):
        """Return number of parameters."""
        return 0

class ModelNullPot(Model):
    """This is a form of the model without any gravitational
    potential parameters.

    Density, temperature and abunance are all separately fit.
    """

    def __init__(self, ne_model, T_model, Z_model):
        self.ne_model = ne_model
        self.T_model = T_model
        self.Z_model = Z_model

        self.ne_npars = ne_model.numPars()
        self.T_npars = T_model.numPars()
        self.Z_npars = Z_model.numPars()

    def defPars(self):
        return (
            self.ne_model.defPars() +
            self.T_model.defPars() +
            self.Z_model.defPars() )

    def numPars(self):
        return self.ne_npars + self.T_npars + self.Z_npars

    def computeProfs(self, annuli, pars):
        ne_prof = self.ne_model.computeProf(
            annuli, pars[:self.ne_npars])
        T_prof = self.T_model.computeProf(
            annuli, pars[self.ne_npars:self.ne_npars+self.T_npars])
        Z_prof = self.Z_model.computeProf(
            annuli, pars[self.ne_npars+self.T_npars:])
        return ne_prof, T_prof, Z_prof

class ModelHydro(Model):
    """This is a form of the model assuming hydrostatic
    equilibrium.

    Temperature is calculated assuming hydrostatic equilibrium.
    Included parameter is the outer pressure Pout
    """

    def __init__(self, mass_model, ne_model, Z_model):
        self.mass_model = mass_model
        self.ne_model = ne_model
        self.Z_model = Z_model

        self.mass_npars = mass_model.numPars()
        self.ne_npars = ne_model.numPars()
        self.Z_npars = Z_model.numPars()

    def defPars(self):
        return (
            [('Pout_logergpcm3', -15.)] +
            self.mass_model.defPars() +
            self.ne_model.defPars() +
            self.Z_model.defPars() )

    def numPars(self):
        return 1 + self.mass_npars + self.ne_npars + self.Z_npars

    def computeProfs(self, annuli, pars):
        P0_ergpcm3 = 10**pars[0]

        # this is acceleration and potential from mass model
        accn, pot = self.mass_model.computeProf(
            annuli, pars[1:1+self.mass_npars])

        # input density and abundance profiles
        ne_prof = self.ne_model.computeProf(
            annuli,
            pars[1+self.mass_npars:1+self.mass_npars+self.ne_npars])
        Z_prof = self.Z_model.computeProf(
            annuli,
            pars[1+self.mass_npars+self.ne_npars:])

        # compute acceleration from gas
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
        rout_cm, rin_cm = annuli.edges_cm[1:], annuli.edges_cm[:-1]
        gmean = G_cgs*(
            3*Minterior_g +
            ne_pcm3*(mu_e*mu_g*math.pi)*(
                (rout_cm-rin_cm)*((rout_cm+rin_cm)**2 + 2*rin_cm**2)))  / (
            rin_cm**2 + rin_cm*rout_cm + rout_cm**2 )

        # add to total acceleration
        accn += gmean

        # now compute temperatures from hydrostatic equilibrium
        T_prof = []
        P_ergpcm3 = P0_ergpcm3
        for ne_pcm3, width_cm, g_cmps2 in izip(
            ne_prof[::-1], annuli.widths_cm[::-1], accn[::-1]):

            T_keV = P_ergpcm3 / (
                (keV_K * boltzmann_erg_K * (1 + 1./ne_nH)) * ne_pcm3)
            T_prof.insert(0, T_keV)
            P_ergpcm3 += width_cm * g_cmps2 * ne_pcm3 * (mu_e * mu_g)

        T_prof = N.array(T_prof)

        return ne_prof, T_prof, Z_prof
