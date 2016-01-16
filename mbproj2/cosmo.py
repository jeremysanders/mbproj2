"""Routines for calculating distances from cosmology,
Taken from Ned Wright's cosmology calculator."""

from __future__ import division, print_function
from math import sqrt, pi, sin, exp

c = 299792.458 # velocity of light in km/sec
Tyr = 977.8    # coefficent for converting 1/H into Gyr

class Cosmology(object):
    """Cosmology calculation object."""

    def __init__(self, z, H0=70., q0=0.5, WM=0.3, WV=0.7):
        """Set up cosmology object for cosmology of H0, q0, WM, WV and z."""
        self.H0 = H0
        self.WM = WM
        self.WV = WV
        self.z = z
        self._lastparams = ()

    def calculate(self):
        """Recalculate distances if necessary."""
        params = (self.H0, self.WM, self.WV, self.z)
        if self._lastparams == params:
            return
        self._lastparams = params
        H0, WM, WV, z = params

        WR = 0.        # Omega(radiation)
        WK = 0.        # Omega curvaturve = 1-Omega(total)
        DTT = 0.5      # time from z to now in units of 1/H0
        DTT_Gyr = 0.0  # value of DTT in Gyr
        age = 0.5      # age of Universe in units of 1/H0
        age_Gyr = 0.0  # value of age in Gyr
        zage = 0.1     # age of Universe at redshift z in units of 1/H0
        zage_Gyr = 0.0 # value of zage in Gyr
        DCMR = 0.0     # comoving radial distance in units of c/H0
        DCMR_Mpc = 0.0
        DCMR_Gyr = 0.0
        DA = 0.0       # angular size distance
        DA_Mpc = 0.0
        DA_Gyr = 0.0
        kpc_DA = 0.0
        DL = 0.0       # luminosity distance
        DL_Mpc = 0.0
        DL_Gyr = 0.0   # DL in units of billions of light years
        V_Gpc = 0.0
        a = 1.0        # 1/(1+z), the scale factor of the Universe
        az = 0.5       # 1/(1+z(object))

        h = H0/100.
        WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
        WK = 1-WM-WR-WV
        az = 1.0/(1+1.0*z)
        age = 0.
        n=1000         # number of points in integrals
        for i in xrange(n):
            a = az*(i+0.5)/n
            adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
            age = age + 1./adot

        zage = az*age/n
        zage_Gyr = (Tyr/H0)*zage
        DTT = 0.0
        DCMR = 0.0

        # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
        for i in xrange(n):
            a = az+(1-az)*(i+0.5)/n
            adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
            DTT = DTT + 1./adot
            DCMR = DCMR + 1./(a*adot)

        DTT = (1.-az)*DTT/n
        DCMR = (1.-az)*DCMR/n
        age = DTT+zage
        age_Gyr = age*(Tyr/H0)
        DTT_Gyr = (Tyr/H0)*DTT
        DCMR_Gyr = (Tyr/H0)*DCMR
        DCMR_Mpc = (c/H0)*DCMR

        # tangential comoving distance
        ratio = 1.00
        x = sqrt(abs(WK))*DCMR
        if x > 0.1:
            if WK > 0:
                ratio =  0.5*(exp(x)-exp(-x))/x
            else:
                ratio = sin(x)/x
        else:
            y = x*x
            if WK < 0: y = -y
            ratio = 1. + y/6. + y*y/120.
        DCMT = ratio*DCMR
        DA = az*DCMT
        DA_Mpc = (c/H0)*DA
        kpc_DA = DA_Mpc/206.264806
        DA_Gyr = (Tyr/H0)*DA
        DL = DA/(az*az)
        DL_Mpc = (c/H0)*DL
        DL_Gyr = (Tyr/H0)*DL

        # comoving volume computation
        ratio = 1.00
        x = sqrt(abs(WK))*DCMR
        if x > 0.1:
            if WK > 0:
                ratio = (0.125*(exp(2.*x)-exp(-2.*x))-x/2.)/(x*x*x/3.)
            else:
                ratio = (x/2. - sin(2.*x)/4.)/(x*x*x/3.)
        else:
            y = x*x
            if WK < 0:
                y = -y
            ratio = 1. + y/5. + (2./105.)*y*y
        VCM = ratio*DCMR*DCMR*DCMR/3.
        V_Gpc = 4.*pi*((0.001*c/H0)**3)*VCM

        self._calc_D_A = DA_Mpc
        self._calc_D_L = DL_Mpc
        self._calc_kpc_DA = kpc_DA

    @property
    def D_A(self):
        """Get angular diameter distance in Mpc."""
        self.calculate()
        return self._calc_D_A

    @property
    def D_L(self):
        """Get luminosity distance in Mpc."""
        self.calculate()
        return self._calc_D_L

    @property
    def kpc_per_arcsec(self):
        """Get number of kpc per arcsec."""
        self.calculate()
        return self._calc_kpc_DA