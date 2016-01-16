# -*- coding: utf-8 -*-

# Mpc in km
Mpc_km = 3.0856776e19
Mpc_cm = 3.0856776e24
kpc_cm = 3.0856776e21

# km in cm
km_cm = 1e5

# Gravitational constant (cm^3 g^-1 s^-2)
G_cgs = 6.67428e-8

# solar mass in g
solar_mass_g = 1.98892e33

# ratio of electrons to Hydrogen atoms
ne_nH = 1.2

# energy conversions
keV_erg = 1.6021765e-09
keV_K = 11.6048e6

# boltzmann constant iin erg K^-1
boltzmann_erg_K = 1.3806503e-16

# unified atomic mass constant in g
mu_g = 1.6605402e-24

# unified mass constants per electron
# 1.41 is the mean atomic mass of solar abundance (mostly H and He)
mu_e = 1.41 / ne_nH

# year in s
yr_s = 31556926

# convert pressure in erg cm^-3 and electron density in cm^-3 to
# temperature in keV
P_ne_to_T = keV_K * boltzmann_erg_K * (1 + 1/ne_nH)
