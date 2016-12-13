MBPROJ2
=======

https://github.com/jeremysanders/mbproj2

Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>

MBPROJ2 is released under the GNU Library General Public License
version 2 or greater. See the file COPYING for details.

Described in *Evolution, core properties and masses of SPT-selected
galaxy clusters from hydrostatic Chandra X-ray profiles*,
J.S. Sanders, A.C. Fabian, H.R. Russell and S.A. Walker, submitted to
MNRAS

Introduction
------------

MBPROJ2 is a code which forward-models surface brightness profiles of
galaxy clusters. Using a single X-ray band the model would be
sensitive to the density of the intracluster medium (ICM). By using
multiple bands the code is able to model the temperature variation
within the cluster. Given sufficient energy bands the metallicity of
the ICM can also be fitted.

MBPROJ2 can assume hydrostatic equilibrium using a mass model. From
this model (with an outer pressure parameter and the density profile)
the pressure profile can be computed for the cluster. This allows the
temperature to be computed on small spatial scales, which would
otherwise not have enough data to compute the temperature
independently. If no hydrostatic equilibrium is assumed then MBPROJ2
can fit for the temperature of the ICM instead.

The defined model is normally first fit to the surface brightness
profiles. MCMC using the emcee module is used to compute a chain of
model parameters, from which posterior probability distributions can
be computed for a large number of model and physical parameters.

Requirements
------------
MBPROJ2 requires the following:

1. Python 2.7+ or 3.3+ or greater
2. emcee http://dan.iel.fm/emcee/
3. h5py  http://www.h5py.org/
4. numpy http://www.numpy.org/
5. scipy http://www.scipy.org/
6. yaml  http://pyyaml.org/wiki/PyYAML
7. xspec https://heasarc.gsfc.nasa.gov/xanadu/xspec/

