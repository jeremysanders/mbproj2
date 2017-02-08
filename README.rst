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

More documentation can be found at http://mbproj2.readthedocs.io/

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
2. emcee  http://dan.iel.fm/emcee/ (Python module)
3. h5py   http://www.h5py.org/ (Python module)
4. numpy  http://www.numpy.org/ (Python module)
5. scipy  http://www.scipy.org/ (Python module)
6. PyYAML http://pyyaml.org/wiki/PyYAML (Python module)
7. xspec  https://heasarc.gsfc.nasa.gov/xanadu/xspec/
8. veusz  http://home.gna.org/veusz/ (optional plotting)

The Python module requirements can be installed using a Unix package
manager, the ``pip`` Python tool or a Python distribution such as
Anaconda, as appropriate for your setup. Please see the link above for
installing xspec. Before using MBPROJ2, make sure you have initialised
HEADAS.

Installation
------------

The standard way to install this module is to use the provided
``setup.py`` script:

::

  $ python setup.py build
  $ python setup.py install

This will place the module in the default install location. Please see
the Python distutils documentation if you want to choose different
install locations.

As the module is pure Python, if you want to install it manually you
can copy the ``mbproj2`` directory somewhere and modify your
``PYTHONPATH`` environment variable to include the directory where it
is located.

Using the module
----------------

The code can either be used as a Python module or driven using the
program ``mbproj2_compat`` using an external configuration file in YML
format. The second option is designed to be compatible with the format
of the file used by the previous version MBPROJ. This is not properly
documented, but the interested user can examine the source code.

The best way to understand how to use the module is to look at the
example source code provided with the distribution. The API
documentation provided details the various classes which make up
mbproj2.

The usual procedure to analyse data involves the following:

- Use the redshift of the object to make a Cosmology object

- Make an Annuli object which describes the annuli geometry in the
  count profiles

- Load cluster count profiles into Band objects (either manually or
  with loadBand).

- Construct a Data object from the list of Band objects.

- Make the density and metallicity model component profiles by using
  one of the provided Cmpt model component objects (e.g. CmptFlat or
  CmptProfile).

- Make the mass model component (if assuming hydrostatic equilibrium)
  from one of the CmptMass objects.

- Use ModelHydro to make a model from the various components.

- Get a list of default parameters and modify if necessary.

- Construct a Fit object which takes the model, parameters and data.

- Find the best fitting set of parameters.

- Use a MCMC object to do the Markov Chain Monte Carlo and write the
  chain to a HDF5 file.

- Use replayChainPhys to construct chains of physical parameters and
  savePhysProfilesHDF5 to write profiles of median values.

