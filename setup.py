#!/usr/bin/env python

from distutils.core import setup

setup(
    name='mbproj2',
    version='0.1',
    description='MultiBand surface brightness PROJector 2',
    author='Jeremy Sanders',
    author_email='jeremy@jeremysanders.net',
    url='https://github.com/jeremysanders/mbproj2',

    requires = [
        'emcee',
        'h5py',
        'numpy',
        'six',
        'scipy',
        'yaml',
        ],

    extras_require = {
        'Plotting': ['veusz'],
        },

    packages=['mbproj2'],
    )
