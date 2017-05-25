#!/usr/bin/env python
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

from __future__ import division, print_function
from distutils.core import setup

setup(
    name='mbproj2',
    version='0.2',
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
