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

"""Collections of functions used by MBPROJ2.
"""

from __future__ import division, print_function, absolute_import

from six.moves import range
import numpy as N
from scipy.special import gammaln

import sys
import os
import time
import uuid

def uprint(*args, **argsv):
    """Unbuffered print."""
    print(*args, **argsv)
    sys.stdout.flush()

def projectionVolume(R1, R2, y1, y2):
    """Return the projected volume of a shell of radius R1->R2 onto an
    annulus on the sky of y1->y2.

    this is the integral:
    Int(y=y1,y2) Int(x=sqrt(R1^2-y^2),sqrt(R2^2-y^2)) 2*pi*y dx dy
     =
    Int(y=y1,y2) 2*pi*y*( sqrt(R2^2-y^2) - sqrt(R1^2-y^2) ) dy

    This is half the total volume (front only)
    """

    def truncSqrt(x):
        return N.sqrt(N.clip(x, 0., 1e200))

    p1 = truncSqrt(R1**2 - y2**2)
    p2 = truncSqrt(R1**2 - y1**2)
    p3 = truncSqrt(R2**2 - y2**2)
    p4 = truncSqrt(R2**2 - y1**2)

    return (2/3) * N.pi * ((p1**3 - p2**3) + (p4**3 - p3**3))

def projectionVolumeMatrix(radii):
    """Calculate volumes (front and back) using a matrix calculation.

    Dot matrix with emissivity array to compute projected surface
    brightnesses.

    Output looks like this:
    >>> utils.projectionVolumeMatrix(N.arange(5))
    array([[  4.1887902 ,   7.55593906,   6.57110358,   6.4200197 ],
           [  0.        ,  21.76559237,  26.1838121 ,  21.27257712],
           [  0.        ,   0.        ,  46.83209821,  49.71516053],
           [  0.        ,   0.        ,   0.        ,  77.57748023]])

    """

    i_s, j_s = N.indices((len(radii)-1, len(radii)-1))

    radii_2 = radii**2
    y1_2 = radii_2[i_s]
    y2_2 = radii_2[i_s+1]
    R1_2 = radii_2[j_s]
    R2_2 = radii_2[j_s+1]

    p1 = (R1_2-y2_2).clip(0)
    p2 = (R1_2-y1_2).clip(0)
    p3 = (R2_2-y2_2).clip(0)
    p4 = (R2_2-y1_2).clip(0)

    return 2 * (2/3) * N.pi * ((p1**1.5 - p2**1.5) + (p4**1.5 - p3**1.5))

def symmetriseErrors(data):
    """Take numpy-format data,+,- and convert to data,+-."""
    symerr = N.sqrt( 0.5*(data[:,1]**2 + data[:,2]**2) )
    datacpy = N.array(data[:,0:2])
    datacpy[:,1] = symerr
    return datacpy

def getMedianAndErrors(results):
    """Take a set of repeated results, and calculate the median and errors (from perecentiles)."""
    r = N.array(results)
    r.sort(0)
    num = r.shape[0]
    medians = r[ int(num*0.5) ]
    lowpcs = r[ int(num*0.1585) ]
    uppcs = r[ int(num*0.8415) ]
    
    return medians, uppcs-medians, lowpcs-medians

def calcChi2(model, data, error):
    """Calculate chi2 between model and data."""
    return (((data-model)/error)**2).sum()

def readProfile(filename, column, cnvtfloat=True):
    """Read a particular column from a file, ignoring blank lines."""
    out = []
    with open(filename) as f:
        for line in f:
            comment = line.find('#')
            if comment >= 0:
                line = line[:comment]
            p = line.split()
            if len(p) > 0:
                out.append(p[column])
    if cnvtfloat:
        out = N.array([float(x) for x in out])
    return out

def cashLogLikelihood(data, model):
    """Calculate log likelihood of Cash statistic."""

    like = N.sum(data * N.log(model)) - N.sum(model) - N.sum(gammaln(data+1))
    if N.isfinite(like):
        return like
    return -N.inf

class WithLock:
    """Hacky lockfile class."""

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        timeout = 500 # seconds
        for i in range(timeout):
            try:
                os.mkdir(self.filename)
                break
            except OSError:
                time.sleep(1)

    def __exit__(self, type, value, traceback):
        os.rmdir(self.filename)

class AtomicWriteFile(object):
    """Write to a file, renaming to final name when finished."""

    def __init__(self, filename):
        self.filename = filename
        self.tempfilename = filename + '.temp' + str(uuid.uuid4())
        self.f = open(self.tempfilename, 'w')

    def __enter__(self):
        return self.f

    def __exit__(self, type, value, traceback):
        self.f.close()
        os.rename(self.tempfilename, self.filename)
    
def gehrels(c):
    return 1. + N.sqrt(c + 0.75)
