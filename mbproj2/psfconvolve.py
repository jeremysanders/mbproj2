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

"""Compute matrix to convolve profile with for accounting for PSF.

Note: the results are prone to instabilities if fitting binned
profiles when using a PSF.

"""

from __future__ import division, print_function, absolute_import

import warnings
import hashlib

import numpy as N
import h5py
import scipy.signal
import scipy.sparse
import six

from . import utils
from .utils import uprint

def makePSFImage(psf_edges, psf_val, pix_size):
    """Compute an image of the PSF using the angular scale."""

    npsfbins = int(N.ceil(psf_edges[-1]/pix_size))*2+1
    cpsf = npsfbins//2
    psfimg = N.zeros((npsfbins, npsfbins))
    psfrad = N.fromfunction(
        lambda y,x: N.sqrt((y-cpsf)**2+(x-cpsf)**2)*pix_size,
        (npsfbins, npsfbins))
    for r1, r2, v in zip(psf_edges[:-1], psf_edges[1:], psf_val):
        pix = (psfrad >= r1) & (psfrad < r2)
        psfimg[pix] = v
    return psfimg

def linearComputePSFMatrix(
    psf_edges, psf_val, shell_edges, psfoversample=4, annoversample=16):
    """A PSF matrix calculation which doesn't require convolution.

    Idea is to slide PSF radially along an axis and compute the
    contribution to other shells.

    psf_edges: array of edges for PSF
    psf_val: weighting for radial bin
    shell_edges: array of edges for shells (e.g. annuli.edges_arcmin)

    psfoversample: pixel size of psf is smallest psf radial bin
    divided by this

    annoversample: how many subdivisions in each annulus to slide psf
    over
    """

    uprint('Computing PSF matrix')

    # turn psf into a symmetric image (method assumes this)
    pix_size = N.diff(psf_edges).min() / psfoversample
    psfimg = makePSFImage(psf_edges, psf_val, pix_size)
    psfimg *= 1/psfimg.sum()

    shell_edges_sqd = shell_edges**2

    # coordinates of pixels
    psfy = N.fromfunction(
        lambda y,x: pix_size*(y-psfimg.shape[0]//2), psfimg.shape)
    psfx = N.fromfunction(
        lambda y,x: pix_size*(x-psfimg.shape[1]//2), psfimg.shape)

    # extract non-zero regions, as we only care about the distribution
    nonzero = psfimg != 0
    psfimg_f = psfimg[nonzero]
    psfy_f_sqd = (psfy[nonzero])**2
    psfx_f = psfx[nonzero]

    # output response matrix
    matout = N.zeros( (len(shell_edges)-1, len(shell_edges)-1) )

    for i, (e1, e2) in enumerate(six.zip(shell_edges[:-1], shell_edges[1:])):
        uprint(' shell', i)

        # split up shell and compute average midpoint radius
        subedges = N.linspace(e1, e2, annoversample)
        subrads = 0.5*(subedges[1:]+subedges[:-1])
        # scale contributions by area on sky
        subscales = (1./(e2**2-e1**2))*(subedges[1:]**2-subedges[:-1]**2)

        for subrad, subscale in six.zip(subrads, subscales):
            psfrad_sqd = (psfx_f+subrad)**2 + psfy_f_sqd

            hist, edges = N.histogram(
                psfrad_sqd, bins=shell_edges_sqd, weights=psfimg_f)
            matout[:, i] += hist*subscale

    return matout

def convComputePSFMatrix(psf_edges, psf_val, shell_edges, oversample=4):
    """Slow form of PSF matrix calculation using image convolution.

    Creates an image and does a convolution.
    """

    uprint('Computing PSF matrix')
    small_delta_psf = N.diff(psf_edges).min()
    small_delta_annuli = N.diff(shell_edges).min()
    pix_size = min(small_delta_psf, small_delta_annuli) / oversample

    # image of psf
    psfimg = makePSFImage(psf_edges, psf_val, pix_size).astype(N.float32)

    # output response matrix
    matout = N.zeros( (len(shell_edges)-1, len(shell_edges)-1) )

    for i, (e1, e2) in enumerate(six.zip(shell_edges[:-1], shell_edges[1:])):
        uprint(' shell', i)
        # make an image of shell and convolve with psf
        imgsize = int(N.ceil((e2+psf_edges[-1])/pix_size))*2+1

        radii = N.fromfunction(
            lambda y, x: (N.sqrt((y-imgsize//2)**2+(x-imgsize//2)**2)*
                          pix_size),
            (imgsize, imgsize))
        modimg = ((radii >= e1) & (radii < e2)).astype(N.float32)

        conv = scipy.signal.fftconvolve(modimg, psfimg, mode='same')
        # attempt at noise removal
        conv[conv < 2*abs(conv.min())] = 0
        # normalise to 1
        conv *= (1./conv.sum())

        hist, edges = N.histogram(radii, bins=shell_edges, weights=conv)
        matout[:, i] = hist
    uprint('Done')

    return matout

def convImagePSFMatrix(psfimg, pixsize_arcmin, shell_edges):
    """Compute a convolution PSF matrix using the image given."""

    imgsize = 2*(int(shell_edges[-1]/pixsize_arcmin)+1+max(psfimg.shape)//2) + 1

    radii = N.fromfunction(
        lambda y,x: N.sqrt(
            (x-imgsize//2)**2+(y-imgsize//2)**2)*pixsize_arcmin,
        (imgsize, imgsize))

    psfimgnorm = psfimg * (1./psfimg.sum())

    # output response matrix
    matout = N.zeros( (len(shell_edges)-1, len(shell_edges)-1) )

    for i, (e1, e2) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        if i % 20 == 0:
            uprint(' shell', i)

        annimg = ((radii >= e1) & (radii < e2)).astype(N.float)
        annimg *= 1./annimg.sum()

        conv = scipy.signal.fftconvolve(annimg, psfimgnorm, mode='same')
        # attempt at noise removal
        conv[conv < 2*abs(conv.min())] = 0

        #fitsgz.writeImageSimple(conv, 'conv_%i.fits' % i)
 
        hist, shell_edges = N.histogram(radii, bins=shell_edges, weights=conv)
        matout[:, i] = hist
    uprint('Done')

    return matout

def cachedPSFMatrix(psf_edge, psf_val, shell_edges):
    """Return PSF matrix, getting cached version if possible."""

    h = hashlib.md5()
    h.update(N.ascontiguousarray(psf_edge))
    h.update(N.ascontiguousarray(psf_val))
    h.update(N.ascontiguousarray(shell_edges))
    key = h.hexdigest()

    cachefile = 'psf_cache.hdf5'

    with utils.WithLock(cachefile+'.lockdir') as lock:
        with h5py.File(cachefile) as cache:
            if key not in cache:
                psf = linearComputePSFMatrix(psf_edge, psf_val, shell_edges)
                cache[key] = psf
            m = N.array(cache[key])
    return m
