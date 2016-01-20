#!/usr/bin/env python

# compute matrix to convolve profile with for accounting for SPF

from __future__ import division, print_function

import warnings
import numpy as N
import h5py
import scipy.signal

import utils

def computePSFMatrix(psf_edges, psf_val, annuli, oversample=4):
    """Stupid form of PSF matrix calculation.

    Creates an image and does a convolution.
    """

    print('Computing PSF matrix')
    annuli_edges = annuli.edges_arcmin
    small_delta_psf = N.diff(psf_edges).min()
    small_delta_annuli = N.diff(annuli_edges).min()
    small_delta = min(small_delta_psf, small_delta_annuli)
    nbins = (annuli_edges[-1]+psf_edges[-1]) / small_delta * oversample

    # convert PSF profile to an image
    npsfbins = int(N.ceil(psf_edges[-1]/small_delta*oversample))*2+1
    cpsf = npsfbins//2
    psfimg = N.zeros((npsfbins, npsfbins))
    psfrad = N.fromfunction(
        lambda y,x: N.sqrt((y-cpsf)**2+(x-cpsf)**2)*small_delta/oversample,
        (npsfbins, npsfbins))
    for r1, r2, v in zip(psf_edges[:-1], psf_edges[1:], psf_val):
        pix = (psfrad >= r1) & (psfrad < r2)
        npix = N.sum(pix)
        # split up signal in pixels
        psfimg[pix] = v

    matout = N.zeros( (len(annuli_edges)-1, len(annuli_edges)-1) )

    for i, (e1, e2) in enumerate(zip(annuli_edges[:-1], annuli_edges[1:])):
        print(' shell', i)
        # compute image of cluster, assuming constant density in shell
        imgsize = int(N.ceil((e2+psf_edges[-1])/small_delta*oversample))*2+1

        r = N.fromfunction(
            lambda y, x: (N.sqrt((y-imgsize//2)**2+(x-imgsize//2)**2)*
                          (small_delta/oversample)),
            (imgsize, imgsize))
        modimg = (r >= e1) & (r <= e2)

        conv = scipy.signal.fftconvolve(modimg, psfimg, mode='same')
        # attempt at noise removal
        conv[conv < 2*abs(conv.min())] = 0
        # normalise to 1
        conv *= (1./conv.sum())

        hist, edges = N.histogram(r, bins=annuli_edges, weights=conv)
        matout[i, :] = hist
    print('Done')

    return matout

def cachedPSFMatrix(psf_edge, psf_val, annuli):
    """Return PSF matrix, getting cached version if possible."""

    key = str(
        utils.hashNumpy(psf_edge) ^ utils.hashNumpy(psf_val) ^
        utils.hashNumpy(annuli.edges_arcmin))

    cachefile = 'psf_cache.hdf5'

    with utils.WithLock(cachefile+'.lockdir') as lock:
        with h5py.File(cachefile) as cache:
            if key not in cache:
                psf = computePSFMatrix(psf_edge, psf_val, annuli)
                cache[key] = psf
            return N.array(cache[key])
