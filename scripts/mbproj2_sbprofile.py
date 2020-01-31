#!/usr/bin/env python

"""
Generate a surface brightness profile for feeding into autorebin.py

NaN values are ignored, so use them for masking point sources
"""

from __future__ import print_function, division

import argparse
import sys
from astropy.io import fits
import numpy as N

def makeProfile(img, xc, yc, rmax, binf):
    """Take image and compute profile. Ignores nan values.

    Returns: array of total values in bins, array of number of pixels in bins

    xc, yc: centre in pixels
    rmax: maximum radius in pixels
    binf: binning factor
    """

    radii = N.fromfunction(
        lambda y, x: (N.sqrt((x-xc)**2 + (y-yc)**2)/binf).astype(N.int32),
        img.shape)

    rmaxdiv = int(rmax / binf)
    radii[ N.logical_not(N.isfinite(img)) | (radii >= rmaxdiv) ] = rmaxdiv

    numpix = N.bincount(radii.ravel())
    ctsums = N.bincount(radii.ravel(), weights=img.ravel())

    return ctsums[:rmaxdiv], numpix[:rmaxdiv]

def oversampleCounts(inimage, oversample):
    """Oversample by randomizing counts over pixels which are
    oversample times larger."""

    if oversample == 1:
        return inimage

    if inimage.dtype.kind not in 'iu':
        raise ValueError("Non-integer input image")
    if N.any(inimage < 0):
        raise ValueError("Input counts image has negative pixels")

    y, x = N.indices(inimage.shape)
    
    # make coordinates for each count
    yc = N.repeat(N.ravel(y), N.ravel(inimage))
    xc = N.repeat(N.ravel(x), N.ravel(inimage))

    # add on random amount
    xc = xc*oversample + N.random.randint(oversample, size=len(xc))
    yc = yc*oversample + N.random.randint(oversample, size=len(yc))

    outimages = N.histogram2d(
        yc, xc, (inimage.shape[0]*oversample, inimage.shape[1]*oversample))
    outimage = N.array(outimages[0], dtype=N.int)

    return outimage

def oversampleSimple(inimage, oversample, average=False):
    """Oversample by repeating elements by oversample times."""
    os1 = N.repeat(inimage, oversample, axis=1)
    os0 = N.repeat(os1, oversample, axis=0)

    if average:
        os0 *= (1./oversample**2)

    return os0

def main():
    parser = argparse.ArgumentParser(
        description='extract a surface brightness profile',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inimage',
                        help='input image')
    parser.add_argument('outprofile',
                        help='output profile')
    parser.add_argument('--mask',
                        help='mask_file')
    parser.add_argument('--xc', type=float,
                        help='x centre (pixels)')
    parser.add_argument('--yc', type=float,
                        help='y centre (pixels)')
    parser.add_argument('--rmax', type=int, default=120,
                        help='maximum radius (pixels)')
    parser.add_argument('--exposuremap',
                        help='exposure map for scaling areas (optional)')
    parser.add_argument('--exp-from-expmap',
                        action='store_true',
                        help='take exposure from exposure map (s)')
    parser.add_argument('--bin', type=float, default=1,
                        help='bin factor (pixels)')
    parser.add_argument('--oversample', type=int, default=1,
                        help='oversample image by N pixels')

    args = parser.parse_args()

    binf, rmax, xc, yc = args.bin, args.rmax, args.xc, args.yc

    print('Opening', args.inimage)
    with fits.open(args.inimage) as f:

        exposure = f[0].header.get('EXPOSURE')
        pixsize_arcmin = abs(f[0].header['CDELT1'] * 60)

        img = N.array(f[0].data)
        if args.oversample != 1:
            if img.dtype.kind in 'ui':
                print('Oversampling in integer count mode')
                img = oversampleCounts(img, args.oversample)
            else:
                print('Oversampling in simple (non-count) mode')
                img = oversampleSimple(img, args.oversample, average=True)

            # output radii are scaled after resampling
            pixsize_arcmin /= args.oversample
            binf *= args.oversample
            rmax *= args.oversample
            xc *= args.oversample
            yc *= args.oversample

    if args.exposuremap:
        print('Reading exposure map', args.exposuremap)
        with fits.open(args.exposuremap) as f:
            expmap = N.array(f[0].data)

        expmap = oversampleSimple(expmap, args.oversample, average=True)

        # make sure same bad pixels used (and use nan in expmap for bad pixels)
        expmap = N.where(expmap != 0, expmap, N.nan)
        expmap = N.where(N.isfinite(img), expmap, N.nan)
        img = N.where(N.isfinite(expmap), img, N.nan)
    else:
        expmap = None

    # exposure from header
    if args.exp_from_expmap:
        assert expmap is not None
        exposure = expmap[int(args.yc), int(args.xc)]
        if not N.isfinite(exposure) or exposure <= 10:
            print("WARNING: likely invalid exposure obtained from exposure image",
                  file=sys.stderr)
    else:
        if exposure is None:
            print("WARNING: no EXPOSURE keyword in image. Assuming 1.", file=sys.stderr)
            exposure = 1.0

    if args.mask:
        with fits.open(args.mask) as f:
            img = img.astype(N.float64)
            img = oversampleSimple(img, args.oversample)
            img[f[0].data==0] = N.nan

    ctsum, pixsum = makeProfile(img, xc, yc, rmax, binf)
    areas = pixsum * pixsize_arcmin**2
    exposures = N.full_like(ctsum, exposure)

    # if there is an exposure map, scale the exposures by the
    # variation in exposure from the centre to the annulus
    if expmap is not None:
        expsum, exppixsum = makeProfile(expmap, xc, yc, rmax, binf)
        avexp = expsum / exppixsum

        # get estimate of central value in profile
        npix = sumexp = i = 0
        while npix < 20:
            npix += exppixsum[i]
            sumexp += expsum[i]
            i += 1

        # scale by central value (where rmf is calculated)
        scaledexp = avexp / (sumexp / npix)

        # scale outputted exposures by this factor
        exposures = N.where(N.isfinite(scaledexp), exposures*scaledexp, 0.)

    with open(args.outprofile, 'w') as fout:
        incentre = True
        print('# sbprofile_multiband.py arguments:', file=fout)
        for a in sorted(vars(args)):
            print('#  %s = %s' % (a, getattr(args, a)), file=fout)

        print( '# rcentre(amin) rhalfwidth(amin) counts area(amin2) exposure sb',
               file=fout )
        for i in range(int(rmax/binf)):
            if areas[i] == 0 and incentre:
                continue
            incentre = False

            print( (0.5+i)*binf*pixsize_arcmin,
                   0.5*binf*pixsize_arcmin,
                   ctsum[i], areas[i], exposures[i],
                   ctsum[i]/areas[i]/exposures[i], file=fout )

if __name__ == '__main__':
    main()
