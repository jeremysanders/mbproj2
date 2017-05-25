#!/usr/bin/env python

# this allows mbproj2 to be used to compute likelihoods for dnest3
# using dnest3_remote - see
# https://github.com/jeremysanders/dnest3_remote

# MBPROJ2_FIT environment variable should be set to point to a pickle
# file containing a fit object to run the analysis on

from __future__ import print_function, division

import mbproj2

import numpy as N
import os
import struct
import sys
import cPickle as pickle

def readlen(fd, length):
    """Read length bytes of data from file descriptor. read may return
    less than we expect, so this helper keeps reading until we get the
    required amount.
    """

    retn = b''
    while len(retn) < length:
        extra = os.read(fd, length-len(retn))
        if(len(extra)==0):
            print("End of data")
            sys.exit(0)
        retn += extra
    return retn

def params(fd, fit):
    """Return list of parameters to dnest3 caller."""

    out = struct.pack('i', len(fit.thawed))
    for name in fit.thawed:
        p = fit.pars[name]
        out += struct.pack('dddd32s', p.val, p.minval, p.maxval, p.maxval-p.minval, name)
    os.write(fd, out)
    os.close(fd)

def run(fd, fit):
    """Normal operation. Read a set of parameter values from the socket,
    then write out the likelihood. Parameters and result are binary
    double values."""

    npars = len(fit.thawed)

    while True:
        # FIXME: assumes sizeof double == 8
        remotepars = readlen(fd, 8*npars)
        pars = N.fromstring(remotepars, dtype=N.float64)

        like = fit.getLikelihood(pars)
        os.write(fd, struct.pack('d', float(like)))

def main():
    with open(os.environ['MBPROJ2_FIT']) as f:
        fit = pickle.load(f)

    # this is the file descriptor we read and write to
    fd = int(sys.argv[2])

    # select correct mode
    if sys.argv[1] == "params":
        params(fd, fit)
    elif sys.argv[1] == "run":
        run(fd, fit)

if __name__ == '__main__':
    main()
