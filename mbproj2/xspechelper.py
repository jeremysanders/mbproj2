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

"""Module to interrogate xspec to get count rates and luminosities
given model parameters.

"""

from __future__ import division, print_function, absolute_import

import subprocess
import os
import select
import atexit
import re
import sys
from math import pi
import signal

from .physconstants import Mpc_cm, ne_nH
from . import cosmo

def deleteFile(filename):
    """Delete file, ignoring errors."""
    try:
        os.unlink(filename)
    except OSError:
        pass

# keep track of xspec invocations which need finishing
_finishatexit = []

# tcl code to do an infinite evaluation of commands until end
tclloop = '''
autosave off
while { 1 } {
 set s [gets stdin]
 if { [eof stdin] } {
   tclexit
 }
 eval $s
}
'''

class XSpecHelper:
    """A helper to get count rates for temperature and densities."""

    specialcode = '@S@T@V'
    specialre = re.compile('%s (.*) %s' % (specialcode, specialcode))
    normfactor = 1e75 # multiply norm by this to get into sensible units in xspec

    def __init__(self):
        self.xspecsub = subprocess.Popen(
            ['xspec'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )
        self.throwAwayOutput()
        self.tempoutput = None
        _finishatexit.append(self)

        self.write('set SCODE %s\n' % self.specialcode)

        # debugging
        logfile = os.path.join(os.environ['HOME'], 'xspec.log.%i' % id(self))
        deleteFile(logfile)
        #self.xspecsub.stdin.write('log %s\n' % logfile)

        # form of xspec loop where it doesn't blow up if this program
        # closes its stdin
        self.write(tclloop)

    def write(self, text):
        self.xspecsub.stdin.write(text)#.encode('utf-8'))

    def throwAwayOutput(self):
        """Ignore output from program until no more data available."""
        while True:
            i, o, e = select.select([self.xspecsub.stdout.fileno()], [], [], 0)
            if i:
                t = os.read(i[0], 1024)
                if not t:  # file was closed
                    break
            else:
                break

    def readResult(self):
        """Return result from xspec."""
        search = None
        while not search:
            line = self.xspecsub.stdout.readline()
            #line = line.decode('utf-8')
            search = XSpecHelper.specialre.search(line)
        return search.group(1)

    def setModel(self, NH_1022, T_keV, Z_solar, cosmo, ne_cm3):
        """Make a model with column density, temperature and density given."""
        self.write('model none\n')
        norm = 1e-14 / 4 / pi / (cosmo.D_A*Mpc_cm * (1.+cosmo.z))**2 * ne_cm3**2 / ne_nH
        self.write('model phabs(apec) & %g & %g & %g & %g & %g\n' %
            (NH_1022, T_keV, Z_solar, cosmo.z, norm*XSpecHelper.normfactor))
        self.throwAwayOutput()

    def changeResponse(self, rmf, arf, minenergy_keV, maxenergy_keV):
        """Create a fake spectrum using the response and use energy range given."""

        self.setModel(0.1, 1, 1, cosmo.Cosmology(0.1), 1.)
        self.tempoutput = '/tmp/jsproj_temp_%i.fak' % os.getpid()
        deleteFile(self.tempoutput)
        self.write('data none\n')
        self.write('fakeit none & %s & %s & y & foo & %s & 1.0\n' %
            (rmf, arf, self.tempoutput))

        # this is the range we are interested in getting rates for
        self.write('ignore **:**-%f,%f-**\n' % (minenergy_keV, maxenergy_keV))
        self.throwAwayOutput()

    def dummyResponse(self):
        """Make a wide-energy band dummy response."""
        self.write('data none\n')
        self.write('dummyrsp 0.01 100. 1000\n')
        self.throwAwayOutput()

    def getCountsPerSec(self, NH_1022, T_keV, Z_solar, cosmo, ne_cm3):
        """Return number of counts per second given parameters."""
        self.setModel(NH_1022, T_keV, Z_solar, cosmo, ne_cm3)
        self.write('puts "$SCODE [tcloutr rate 1] $SCODE"\n')
        #self.xspecsub.stdin.flush()
        retn = self.readResult()
        modelrate = float( retn.split()[2] ) / XSpecHelper.normfactor
        return modelrate

    def getFlux(self, T_keV, Z_solar, cosmo, ne_cm3):
        """Get flux in erg cm^-2 s^-1 from parcel of gas with the above parameters."""
        self.setModel(0., T_keV, Z_solar, cosmo, ne_cm3)
        self.write('flux 0.01 100.0\n')
        self.write('puts "$SCODE [tcloutr flux] $SCODE"\n')
        flux = float( self.readResult().split()[0] ) / XSpecHelper.normfactor
        return flux

    def finish(self):
        self.write('tclexit\n')
        #self.xspecsub.stdin.flush()
        self.throwAwayOutput()
        self.xspecsub.stdout.close()
        self.xspecsub.wait()
        if self.tempoutput:
            deleteFile(self.tempoutput)
        del _finishatexit[ _finishatexit.index(self) ]

def _finishXSpecs():
    """Finish any remaining xspecs if finish() does not get called above."""
    while _finishatexit:
        _finishatexit[0].finish()

atexit.register(_finishXSpecs)

# multiprocessing doesn't call atexit unless we do this
def sigterm(num, frame):
    _finishXSpecs()
    sys.exit()
signal.signal(signal.SIGTERM, sigterm)
