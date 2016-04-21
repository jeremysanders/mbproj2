# Compatibility driver for using mbproj1 yml files

from __future__ import division, print_function

import argparse
import os
import cPickle as pickle
import math

import numpy as N
import yaml

import data
import cosmo
import cmpt
import cmpt_mass
import model
import fit
import helpers
import phys
import mcmc

def readProfile(arg):
    """Read a particular column from a file, ignoring blank lines."""
    filename, column = arg
    out = []
    with open(filename) as f:
        for line in f:
            comment = line.find('#')
            if comment >= 0:
                line = line[:comment]
            p = line.split()
            if len(p) > 0:
                out.append(p[column])
    out = N.array([float(x) for x in out])
    return out

def constructAnnuli(pars):
    """Construct an Annuli object."""
    p = pars['radii']
    assert p['units'] == 'arcmin'
    centre = readProfile(p['centre'])
    hw = readProfile(p['halfwidth'])

    cos = cosmo.Cosmology(pars['model']['params']['redshift']['val'])

    return data.Annuli(N.concatenate(([centre[0]-hw[0]], centre+hw)), cos)

def constructData(pars, annuli):
    """Construct a data object."""

    areascales = readProfile(pars['radii']['areas']) / (
        N.pi * (annuli.edges_arcmin[1:]**2 - annuli.edges_arcmin[:-1]**2))

    bands = []
    for b in pars['bands']:
        band = data.Band(
            b['emin_keV'], b['emax_keV'],
            readProfile(b['cts']),
            b['rmfs'], b['arfs'],
            readProfile(b['exposures']),
            backrates=readProfile(b['backrate']),
            areascales=areascales)
        bands.append(band)

    return data.Data(bands, annuli)
 
def constructCmpt(subpars, annuli, name, defpars):
    """Convert input file component type to output component object."""

    islog = name == 'ne'
    defval = subpars.get('val', None)

    minval, maxval = {
        'ne': (-7., 1.),
        'T': (0.1, 50.),
        'Z': (0., 10.),
        }[name]

    if subpars['type'] == 'Flat':
        m = cmpt.CmptFlat(name, annuli, log=islog, minval=minval, maxval=maxval)
        defpars.update(m.defPars())
        defpars[name].val = defval
        defpars[name].frozen = subpars.get('fixed', True)
    elif subpars['type'] == 'Profile':
        m = cmpt.CmptBinnedJumpPrior(
            name, annuli,
            defval=defval,
            minval=minval, maxval=maxval,
            binning=subpars.get('bin', 1),
            interpolate=subpars.get('interpolate', False),
            log=islog)
        defpars.update(m.defPars())
    elif subpars['type'] == 'Beta':
        m = cmpt.CmptBeta(name, annuli)
        defpars.update(m.defPars())
        defpars['n0'].val = subpars.get('n0', -2.)
        defpars['n0'].frozen = subpars.get('n0_fixed', False)
        defpars['beta'].val = subpars.get('beta', 2./3.)
        defpars['beta'].frozen = subpars.get('beta_fixed', False)
        defpars['rc'].val = subpars.get('rc', 50.)
        defpars['rc'].frozen = subpars.get('rc_fixed', False)
    elif subpars['type'] == 'Hydrostatic':
        m = 'hydrostatic'
    else:
        raise RuntimeError('Unsupported component')
    return m

def constructCmptMass(pars, annuli, defpars):
    """Construct mass component."""

    def copypar(outname, insect, defmin, defmax, log=False):
        """Update parameter outname from input section insect with
        minimum and maximum given."""
        def lf(val):
            return math.log10(val) if log else val

        defpars[outname].val = lf(insect['val'])
        defpars[outname].minval = lf(insect.get('min', defmin))
        defpars[outname].maxval = lf(insect.get('max', defmax))
        defpars[outname].frozen = insect.get('fixed', False)

    pots = []
    for pot in pars['potential']:
        potp = pot['params']
        if pot['type'] == 'NFW':
            m = cmpt_mass.CmptMassNFW(annuli)
            defpars.update(m.defPars())
            copypar('nfw_logconc', potp['concentration'],
                    0.01, 200., log=True)
            copypar('nfw_r200_logMpc', potp['r200_Mpc'],
                    0.01, 10., log=True)
        elif pot['type'] == 'King':
            m = cmpt_mass.CmptMassKing(annuli)
            defpars.update(m.defPars())
            copypar(
                'king_sigma_logkmps', potp['sigma_kmps'],
                10., 5000., log=True)
            copypar(
                'king_rcore_logkpc', potp['rcore_kpc'],
                0.1, 2500., log=True)

        else:
            raise RuntimeError('Unsupported mass component')

        pots.append(m)

    if len(pots) == 1:
        return pots[0]
    else:
        return cmpt_mass.CmptMassMulti('multi', annuli, pots)

def constructModel(pars, annuli):
    """Make Model object from input parameters."""

    defpars = {}
    mpars = pars['model']['params']
    ne = constructCmpt(mpars['ne_logpcm3'], annuli, 'ne', defpars)
    Z = constructCmpt(mpars['Z_solar'], annuli, 'Z', defpars)
    T = constructCmpt(mpars['T_keV'], annuli, 'T', defpars)

    # densities should not jump by more than a factor of 10
    # this resolves oscillations in the fitting
    # ugly hack?
    ne.priorjump = 10.

    if mpars['NH_1022pcm2']['type'] != 'Flat':
        raise RuntimeError('Unsupported NH')
    NH = mpars['NH_1022pcm2']['val']

    if T == 'hydrostatic':
        masscmpt = constructCmptMass(pars, annuli, defpars)
        mod = model.ModelHydro(annuli, masscmpt, ne, Z, NH_1022pcm2=NH)
    else:
        mod = model.ModelNullPot(annuli, ne, T, Z, NH_1022pcm2=NH)

    moddefpars = mod.defPars()
    for par in moddefpars:
        if par not in defpars:
            defpars[par] = moddefpars[par]

    return mod, defpars

class YMLDriver:
    """Class wraps behaviour of mbproj1."""

    def __init__(self, inyml, threads=None):
        self.ypars = yaml.load(open(inyml))

        self.name = self.ypars['main']['name']
        self.chainfilename = '%s_mbp2_chain.hdf5' % self.name
        self.threads = self.ypars['mcmc']['threads'] if not threads else threads

        self.annuli = constructAnnuli(self.ypars)
        self.data = constructData(self.ypars, self.annuli)
        self.model, self.pars = constructModel(self.ypars, self.annuli)

        # optional background normalisation scaling
        if 'backscale' in self.ypars['model']['params']:
            bs = self.ypars['model']['params']['backscale']
            val = bs['val']
            rng = bs.get('range', 0.)
            backfrozen = True if rng==0. else bs.get('fixed', False)

            self.pars['backscale'] = fit.Param(
                val, minval=val-rng, maxval=val+rng, frozen=backfrozen)

    def run(self):
        """Run, producing chain."""
        helpers.estimateDensityProfile(self.model, self.data, self.pars)
        thefit = fit.Fit(self.pars, self.model, self.data)

        # first fit, freezing density
        print("Fitting with frozen densities")
        for name, par in self.pars.iteritems():
            if name[:3] == 'ne_':
                par.frozen = True

        # freeze background scaling (if used)
        if 'backscale' in self.pars:
            backfrozen = self.pars['backscale'].frozen
            self.pars['backscale'].frozen = True

        thefit.refreshThawed()
        thefit.doFitting()

        # then thaw again
        print("Thawing densities")
        for name, par in self.pars.iteritems():
            if name[:3] == 'ne_':
                par.frozen = False

        thefit.refreshThawed()
        self.model.ne_cmpt.priorjump = 2.0
        thefit.doFitting()

        # constraining ne
        print("Opening ne constraints")
        self.model.ne_cmpt.priorjump = 4.0
        thefit.doFitting()
        self.model.ne_cmpt.priorjump = 10.0
        thefit.doFitting()

        # thaw background (if set)
        if 'backscale' in self.pars and not backfrozen:
            print('Thawing background scaling')
            self.pars['backscale'].frozen = False
            thefit.refreshThawed()
            thefit.doFitting()

        # do burn in
        y = self.ypars['mcmc']
        m = mcmc.MCMC(thefit, walkers=y['walkers'], processes=self.threads)
        m.burnIn(y['burn'])

        # write pickle containing best fit to file
        with open('%s_mbp2_fit.pickle' % self.name, 'wb') as f:
            pickle.dump(thefit, f, -1)

        m.run(y['length'])

        m.save(self.chainfilename)

    def medians(self, mode='hdf5', thin=10, burn=0, confint=68.269):
        """Convert chain into physical quantities.

        mode: hdf5, text or hdf5+text
        """
        profs = phys.replayChainPhys(
            self.chainfilename, self.model, self.pars,
            thin=thin, burn=burn, confint=confint)

        if mode == 'hdf5' or mode == 'hdf5+text':
            phys.savePhysProfilesHDF5('%s_mbp2_medians.hdf5' % self.name, profs)
        if mode == 'text' or mode == 'hdf5+text':
            phys.savePhysProfilesText('%s_mbp2_medians.txt' % self.name, profs)

def ymlCmdLineParse():
    parser = argparse.ArgumentParser(
        description='MCMC multiband projection analysis (mbproj2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'conf', help='Input yml configuration file')
    parser.add_argument(
        'mode', help='Run mode', choices=['run', 'medians', 'run+medians'])
    parser.add_argument(
        '--medthin', default=10, type=int,
        help='Thin value when using medians (after original thin)')
    parser.add_argument(
        '--medburn', default=0, type=int,
        help='Extra ignore length for medians (after original thin)')
    parser.add_argument(
        '--medconf', default=68.269, type=float,
        help='Confidence interval to produce for medians')
    parser.add_argument(
        '--medfiletype', default='hdf5', choices=['hdf5', 'text', 'hdf5+text'],
        help='Medians output file type')
    parser.add_argument(
        '--override-threads', type=int,
        help='Specify number of threads (ignoring value in conf file)')
    parser.add_argument(
        '--working-dir',
        help='Working directory (defaults to current directory)')

    args = parser.parse_args()

    if args.working_dir:
        os.chdir(args.working_dir)

    yml = YMLDriver(args.conf, threads=args.override_threads)
    if args.mode == 'run' or args.mode == 'run+medians':
        yml.run()
    if args.mode == 'medians' or args.mode == 'run+medians':
        yml.medians(
            mode=args.medfiletype, thin=args.medthin,
            burn=args.medburn, confint=args.medconf)
