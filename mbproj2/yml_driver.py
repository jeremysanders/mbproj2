# Compatibility driver for using mbproj1 yml files

from __future__ import division, print_function

import os
import numpy as N
import yaml

import data
import cosmo
import cmpt
import cmpt_mass
import model
import fit
import helpers

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

def cmptType(subpars, annuli, name, defpars):

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
        m = cmpt.CmptBinned(
            name, annuli,
            defval=defval,
            minval=minval, maxval=maxval,
            binning=subpars.get('bin', 1),
            interpolate=subpars.get('interpolate', False),
            log=islog)
        defpars.update(m.defPars())
    elif subpars['type'] == 'Hydrostatic':
        m = 'hydrostatic'
    else:
        raise RuntimeError('Unsupported component')
    return m

def constructCmptMass(pars, annuli, defpars):
    """Construct mass component."""
    pots = []
    for pot in pars['potential']:
        potp = pot['params']
        if pot['type'] == 'NFW':
            m = cmpt_mass.CmptMassNFW(annuli)
            defpars.update(m.defPars())
            defpars['nfw_conc'].val = potp['concentration']['val']
            defpars['nfw_conc'].minval = potp['concentration'].get('min', 0.01)
            defpars['nfw_conc'].maxval = potp['concentration'].get('max', 200.)
            defpars['nfw_conc'].frozen = potp['concentration'].get('fixed', False)
            defpars['nfw_r200_Mpc'].val = potp['r200_Mpc']['val']
            defpars['nfw_r200_Mpc'].minval = potp['r200_Mpc'].get('min', 0.01)
            defpars['nfw_r200_Mpc'].maxval = potp['r200_Mpc'].get('max', 10.)
            defpars['nfw_r200_Mpc'].frozen = potp['r200_Mpc'].get('fixed', False)
        elif pot['type'] == 'King':
            m = cmpt_mass.CmptMassKing(annuli)
            defpars.update(m.defPars())
            defpars['king_sigma_kmps'].val = potp['sigma_kmps']['val']
            defpars['king_sigma_kmps'].minval = potp['sigma_kmps'].get('min', 10.)
            defpars['king_sigma_kmps'].maxval = potp['sigma_kmps'].get('max', 5000.)
            defpars['king_sigma_kmps'].frozen = potp['sigma_kmps'].get('fixed', False)
            defpars['king_rcore_kpc'].val = potp['rcore_kpc']['val']
            defpars['king_rcore_kpc'].minval = potp['rcore_kpc'].get('min', 0.1)
            defpars['king_rcore_kpc'].maxval = potp['rcore_kpc'].get('max', 2500.)
            defpars['king_rcore_kpc'].frozen = potp['rcore_kpc'].get('fixed', False)
        else:
            raise RuntimeError('Unsupported mass component')

        pots.append(m)

    if len(pots) == 1:
        return pots[0]
    else:
        return cmpt_mass.CmptMassMulti('multi', annuli, pots)

def constructModel(pars, annuli):
    defpars = {}
    mpars = pars['model']['params']
    ne = cmptType(mpars['ne_logpcm3'], annuli, 'ne', defpars)
    Z = cmptType(mpars['Z_solar'], annuli, 'Z', defpars)
    T = cmptType(mpars['T_keV'], annuli, 'T', defpars)

    if mpars['NH_1022pcm2']['type'] != 'Flat':
        raise RuntimeError('Unsupported NH')
    NH = mpars['NH_1022pcm2']['val']

    if T == 'hydrostatic':
        masscmpt = constructCmptMass(pars, annuli, defpars)
        mod = model.ModelHydro(annuli, masscmpt, ne, Z, NH_1022pcm2=NH)
    else:
        mod = model.ModelNullPot(annuli, ne, T, Z, NH_1022pcm2=NH)

    defpars.update(mod.defPars())
    return mod, defpars

def runYML(inyml):
    ypars = yaml.load(open(inyml))

    theannuli = constructAnnuli(ypars)
    thedata = constructData(ypars, theannuli)
    themodel, thepars = constructModel(ypars, theannuli)

    helpers.estimateDensityProfile(themodel, thedata, thepars)

    thefit = fit.Fit(thepars, themodel, thedata)
    thefit.doFitting()

os.chdir('/data11s/jsanders/newprogs/jsproj/examples/pks0745')
runYML('pks0745_nfw.yml')
