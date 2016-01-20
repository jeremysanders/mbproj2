from __future__ import division, print_function
from math import pi
from collections import defaultdict
import os

import numpy as N
import h5py

from physconstants import (
    kpc_cm, keV_erg, ne_nH, mu_g, mu_e, boltzmann_erg_K, keV_K, Mpc_cm,
    yr_s, solar_mass_g, G_cgs, P_ne_to_T)
import fit

# we want to define the cumulative values half way in the
# shell, so we have to split the luminosity and mass across
# the shell
def fracMassHalf(snum, annuli):
    """Fraction of mass of shell which is in the inner and outer
    halves of (r1+r2)/2."""

    r1, r2 = annuli.edges_cm[snum], annuli.edges_cm[snum+1]
    # Integrate[4*Pi*r^2, {r, r1, (r1 + r2)/2}]
    #  (Pi*(-8*r1**3 + (r1 + r2)**3))/6.
    # Integrate[4*Pi*r^2, {r, (r1 + r2)/2, r2}]
    #  4*Pi*(r2**3/3. - (r1 + r2)**3/24.)
    volinside = (pi * (-8*r1**3 + (r1 + r2)**3))/6
    voloutside = 4*pi * (r2**3/3 - (r1 + r2)**3/24)
    finside = volinside / (volinside + voloutside)
    foutside = 1 - finside
    return finside, foutside

def physFromProfs(model, pars):
    """Given model and parameters, calculate physical quantities."""

    annuli = model.annuli
    ne_prof, T_prof, Z_prof = model.computeProfs(pars)
    g_prof, pot_prof = model.computeMassProf(pars)
    nshells = len(ne_prof)

    v = {}
    v['ne_pcm3'] = ne_prof
    v['T_keV'] = T_prof
    v['Z_solar'] = Z_prof
    v['NH_1022pcm2'] = N.full(nshells, model.NH_1022pcm2)
    v['P_ergpcm3'] = T_prof * ne_prof * P_ne_to_T
    v['g_cmps2'] = g_prof
    v['potential_ergpg'] = pot_prof

    v['Pe_keVpcm3'] = ne_prof * T_prof
    v['Se_keVcm2'] = T_prof * ne_prof**(-2/3)
    v['vol_cm3'] = annuli.vols_cm3
    v['Mgas_Msun'] = v['ne_pcm3'] * v['vol_cm3'] * mu_e*mu_g/solar_mass_g

    v['fluxbolshell_ergpcm2'] = annuli.ctrate.getBolometricFlux(
        v['T_keV'], v['Z_solar'], v['ne_pcm3'])
    v['L_ergpspcm3'] = (
        v['fluxbolshell_ergpcm2'] * 4 * pi *
        (annuli.cosmology.D_L * Mpc_cm)**2 )
    v['Lshell_ergps'] = v['L_ergpspcm3'] * v['vol_cm3']
    v['H_ergpcm3'] = (5/2) * v['ne_pcm3'] * (
        1 + 1/ne_nH) * v['T_keV'] * keV_erg
    v['tcool_yr'] = v['H_ergpcm3'] / v['L_ergpspcm3'] / yr_s

    # split quantities about shell midpoint, so result is independent
    # of binning
    fi, fo = fracMassHalf(N.arange(nshells), annuli)

    v['Lcuml_ergps'] = v['Lshell_ergps']*fi + N.concatenate((
            [0], N.cumsum(v['Lshell_ergps'])[:-1]))
    v['Mgascuml_Msun'] = v['Mgas_Msun']*fi + N.concatenate((
            [0], N.cumsum(v['Mgas_Msun'])[:-1]))

    # this is the total mass (calculated from g)
    v['Mtotcuml_Msun'] = v['g_cmps2']*annuli.massav_cm**2/G_cgs/solar_mass_g
    #v['Mtotcuml_Msun'] = v['g_cmps2']*annuli.midpt_cm**2/G_cgs/solar_mass_g

    # and the gas fraction (<r)
    v['fgascuml'] = v['Mgascuml_Msun'] / v['Mtotcuml_Msun']

    # Mdots
    density_gpcm3 = v['ne_pcm3'] * mu_e * mu_g
    v['H_ergpg'] = v['H_ergpcm3'] / density_gpcm3
    v['Mdotpurecool_Msunpyr'] = (
        v['Lshell_ergps'] / v['H_ergpg'] / solar_mass_g * yr_s)
    v['Mdotpurecoolcuml_Msunpyr'] = N.cumsum(v['Mdotpurecool_Msunpyr'])

    # output mdot values go here
    v['Mdot_Msunpyr'] = N.zeros(nshells)
    v['Mdotcuml_Msunpyr'] = N.zeros(nshells)

    # change in potential and enthalpy across each shell
    delta_pot_ergpg = N.concatenate((
            [0], v['potential_ergpg'][1:]-v['potential_ergpg'][:-1]))
    delta_H_ergpg = N.concatenate((
            [0], v['H_ergpg'][1:]-v['H_ergpg'][:-1]))

    Mdotcuml_gps = 0.
    for i in xrange(nshells):
        # total energy going into mdot in this shell, subtracting contribution
        # of matter which flows inwards
        E_tot_ergps = (
            v['Lshell_ergps'][i] -
            Mdotcuml_gps*(delta_pot_ergpg[i] + delta_H_ergpg[i]))
        # energy comes from enthalpy plus change in potential
        E_tot_ergpg = v['H_ergpg'][i] + delta_pot_ergpg[i]

        Mdot_gps = E_tot_ergps / E_tot_ergpg
        v['Mdot_Msunpyr'][i] = Mdot_gps / solar_mass_g * yr_s
        Mdotcuml_gps += Mdot_gps
        v['Mdotcuml_Msunpyr'][i] = Mdotcuml_gps / solar_mass_g * yr_s

    return v

def replayChainPhys(chainfilename,
                    model, pars, burn=0, thin=10, confint=68.269):
    """Replay chain, compute physical quantity profiles.

    confint: confidence interval
    burn: skip initial N items in chain
    thin: skip every N iterations in chain

    Returns medians and confidence interval percentiles (1 sigma by
    default)
    """

    print('Computing physical quantities from chain', chainfilename)
    with h5py.File(chainfilename) as f:
        fakefit = fit.Fit(pars, model, None)
        if fakefit.thawed != list(f['thawed_params']):
            raise RuntimeError('Parameters do not match those in chain')

        chain = f['chain'][:, burn::thin, :]
        chain = chain.reshape(-1, chain.shape[2])

    data = defaultdict(list)
    length = len(chain)
    for i, parvals in enumerate(chain):
        if i % 1000 == 0:
            print(' Step %i / %i (%.1f%%)' % (i, length, i*100/length))

        fakefit.updateThawed(parvals)

        physvals = physFromProfs(model, fakefit.pars)
        for name, vals in physvals.iteritems():
            data[name].append(vals)

    # compute medians and errors
    outprofs = {}
    for name, vals in data.iteritems():
        # compute percentiles
        median, posrange, negrange = N.percentile(
            vals, [50, 50+confint/2, 50-confint/2], axis=0)

        # compute error bars
        prof = N.column_stack((median, posrange-median, negrange-median))
        outprofs[name] = prof

    annuli = model.annuli
    r_arcmin = 0.5*(annuli.edges_arcmin[1:]+annuli.edges_arcmin[:-1])
    r_width_arcmin = 0.5*(annuli.edges_arcmin[1:]-annuli.edges_arcmin[:-1])

    outprofs['r_arcmin'] = N.column_stack((r_arcmin, r_width_arcmin))
    outprofs['r_kpc'] = N.column_stack(
        (annuli.midpt_cm / kpc_cm, 0.5*annuli.widths_cm / kpc_cm))

    print('Done quantity computation')
    return outprofs

def savePhysProfilesHDF5(outfilename, profiles):
    """Save output profile to hdf5."""
    try:
        os.unlink(outfilename)
    except OSError:
        pass
    print('Writing', outfilename)
    with h5py.File(outfilename) as f:
        for name in profiles:
            f[name] = profiles[name]
            f[name].attrs['vsz_twod_as_oned'] = 1

def savePhysProfilesText(outfilename, profiles):
    """Save output profile as text."""
    try:
        os.unlink(outfilename)
    except OSError:
        pass
    print('Writing', outfilename)
    with open(outfilename, 'w') as f:
        for name in sorted(profiles):
            prof = profiles[name]
            err = {1: '', 2: ',+-', 3: ',+,-'}[len(prof[0])]
            fmt = ' '.join(['% e']*len(prof[0])) + '\n'

            f.write('descriptor %s%s\n' % (name, err))
            for line in prof:
                f.write(fmt % tuple(line))
            f.write('\n')
