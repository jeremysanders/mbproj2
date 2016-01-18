from __future__ import division, print_function
import numpy as N

import cmpt
import model
import fit

def estimateDensityProfile(inmodel, data, modelpars):
    """Do a simplified fit to get the initial density profile.

    This assumes an isothermal cluster with constant 0.3 solar
    metallicity and does not use hydrostatic equilibrium.
    """

    print('Estimating densities')

    annuli = inmodel.annuli
    # temporary model with flat temperature and metallicity (fixed)
    tempmodel = model.ModelNullPot(
        annuli,
        inmodel.ne_cmpt,
        cmpt.CmptFlat('T', annuli, defval=3., minval=0.1, maxval=12),
        cmpt.CmptFlat('Z', annuli, defval=0.3, log=False),
        NH_1022pcm2=inmodel.NH_1022pcm2)

    temppars = tempmodel.defPars()
    temppars['Z'].frozen = True
    tempfit = fit.Fit(temppars, tempmodel, data)

    tempfit.doFitting()

    # update parameters in original model
    txt = []
    for par in sorted(temppars):
        if par[:2] == 'ne':
            modelpars[par] = temppars[par]
            txt.append('%4.1f' % temppars[par].val)

    print('Done estimating densities:', ' '.join(txt))
