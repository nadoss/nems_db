#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))

import nems.db as nd
import nems_db.params
import numpy as np
import matplotlib.pyplot as plt
import svdplots


batch = 303
modelname1 = "psth20pup0prebeh_stategain4_basic-nf"
modelname2 = "psth20pupprebeh_stategain4_basic-nf"

bins = 20
range = [-1, 1]
n1 = 'pre gain'
n2 = 'active gain'
i1 = 2
i2 = 3


res = nd.batch_comp(batch, [modelname0, modelname])

d1 = nems_db.params.fitted_params_per_batch(
        batch, modelname,
        stats_keys=[])
d2 = nems_db.params.fitted_params_per_batch(
        batch, modelname2,
        stats_keys=[])

# parse modelname
kws = modelname.split("_")
modname = kws[1]
statecount = int(modname[-1])

g1 = np.zeros([0, statecount])
b1 = np.zeros([0, statecount])
g2 = np.zeros([0, statecount])
b2 = np.zeros([0, statecount])
sig = np.zeros(len(d1.columns))
i = 0
ff = res[modelname] >= res[modelname0] + 0.005

for c in d1.columns:
    # print(c)
    # print(x)

    x = d1.loc['0--'+modname+'--g'][c]
    g1 = np.append(g1, np.reshape(x, [1, -1]), axis=0)
    y = d1.loc['0--'+modname+'--d'][c]
    b1 = np.append(b1, np.reshape(y, [1, -1]), axis=0)

    x = d2.loc['0--'+modname+'--g'][c]
    g2 = np.append(g2, np.reshape(x, [1, -1]), axis=0)
    y = d2.loc['0--'+modname+'--d'][c]
    b2 = np.append(b2, np.reshape(y, [1, -1]), axis=0)

    print("{0}: {1}={2:.3f} {3}={4:.3f}  {5}={6:.3f} {7}={8:.3f}".
          format(c, n1, g1[-1, i1], n2, g1[-1, i2],
                 n1, g2[-1, i1], n2, g2[-1, i2]))

    sig[i] = ff.loc[c]
    i += 1

title = '{}/{}'.format(n1, batch)

svdplots.beta_comp(g1[:, i1], g2[:, i1],
          n1=modelname, n2=modelname2,
          hist_bins=bins, hist_range=range, title=title,
          highlight=sig)

title = '{}/{}'.format(n2, batch)

svdplots.beta_comp(g1[:, i2], g2[:, i2],
          n1=modelname, n2=modelname2,
          hist_bins=bins, hist_range=range, title=title,
          highlight=sig)

#title = '{}/{}'.format(modelname, batch)
#
#svdplots.beta_comp_cols(g1[:, [i1, i2]], b1[:, [i1, i2]], n1=n1, n2=n2,
#          hist_bins=bins, hist_range=range, title=title,
#          highlight=sig)
