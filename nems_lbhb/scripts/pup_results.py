#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
import pandas as pd
import scipy.signal as ss

import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems_db.xform_wrappers as nw
import nems_db.db as nd
import nems_db.params

import nems.recording as recording
import nems.epoch as ep
import nems.plots.api as nplt
import nems.modelspec as ms

# User parameters:
modelnames = [
        "psth.fs20.pup-ld-st.pup_stategain.S_jk.nf10-psthfr.j-basic",
        "psth.fs20.pup-ld-st.pup0_stategain.S_jk.nf10-psthfr.j-basic"
        ]

batch = 294

outpath = "{}/selectivity_{}/".format(
        '/auto/users/svd/docs/current/pupil_behavior',batch)

celldata = nd.get_batch_cells(batch=batch)
cellids = celldata['cellid'].tolist()

d = pd.DataFrame(columns=['cellid','state_chan','MI','MI_pup0','g','d',
                          'r','r_pup0','r_se','r_se_pup0'])
for mod_i, m in enumerate(modelnames):
    print('Loading ', m)
    modelspecs = nems_db.params._get_modelspecs(cellids, batch, m, multi='mean')
    for modelspec in modelspecs:
        c = modelspec[0]['meta']['cellid']
        dc = modelspec[0]['phi']['d']
        gain = modelspec[0]['phi']['g']
        meta = ms.get_modelspec_metadata(modelspec)
        state_mod = meta['state_mod']
        state_mod_se = meta['se_state_mod']
        state_chans = meta['state_chans']
        sc = 'pupil'
        j = 1
        ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
        if np.sum(ii)==0:
            d = d.append({'cellid': c, 'state_chan': sc}, ignore_index=True)
            ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
        if mod_i == 0:
            d.loc[ii, 'MI'] = (state_mod[j])
            d.loc[ii, 'g'] = (gain[0,j])
            d.loc[ii, 'd'] = (dc[0,j])
            d.loc[ii, 'r'] = (meta['r_test'][0])
            d.loc[ii, 'r_se'] = (meta['se_test'][0])
        elif mod_i == 1:
            d.loc[ii, 'MI_pup0'] = (state_mod[j])
            d.loc[ii, 'r_pup0'] = (meta['r_test'][0])
            d.loc[ii, 'r_se_pup0'] = (meta['se_test'][0])

r = d['r'].values.astype(float)
r0 = d['r_pup0'].values.astype(float)
se = d['r_se'].values.astype(float)
se0 = d['r_se_pup0'].values.astype(float)

goodcells = ((r-se) > (r0+se))
ax = None
stateplots.beta_comp(r0, r, n1="r pup0", n2="r pup",
                     title='stategain', hist_range=[-0.05, 0.95],
                     ax=ax, highlight=goodcells)


