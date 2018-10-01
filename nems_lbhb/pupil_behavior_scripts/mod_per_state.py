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


batch = 307

state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

# User parameters:
loader = "psth.fs20.pup-ld-"
fitter = "_jk.nf20-basic"
basemodel = "-ref-psthfr.s_stategain.S"

modelnames = [loader + s + basemodel + fitter for s in state_list]

celldata = nd.get_batch_cells(batch=batch)
cellids = celldata['cellid'].tolist()

d = pd.DataFrame(columns=['cellid','modelname','state_sig','state_chan','MI',
                          'r','r_se','d','g'])

for mod_i, m in enumerate(modelnames):
    print('Loading modelname: ', m)
    modelspecs = nems_db.params._get_modelspecs(cellids, batch, m, multi='mean')

    for modelspec in modelspecs:
        meta = ms.get_modelspec_metadata(modelspec)
        c = meta['cellid']
        state_mod = meta['state_mod']
        state_mod_se = meta['se_state_mod']
        state_chans = meta['state_chans']
        dc = modelspec[0]['phi']['d']
        gain = modelspec[0]['phi']['g']
        for j, sc in enumerate(state_chans):
            r = {'cellid': c, 'state_sig': state_list[mod_i],
                 'state_chan': sc, 'modelname': modelname,
                 'g': gain[0,j], 'd': dc[0,j],
                 'MI': state_mod[j],
                 'r': meta['r_test'], 'r_se': meta['se_test'] }
            d = d.append(r, ignore_index=True)



