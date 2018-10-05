#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd

Call this to get the state-dep results:

d = get_model_results(batch=batch, state_list=state_list, basemodel=basemodel)

Special order requred for state_list (which is part of modelname that
defines the state variables):

state_list = [both shuff, pup_shuff, other_shuff, full_model]

Here's how to set parameters:

batch = 307  # A1 SUA and MUA
batch = 309  # IC SUA and MUA

# pup vs. active/passive
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_stategain.S"

# pup vs. per file
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel = "-ref-psthfr.s_stategain.S"

# pup vs. performance
state_list = ['st.pup0.beh.far0.hit0','st.pup0.beh.far.hit',
              'st.pup.beh.far0.hit0','st.pup.beh.far.hit']
basemodel = "-ref.a-psthfr.s_stategain.S"

# pup vs. pre/post passive
state_list = ['st.pup0.pas0','st.pup0.pas',
              'st.pup.pas0','st.pup.pas']
basemodel = "-ref-pas-psthfr.s_stategain.S"

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


def get_model_results_per_state_model(batch=307, state_list=None,
                      loader = "psth.fs20.pup-ld-",
                      fitter = "_jk.nf20-basic",
                      basemodel = "-ref-psthfr.s_stategain.S"):
    """
    loader = "psth.fs20.pup-ld-"
    fitter = "_jk.nf20-basic"
    basemodel = "-ref-psthfr.s_stategain.S"
    state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

    d=get_model_results_per_state_model(batch=307, state_list=state_list,
                                        loader=loader,fitter=fitter,
                                        basemodel=basemodel)

    state_list defaults to
       ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    """

    if state_list is None:
        state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

    modelnames = [loader + s + basemodel + fitter for s in state_list]

    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()

    d = pd.DataFrame(columns=['cellid','modelname','state_sig',
                              'state_chan','MI',
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
                r = {'cellid': c, 'state_chan': sc, 'modelname': m,
                     'state_sig': state_list[mod_i],
                     'g': gain[0, j], 'd': dc[0, j],
                     'MI': state_mod[j],
                     'r': meta['r_test'][0], 'r_se': meta['se_test'][0]}
                d = d.append(r, ignore_index=True)

    #d['r_unique'] = d['r'] - d['r0']
    #d['MI_unique'] = d['MI'] - d['MI0']

    return d



def get_model_results(batch=307, state_list=None,
                      loader = "psth.fs20.pup-ld-",
                      fitter = "_jk.nf20-basic",
                      basemodel = "-ref-psthfr.s_stategain.S"):
    """
    loader = "psth.fs20.pup-ld-"
    fitter = "_jk.nf20-basic"
    basemodel = "-ref-psthfr.s_stategain.S"

    state_list defaults to
       ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    """

    if state_list is None:
        state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

    modelnames = [loader + s + basemodel + fitter for s in state_list]

    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()

    d = pd.DataFrame(columns=['cellid','modelname','state_sig','state_sig0',
                              'state_chan','MI',
                              'r','r_se','d','g','MI0','r0','r0_se'])

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
                ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
                if np.sum(ii) == 0:
                    r = {'cellid': c, 'state_chan': sc}
                    d = d.append(r, ignore_index=True)
                    ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
                if mod_i == 3:
                    # full model
                    d.loc[ii, ['modelname', 'state_sig', 'g', 'd', 'MI',
                               'r', 'r_se']] = \
                       [m, state_list[mod_i], gain[0, j], dc[0, j],
                        state_mod[j], meta['r_test'][0], meta['se_test'][0]]
                elif (mod_i == 1) & (sc == 'pupil'):
                    # pupil shuffled model
                    d.loc[ii, ['state_sig0', 'MI0', 'r0', 'r0_se']] = \
                       [state_list[mod_i], state_mod[j],
                        meta['r_test'][0], meta['se_test'][0]]
                elif (mod_i == 0) & (sc == 'baseline'):
                    d.loc[ii, ['state_sig0', 'r0', 'r0_se']] = \
                       [state_list[mod_i],
                        meta['r_test'][0], meta['se_test'][0]]
                elif (mod_i == 2) & (sc not in ['baseline', 'pupil']):
                    # pupil shuffled model
                    d.loc[ii, ['state_sig0', 'MI0', 'r0', 'r0_se']] = \
                       [state_list[mod_i], state_mod[j],
                        meta['r_test'][0], meta['se_test'][0]]

    d['r_unique'] = d['r'] - d['r0']
    d['MI_unique'] = d['MI'] - d['MI0']

    return d

