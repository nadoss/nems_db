#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd

Call this to get the state-dep results:

d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

Special order requred for state_list (which is part of modelname that
defines the state variables):

state_list = [both shuff, pup_shuff, other_shuff, full_model]

Here's how to set parameters:

batch = 307  # A1 SUA and MUA
batch = 309  # IC SUA and MUA

# pup vs. active/passive
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_stategain.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

# pup vs. per file
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel = "-ref-psthfr.s_stategain.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

# pup vs. performance
state_list = ['st.pup0.beh.far0.hit0','st.pup0.beh.far.hit',
              'st.pup.beh.far0.hit0','st.pup.beh.far.hit']
basemodel = "-ref.a-psthfr.s_stategain.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

# pup vs. pre/post passive
state_list = ['st.pup0.pas0','st.pup0.pas',
              'st.pup.pas0','st.pup.pas']
basemodel = "-ref-pas-psthfr.s_stategain.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

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
        state_list = ['st.pup0.beh0', 'st.pup0.beh',
                      'st.pup.beh0', 'st.pup.beh']

    modelnames = [loader + s + basemodel + fitter for s in state_list]

    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()

    if state_list[-1].endswith('fil') or state_list[-1].endswith('pas'):
        include_AP = True
    else:
        include_AP = False

    d = pd.DataFrame(columns=['cellid', 'modelname', 'state_sig',
                              'state_chan', 'MI',
                              'r', 'r_se', 'd', 'g', 'state_chan_alt'])

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
            a_count=0
            p_count=0
            for j, sc in enumerate(state_chans):
                r = {'cellid': c, 'state_chan': sc, 'modelname': m,
                     'state_sig': state_list[mod_i],
                     'g': gain[0, j], 'd': dc[0, j],
                     'MI': state_mod[j],
                     'r': meta['r_test'][0], 'r_se': meta['se_test'][0]}
                d = d.append(r, ignore_index=True)
                l = len(d) - 1

                if include_AP and sc.startswith("FILE_"):
                    siteid = c.split("-")[0]
                    fn = "%" + sc.replace("FILE_","") + "%"
                    sql = "SELECT * FROM gDataRaw WHERE cellid=%s" +\
                       " AND parmfile like %s"
                    dcellfile = nd.pd_query(sql, (siteid, fn))
                    if dcellfile.loc[0]['behavior'] == 'active':
                        a_count += 1
                        d.loc[l,'state_chan_alt'] = "ACTIVE_{}".format(a_count)
                    else:
                        p_count += 1
                        d.loc[l,'state_chan_alt'] = "PASSIVE_{}".format(p_count)
                else:
                    d.loc[l,'state_chan_alt'] = d.loc[l,'state_chan']



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


def aud_vs_state(df, nb=5, title=None, state_list=None):
    """
    d = dataframe output by get_model_results_per_state_model()
    nb = number of bins
    """

    plt.figure(figsize=(4,6))

    da = df[df['state_chan']=='active']

    dp = da.pivot(index='cellid',columns='state_sig',values=['r','r_se'])

    dr = dp['r'].copy()
    dr['b_unique'] = dr[state_list[3]]**2 - dr[state_list[2]]**2
    dr['p_unique'] = dr[state_list[3]]**2 - dr[state_list[1]]**2
    dr['bp_common'] = dr[state_list[3]]**2 - dr[state_list[0]]**2 - dr['b_unique'] - dr['p_unique']
    dr['bp_full'] = dr['b_unique']+dr['p_unique']+dr['bp_common']
    dr['null']=dr[state_list[0]]**2 * np.sign(dr[state_list[0]])
    dr['full']=dr[state_list[3]]**2 * np.sign(dr[state_list[3]])

    dr['sig']=((dp['r'][state_list[3]]-dp['r'][state_list[0]]) > \
         (dp['r_se'][state_list[3]]+dp['r_se'][state_list[0]]))

    #dm = dr.loc[dr['sig'].values,['null','full','bp_common','p_unique','b_unique']]
    dm = dr.loc[:,['null','full','bp_common','p_unique','b_unique','sig']]
    dm = dm.sort_values(['null'])
    mfull=dm[['null','full','bp_common','p_unique','b_unique','sig']].values

    if nb > 0:
        stepsize=mfull.shape[0]/nb
        mm=np.zeros((nb,mfull.shape[1]))
        for i in range(nb):
            #x0=int(np.floor(i*stepsize))
            #x1=int(np.floor((i+1)*stepsize))
            #mm[i,:]=np.mean(m[x0:x1,:],axis=0)
            x01=(mfull[:,0]>i/nb) & (mfull[:,0]<=(i+1)/nb)
            mm[i,:]=np.nanmean(mfull[x01,:],axis=0)
        print(np.round(mm,3))

        m = mm.copy()
    else:
        # alt to look at each cell individually:
        m = mfull.copy()

    mb=m[:,2:]

    ax=plt.subplot(3,1,1)
    stateplots.beta_comp(mfull[:,0],mfull[:,1],n1='State independent',n2='Full state-dep',
                         ax=ax, highlight=dm['sig'], hist_range=[-0.1, 1])

    plt.subplot(3,1,2)
    ind = np.arange(mb.shape[0])
    width=0.8
    #ind = m[:,0]
    p1 = plt.bar(ind, mb[:,0], width=width)
    p2 = plt.bar(ind, mb[:,1], width=width, bottom=mb[:,0])
    p3 = plt.bar(ind, mb[:,2], width=width, bottom=mb[:,0]+mb[:,1])
    plt.legend(('common','p_unique','b-unique'))
    if title is not None:
        plt.title(title)
    plt.xlabel('behavior-independent quintile')
    plt.ylabel('mean r2')

    plt.subplot(3,1,3)
    ind = np.arange(mb.shape[0])
    #ind = m[:,0]
    p1 = plt.plot(ind, mb[:,0])
    p2 = plt.plot(ind, mb[:,1]+mb[:,0])
    p3 = plt.plot(ind, mb[:,2]+mb[:,0]+mb[:,1])
    plt.legend(('common','p_unique','b-unique'))
    plt.xlabel('behavior-independent quintile')
    plt.ylabel('mean r2')

    plt.tight_layout()


def aud_vs_state_wrapper():

    #batch = 307  # A1 SUA and MUA
    #batch = 309  # IC SUA and MUA

    # pup vs. active/passive
    state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    basemodel = "-ref-psthfr.s_stategain.S"

    #plt.close('all')
    for bi, batch in enumerate([309,307]):
        df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

        aud_vs_state(df, nb=5, title='batch {}'.format(batch),
                     state_list=state_list)

