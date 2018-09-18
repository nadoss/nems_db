#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
import pandas as pd

import nems_db.params
import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems_db.xform_wrappers as nw
import nems_db.db as nd
import nems_db.params

import nems.recording as recording
import nems.epoch as ep
import nems.plots.api as nplt

sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))




def plot_save_examples(batch, compare, loader, basemodel, fitter, RELOAD=False):

    if batch in [301, 307]:
        area = "AC"
    else:
        area = "IC"

    d = nd.get_batch_cells(batch)
    cellids = list(d['cellid'])

    stats_list = []
    root_path = '/auto/users/svd/projects/pupil-behavior'

    modelset = '{}_{}_{}_{}_{}_{}'.format(compare, area,
                batch, loader, basemodel, fitter)
    out_path = '{}/{}/'.format(root_path, modelset)

    if os.access(root_path, os.W_OK) and not(os.path.exists(out_path)):
        os.makedirs(out_path)

    datafile = out_path + 'results.csv'
    plt.close('all')

    if (not RELOAD) and (not os.path.isfile(datafile)):
        RELOAD = True
        print('datafile not found, reloading')

    if RELOAD:
        for cellid in cellids:
            if compare == "pb":
                fh, stats = stateplots.pb_model_plot(
                        cellid, batch, loader=loader, basemodel=basemodel,
                        fitter=fitter)
            elif compare == "ppas":
                fh, stats = stateplots.ppas_model_plot(
                        cellid, batch, loader=loader, basemodel=basemodel,
                        fitter=fitter)
            else:
                fh, stats = stateplots.pp_model_plot(
                        cellid, batch, loader=loader, basemodel=basemodel,
                        fitter=fitter)

            # fh2 = stateplots.pp_model_plot(cellid,batch)
            stats_list.append(stats)
            if os.access(out_path, os.W_OK):
                fh.savefig(out_path+cellid+'.pdf')
                fh.savefig(out_path+cellid+'.png')
            plt.close(fh)

        col_names = ['cellid', 'r_p0b0', 'r_p0b', 'r_pb0', 'r_pb',
                     'e_p0b0', 'e_p0b', 'e_pb0', 'e_pb',
                     'rf_p0b0', 'rf_p0b', 'rf_pb0', 'rf_pb',
                     'r_pup', 'r_beh', 'r_beh_pup0', 'pup_mod', 'beh_mod',
                     'pup_mod_n', 'beh_mod_n',
                     'pup_mod_beh0', 'beh_mod_pup0',
                     'pup_mod_beh0_n', 'beh_mod_pup0_n',
                     'd_pup','d_beh','g_pup','g_beh',
                     'ref_all_resp', 'ref_common_resp',
                     'tar_max_resp', 'tar_probe_resp']
        df = pd.DataFrame(columns=col_names)

        for stats in stats_list:
            df0 = pd.DataFrame([[stats['cellid'],
                                 stats['r_test'][0], stats['r_test'][1],
                                 stats['r_test'][2], stats['r_test'][3],
                                 stats['se_test'][0], stats['se_test'][1],
                                 stats['se_test'][2], stats['se_test'][3],
                                 stats['r_floor'][0], stats['r_floor'][1],
                                 stats['r_floor'][2], stats['r_floor'][3],
                                 stats['r_test'][3]-stats['r_test'][1],
                                 stats['r_test'][3]-stats['r_test'][2],
                                 stats['r_test'][1]-stats['r_test'][0],
                                 stats['pred_mod'][0, 1], stats['pred_mod'][1, 2],
                                 stats['pred_mod_norm'][0, 1],
                                 stats['pred_mod_norm'][1, 2],
                                 stats['pred_mod_full'][0, 1],
                                 stats['pred_mod_full'][1, 2],
                                 stats['pred_mod_full_norm'][0, 1],
                                 stats['pred_mod_full_norm'][1, 2],
                                 stats['b'][3,1],
                                 stats['b'][3,2],
                                 stats['g'][3,1],
                                 stats['g'][3,2],
                                 stats['ref_all_resp'],
                                 stats['ref_common_resp'],
                                 stats['tar_max_resp'],
                                 stats['tar_probe_resp']
                                 ]], columns=col_names)
            df = df.append(df0)
        df.set_index(['cellid'], inplace=True)
        if os.access(out_path, os.W_OK):
            df.to_csv(datafile)
    else:
        # load cached dataframe
        df = pd.read_csv(datafile, index_col=0)

    sig_mod = list(df['r_pb']-df['e_pb'] > df['r_p0b0'] + df['e_p0b0'])
    if compare == "pb":
        alabel = "active"
    elif compare == "ppas":
        alabel = "each passive"
    else:
        alabel = "pre/post"

    mi_bounds = [-0.4, 0.4]

    fh1 = stateplots.beta_comp(df['r_pup'], df['r_beh'], n1='pupil', n2=alabel,
                               title=modelset+' unique pred',
                               hist_range=[-0.02, 0.15], highlight=sig_mod)
    fh2 = stateplots.beta_comp(df['pup_mod_n'], df['beh_mod'],
                               n1='pupil', n2=alabel,
                               title=modelset+' mod index',
                               hist_range=mi_bounds, highlight=sig_mod)
    fh3 = stateplots.beta_comp(df['beh_mod_pup0'], df['beh_mod'],
                               n1=alabel+'-nopup', n2=alabel,
                               title=modelset+' unique mod',
                               hist_range=mi_bounds, highlight=sig_mod)

    # unique behavior:
    #    performance of full model minus performance behavior shuffled
    # r_beh_with_pupil = df['r_pb'] - df['r_pb0']
    # naive behavior (ignorant of pupil):
    #    performance of behavior alone (pupil shuffled) minus all shuffled
    # r_beh_no_pupil = df['r_p0b'] - df['r_p0b0']

    fh4 = stateplots.beta_comp(df['r_beh_pup0'], df['r_beh'],
                               n1=alabel+'-nopup', n2=alabel,
                               title=modelset+' unique r', hist_range=[-0.02, .15],
                               highlight=sig_mod)

    #fh4 = stateplots.beta_comp(df['r_beh'], df['beh_mod'], n1='pred', n2='mod',
    #                           title='behavior', hist_range=[-0.4, 0.4])
    #fh5 = stateplots.beta_comp(df['r_pup'], df['pup_mod'], n1='pred', n2='mod',
    #                           title='pupil', hist_range=[-0.1, 0.1])

    if os.access(out_path, os.W_OK):
        fh1.savefig(out_path+'summary_pred.pdf')
        fh2.savefig(out_path+'summary_mod.pdf')
        fh3.savefig(out_path+'summary_mod_ctl.pdf')
        fh4.savefig(out_path+'summary_r_ctl.pdf')


# BEGIN main code

# User parameters:
loader = "psth.fs20.pup-ld-"
fitter = "_jk.nf10-init.st-basic"
basemodel = "-ref-psthfr.s_stategain.S"

batch = 307

statevars = ['pup','beh','far.hit']
statevars0 = ['pup0','beh0','far0.hit0']

states = []
for i,s in enumerate(statevars0):
    st = statevars.copy()
    st[i]=s
    states.append("st."+".".join(st))
states.append("st."+".".join(statevars))
modelnames = [loader + s + basemodel + fitter for s in states]


meta=['r_test', 'se_test', 'state_mod']
stats_keys=[]
state_mod = []
r_test = []
for m in modelnames:
    d = nems_db.params.fitted_params_per_batch(batch, m, meta=meta,
                                               stats_keys=stats_keys)
    d = d.reindex(sorted(d.columns), axis=1)
    state_mod.append(np.stack(d.loc['meta--state_mod']))
    r_test.append(np.stack(d.loc['meta--r_test']))

state_mod = np.stack(state_mod)
r_test = np.stack(r_test)

u_state_mod = state_mod[[3],:] - state_mod[:3, :]
u_r_test = r_test[[3],:] - r_test[:3,:]

plt.close('all')
plt.figure()
for i,p in enumerate([(0,1),(0,2),(1,2)]):
    ax = plt.subplot(3,2,i*2+1)
    stateplots.beta_comp(u_r_test[p[0],:], u_r_test[p[1],:],
                         n1=statevars[p[0]], n2=statevars[p[1]],
                         title='u r_test', hist_range=[-0.05, 0.15],
                         ax=ax)

    ax = plt.subplot(3,2,i*2+2)
    stateplots.beta_comp(u_state_mod[p[0],:,p[0]+1], u_state_mod[p[1],:,p[1]+1],
                         n1=statevars[p[0]], n2=statevars[p[1]],
                         title='u state_mod', hist_range=[-0.6, 0.6],
                         ax=ax)

plt.tight_layout()



#batches = [301]
basemodels = ["ref.b-psthfr_sdexp.S", "ref.b-psthfr.s_sdexp.S"]
#basemodels = ["ref.b-psthfr_sdexp.S"]
comparisons = ["pb"]
##
for batch in batches:
    for basemodel in basemodels:
        for compare in comparisons:
            print("plot_save_examples({},{},{},{},{})".format(
                    batch, compare, loader, basemodel, fitter))
            plot_save_examples(batch, compare, loader, basemodel,
                               fitter, RELOAD)
