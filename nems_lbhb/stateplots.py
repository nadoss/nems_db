#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:05:34 2018

@author: svd
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image

import nems_db.xform_wrappers as nw
import nems.plots.api as nplt
import nems.xforms as xforms
import nems.epoch as ep
import nems_lbhb.plots as lplt

font_size=8
params = {'legend.fontsize': font_size,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

line_colors = {'actual_psth': (0,0,0),
               'predicted_psth': 'red',
               #'passive': (255/255, 133/255, 133/255),
               'passive': (216/255, 151/255, 212/255),
               #'active': (196/255, 33/255, 43/255),
               'active': (129/255, 64/255, 138/255),
               'false_alarm': (79/255, 114/255, 184/255),
               'miss': (183/255, 196/255, 229/255),
               'hit': (36/255, 49/255, 103/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'pas1': 'green',
               'pas2': (153/255, 124/255, 248/255),
               'pas3': (173/255, 144/255, 255/255),
               'pas4': (193/255, 164/255, 255/255),
               'pas5': 'green',
               'pas6': (123/255, 104/255, 238/255),
               'hard': (196/255, 149/255, 44/255),
               'easy': (255/255, 206/255, 6/255),
               'puretone': (247/255, 223/255, 164/255),
               'large': (44/255, 125/255, 61/255),
               'small': (181/255, 211/255, 166/255)}
fill_colors = {'actual_psth': (.8,.8,.8),
               'predicted_psth': 'pink',
               #'passive': (226/255, 172/255, 185/255),
               'passive': (234/255, 176/255, 223/255),
               #'active': (244/255, 44/255, 63/255),
               'active': (163/255, 102/255, 163/255),
               'false_alarm': (107/255, 147/255, 204/255),
               'miss': (200/255, 214/255, 237/255),
               'hit': (78/255, 92/255, 135/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'hard':  (229/255, 172/255, 57/255),
               'easy': (255/255, 225/255, 100/255),
               'puretone': (255/255, 231/255, 179/255),
               'large': (69/255, 191/255, 89/255),
               'small': (215/255, 242/255, 199/255)}


def beta_comp(beta1, beta2, n1='model1', n2='model2', hist_bins=20,
              hist_range=[-1, 1], title=None,
              highlight=None):
    """
    beta1, beta2 are T x 1 vectors
    scatter plot comparing beta1 vs. beta2
    histograms of marginals
    """

    beta1 = np.array(beta1)
    beta2 = np.array(beta2)

    nncells = np.isfinite(beta1) & np.isfinite(beta2)
    beta1 = beta1[nncells]
    beta2 = beta2[nncells]
    if highlight is not None:
        highlight = np.array(highlight)
        highlight = highlight[nncells]

    if title is None:
        title="n={}/{}".format(np.sum(highlight),len(highlight))

    # exclude cells without prepassive
    outcells = ((beta1 > hist_range[1]) | (beta1 < hist_range[0]) |
                (beta2 > hist_range[1]) | (beta2 < hist_range[0]))
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    beta1[beta1 > hist_range[1]] = hist_range[1]
    beta1[beta1 < hist_range[0]] = hist_range[0]
    beta2[beta2 > hist_range[1]] = hist_range[1]
    beta2[beta2 < hist_range[0]] = hist_range[0]

    if highlight is None:
        set1 = goodcells.astype(bool)
        set2 = (1-goodcells).astype(bool)
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    fh = plt.figure(figsize=(8, 6))

    ax = plt.subplot(2, 2, 3)
    #plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    #plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.axvline(0, color='k', linestyle='--')
    plt.axhline(0, color='k', linestyle='--')
    plt.plot(np.array(hist_range), np.array(hist_range), 'k--')
    plt.plot(beta1[outcells], beta2[outcells], '.', color='red')
    plt.plot(beta1[set2], beta2[set2], '.', color='lightgray')
    plt.plot(beta1[set1], beta2[set1], 'k.')
    plt.axis('equal')
    plt.ylim(hist_range)
    plt.xlim(hist_range)
    #plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 2)
#    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
#              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
#             bins=hist_bins-1, range=[hist_range[0]/2,hist_range[1]/2],
#             histtype='bar', stacked=True,
#             color=['black','lightgray'])

    d=np.sort(np.sign(beta1[goodcells])*(beta2[goodcells]-beta1[goodcells]))
    plt.bar(np.arange(np.sum(goodcells)), d,
            color='black')
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.ylabel('difference')

    plt.tight_layout()
    lplt.ax_remove_box(ax)

    old_title=fh.canvas.get_window_title()
    fh.canvas.set_window_title(old_title+': '+title)

    return fh


def display_png(event, cellids, path):
    ind = event.ind
    if len(ind) > 1:
        ind = [ind[0]]
    else:
        ind = ind
    cell1 = cellids[ind]
    print('cell1: {0}'.format(cell1))
    print(ind)
    # img = mpimg.imread(path+'/'+cell1[0]+'.png')
    # img = plt.imread(path+'/'+cell1[0]+'.png')
    img = Image.open(path+'/'+cell1[0]+'.png')
    img.show(img)


def beta_comp_from_folder(beta1='r_pup', beta2='r_beh',
                          n1='model1', n2='model2', hist_bins=20,
                          hist_range=[-1, 1], title='modelname/batch',
                          folder=None, highlight=None):

    if folder is None:
        raise ValueError('Must specify the results folder!')
    elif folder[-1] == '/':
        folder = folder[:-1]
        
    if highlight is not None:
        highlight = np.array(highlight)

    results = pd.read_csv(folder+'/results.csv')
    cellids = results['cellid'].values

    beta1 = results[beta1].values
    beta2 = results[beta2].values

    nncells = np.isfinite(beta1) & np.isfinite(beta2)
    beta1 = beta1[nncells]
    beta2 = beta2[nncells]

    # exclude cells without prepassive
    outcells = ((beta1 > hist_range[1]) | (beta1 < hist_range[0]) |
                (beta2 > hist_range[1]) | (beta2 < hist_range[0]))
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    beta1[beta1 > hist_range[1]] = hist_range[1]
    beta1[beta1 < hist_range[0]] = hist_range[0]
    beta2[beta2 > hist_range[1]] = hist_range[1]
    beta2[beta2 < hist_range[0]] = hist_range[0]
    
    if highlight is None:
        set1 = goodcells.astype(bool)
        set2 = (1-goodcells).astype(bool)
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))


    fh = plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 3)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(np.array(hist_range), np.array(hist_range), 'k--')
    plt.plot(beta1[set1], beta2[set1], 'k.', picker=3)
    plt.plot(beta1[set2], beta2[set2], '.', color='lightgray', picker=3)
    plt.plot(beta1[outcells], beta2[outcells], '.', color='red')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    def display_wrapper(event):
        
        if sum(set2)==0:
            display_png(event, cellids[set1], folder)
        elif event.mouseevent.button==1: 
            print("Left-click detected, displaying from 'highlighted' cells")
            display_png(event, cellids[set1], folder)
        elif event.mouseevent.button==3:
            print("Right-click detected, loading from 'non-highlighted' cells")
            display_png(event, cellids[set2], folder)

    fh.canvas.mpl_connect('pick_event', lambda event: display_wrapper(event))
    
    
    ax = plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 2)
    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
             bins=hist_bins-1, range=[hist_range[0]/2,hist_range[1]/2],
             histtype='bar', stacked=True,
             color=['black','lightgray'])

#    d=np.sort(np.sign(beta1[goodcells])*(beta2[goodcells]-beta1[goodcells]))
#    plt.bar(np.arange(np.sum(goodcells)), d,
#            color='black')
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.ylabel('difference')

    plt.tight_layout()
    lplt.ax_remove_box(ax)

    old_title=fh.canvas.get_window_title()
    fh.canvas.set_window_title(old_title+': '+title)
    
    return fh

#    plt.subplot(2, 2, 1)
#    plt.hist([beta1[set1]], bins=hist_bins, range=hist_range,
#             histtype='bar', stacked=True,
#             color=['black'])
#    plt.title('mean={:.3f} abs={:.3f}'.
#              format(np.mean(beta1[goodcells]),
#                     np.mean(np.abs(beta1[goodcells]))))
#    plt.xlabel(n1)
#
#    ax = plt.subplot(2, 2, 4)
#    plt.hist([beta2[set1]], bins=hist_bins, range=hist_range,
#             histtype='bar', stacked=True, orientation="horizontal",
#             color=['black'])
#    plt.title('mean={:.3f} abs={:.3f}'.
#              format(np.mean(beta2[goodcells]),
#                     np.mean(np.abs(beta2[goodcells]))))
#    plt.xlabel(n2)
#
#    ax = plt.subplot(2, 2, 2)
#    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1])],
#             bins=hist_bins-1, range=[hist_range[0]/2,hist_range[1]/2],
#             histtype='bar', stacked=True,
#             color=['black'])
#    plt.title('mean={:.3f} abs={:.3f}'.
#              format(np.mean(beta2[goodcells]),
#                     np.mean(np.abs(beta2[goodcells]))))
#    plt.xlabel('difference')
#
#    plt.tight_layout()
#
#    old_title=fh.canvas.get_window_title()
#    fh.canvas.set_window_title(old_title+': '+title)




def beta_comp_cols(g, b, n1='A', n2='B', hist_bins=20,
                  hist_range=[-1,1], title='modelname/batch',
                  highlight=None):

    #exclude cells without prepassive
    goodcells=(np.abs(g[:,0]) > 0) & (np.abs(g[:,1])>0)

    if highlight is None:
        set1=goodcells
        set2=goodcells * 0
    else:
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    plt.figure(figsize=(6,8))

    plt.subplot(3, 2, 1)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(g[set1, 0], g[set1, 1], 'k.')
    plt.plot(g[set2, 0], g[set2, 1], 'b.')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    plt.subplot(3, 2, 3)
    plt.hist(g[goodcells,0],bins=hist_bins,range=hist_range)
    plt.title('mean={:.3f}'.format(np.mean(g[goodcells,0])))
    plt.xlabel(n1)

    plt.subplot(3, 2, 5)
    plt.hist(g[goodcells,1],bins=hist_bins,range=hist_range)
    plt.title('mean={:.3f}'.format(np.mean(g[goodcells,1])))
    plt.xlabel(n2)

    plt.subplot(3, 2, 2)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(b[set1, 0], b[set1, 1], 'k.')
    plt.plot(b[set2, 0], b[set2, 1], 'b.')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title('baseline')

    plt.subplot(3, 2, 4)
    plt.hist(b[goodcells, 0], bins=hist_bins, range=hist_range)
    plt.title('mean={:.3f}'.format(np.mean(b[goodcells, 0])))
    plt.xlabel(n1)

    plt.subplot(3, 2, 6)
    plt.hist(b[goodcells, 1], bins=hist_bins, range=hist_range)
    plt.xlabel(n2)
    plt.title('mean={:.3f}'.format(np.mean(b[goodcells, 1])))
    plt.tight_layout()


def state_mod_index(rec, epoch='REFERENCE', psth_name='resp',
                    state_sig='pupil'):

    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)

    full_var = rec['state'].loc[state_sig]
    folded_var = np.squeeze(full_var.extract_epoch(epoch))

    # compute the mean state for each occurrence
    m = np.nanmean(folded_var, axis=1)

    # compute the mean state across all occurrences
    mean = np.nanmean(m)
    gtidx = (m >= mean)
    ltidx = np.logical_not(gtidx)

    # low = response on epochs when state less than mean
    if np.sum(ltidx):
        low = np.nanmean(folded_psth[ltidx, :, :], axis=0).T
    else:
        low = np.ones(folded_psth[0, :, :].shape).T * np.nan

    # high = response on epochs when state greater than or equal to mean
    if np.sum(gtidx):
        high = np.nanmean(folded_psth[gtidx, :, :], axis=0).T
    else:
        high = np.ones(folded_psth[0, :, :].shape).T * np.nan

    mod = np.sum(high - low) / np.sum(high + low)

    return mod


def _model_step_plot(cellid, batch, modelnames, factors, state_colors=None):
    """
    state_colors : N x 2 list
       color spec for high/low lines in each of the N states
    """
    global line_colors
    global fill_colors

    modelname_p0b0, modelname_p0b, modelname_pb0, modelname_pb = \
       modelnames
    factor0, factor1, factor2 = factors

    xf_p0b0, ctx_p0b0 = nw.load_model_baphy_xform(cellid, batch, modelname_p0b0,
                                                  eval_model=False)
    # ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, stop=-2)

    ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, start=0, stop=-2)

    xf_p0b, ctx_p0b = nw.load_model_baphy_xform(cellid, batch, modelname_p0b,
                                                eval_model=False)
    ctx_p0b, l = xforms.evaluate(xf_p0b, ctx_p0b, start=0, stop=-2)

    xf_pb0, ctx_pb0 = nw.load_model_baphy_xform(cellid, batch, modelname_pb0,
                                                eval_model=False)
    #ctx_pb0['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb0, l = xforms.evaluate(xf_pb0, ctx_pb0, start=0, stop=-2)

    xf_pb, ctx_pb = nw.load_model_baphy_xform(cellid, batch, modelname_pb,
                                              eval_model=False)
    #ctx_pb['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb, l = xforms.evaluate(xf_pb, ctx_pb, start=0, stop=-2)

    val = ctx_pb['val'][0].copy()

    # val['pred_p0b0'] = ctx_p0b0['val'][0]['pred'].copy()
    val['pred_p0b'] = ctx_p0b['val'][0]['pred'].copy()
    val['pred_pb0'] = ctx_pb0['val'][0]['pred'].copy()

    state_var_list = val['state'].chans

    pred_mod = np.zeros([len(state_var_list), 2])
    pred_mod_full = np.zeros([len(state_var_list), 2])
    resp_mod_full = np.zeros([len(state_var_list), 1])

    state_std = np.nanstd(val['state'].as_continuous(), axis=1, keepdims=True)
    for i, var in enumerate(state_var_list):
        if state_std[i]:
            # actual response modulation index for each state var
            resp_mod_full[i] = state_mod_index(val, epoch='REFERENCE',
                                               psth_name='resp', state_sig=var)

            mod2_p0b = state_mod_index(val, epoch='REFERENCE',
                                       psth_name='pred_p0b', state_sig=var)
            mod2_pb0 = state_mod_index(val, epoch='REFERENCE',
                                       psth_name='pred_pb0', state_sig=var)
            mod2_pb = state_mod_index(val, epoch='REFERENCE',
                                      psth_name='pred', state_sig=var)

            pred_mod[i] = np.array([mod2_pb-mod2_p0b, mod2_pb-mod2_pb0])
            pred_mod_full[i] = np.array([mod2_pb0, mod2_p0b])

    pred_mod_norm = pred_mod / (state_std + (state_std == 0).astype(float))
    pred_mod_full_norm = pred_mod_full / (state_std +
                                          (state_std == 0).astype(float))

    if 'each_passive' in factors:
        psth_names_ctl = ["pred_p0b"]
        factors.remove('each_passive')
        for v in state_var_list:
            if v.startswith('FILE_'):
                factors.append(v)
                psth_names_ctl.append("pred_pb0")
    else:
        psth_names_ctl = ["pred_p0b", "pred_pb0"]

    col_count = len(factors) - 1
    if state_colors is None:
        state_colors = [[None, None]]*col_count

    fh = plt.figure()
    ax = plt.subplot(3, 1, 1)
    nplt.state_vars_timeseries(val, ctx_pb['modelspecs'][0],
                               state_colors=[s[1] for s in state_colors])
    ax.set_title("{}/{} - {}".format(cellid, batch, modelname_pb))
    ax.set_ylabel("{} r={:.3f}".format(factor0,
                  ctx_p0b0['modelspecs'][0][0]['meta']['r_test']))
    lplt.ax_remove_box(ax)

    for i, var in enumerate(factors[1:]):
        if var.startswith('FILE_'):
           varlbl = var[5:]
        else:
           varlbl = var
        ax = plt.subplot(3, col_count, col_count+i+1)

        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2=psth_names_ctl[i],
                                       state_sig=var, ax=ax,
                                       colors=state_colors[i])
        if i == 0:
            ax.set_ylabel("Control model")
            ax.set_title("{} ctl r={:.3f}"
                         .format(varlbl.lower(),
                                 ctx_p0b['modelspecs'][0][0]['meta']['r_test']),
                         fontsize=6)
        else:
            ax.yaxis.label.set_visible(False)
            ax.set_title("{} ctl r={:.3f}"
                         .format(varlbl.lower(),
                                 ctx_pb0['modelspecs'][0][0]['meta']['r_test']),
                         fontsize=6)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        lplt.ax_remove_box(ax)

        ax = plt.subplot(3, col_count, col_count*2+i+1)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred",
                                       state_sig=var, ax=ax,
                                       colors=state_colors[i])
        if i == 0:
            ax.set_ylabel("Full Model")
        else:
            ax.yaxis.label.set_visible(False)
        if ax.legend_:
            ax.legend_.remove()

        if psth_names_ctl[i] == "pred_p0b":
            j=0
        else:
            j=1

        ax.set_title("r={:.3f} rawmod={:.3f} umod={:.3f}"
                     .format(ctx_pb['modelspecs'][0][0]['meta']['r_test'],
                             pred_mod_full_norm[i+1][j], pred_mod_norm[i+1][j]),
                     fontsize=6)

        if var == 'active':
            ax.legend(('pas', 'act'))
        elif var == 'pupil':
            ax.legend(('small', 'large'))
        elif var == 'PRE_PASSIVE':
            ax.legend(('act+post', 'pre'))
        elif var.startswith('FILE_'):
            ax.legend(('this', 'others'))
        lplt.ax_remove_box(ax)

    plt.tight_layout()

    stats = {'cellid': cellid,
             'batch': batch,
             'modelnames': modelnames,
             'state_vars': state_var_list,
             'factors': factors,
             'r_test': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_test'],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_test'],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_test'],
                     ctx_pb['modelspecs'][0][0]['meta']['r_test']
                     ]),
             'se_test': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['se_test'],
                     ctx_p0b['modelspecs'][0][0]['meta']['se_test'],
                     ctx_pb0['modelspecs'][0][0]['meta']['se_test'],
                     ctx_pb['modelspecs'][0][0]['meta']['se_test']
                     ]),
             'r_floor': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_pb['modelspecs'][0][0]['meta']['r_floor']
                     ]),
             'pred_mod': pred_mod.T,
             'pred_mod_full': pred_mod_full.T,
             'pred_mod_norm': pred_mod_norm.T,
             'pred_mod_full_norm': pred_mod_full_norm.T,
             'g': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['g'],
                     ctx_p0b['modelspecs'][0][0]['phi']['g'],
                     ctx_pb0['modelspecs'][0][0]['phi']['g'],
                     ctx_pb['modelspecs'][0][0]['phi']['g']]),
             'b': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['d'],
                     ctx_p0b['modelspecs'][0][0]['phi']['d'],
                     ctx_pb0['modelspecs'][0][0]['phi']['d'],
                     ctx_pb['modelspecs'][0][0]['phi']['d']])
        }

    return fh, stats


def pb_model_plot(cellid='TAR010c-06-1', batch=301,
                  loader="psth.fs20", basemodel="stategain.S", fitter="basic.st.nf10"):
    """
    test for pupil-behavior interaction.
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'

    """
    global line_colors

    # modelname_p0b0 = loader + "20pup0beh0_stategain3_" + fitter
    # modelname_p0b = loader + "20pup0beh_stategain3_" + fitter
    # modelname_pb0 = loader + "20pupbeh0_stategain3_" + fitter
    # modelname_pb = loader + "20pupbeh_stategain3_" + fitter
    modelname_p0b0 = loader + "-ld-st.pup0.beh0-ref-" + basemodel + "_" + fitter
    modelname_p0b = loader + "-ld-st.pup0.beh-ref-" + basemodel + "_" + fitter
    modelname_pb0 = loader + "-ld-st.pup.beh0-ref-" + basemodel + "_" + fitter
    modelname_pb = loader + "-ld-st.pup.beh-ref-" + basemodel + "_" + fitter

    factor0 = "baseline"
    factor1 = "pupil"
    factor2 = "active"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]
    state_colors = [[line_colors['small'], line_colors['large']],
                    [line_colors['passive'], line_colors['active']]]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors,
                                 state_colors=state_colors)

    return fh, stats


def pp_model_plot(cellid='TAR010c-06-1', batch=301,
                  loader="psth", basemodel="stategain.N", fitter="basic-nf"):
    """
    test for pre-post effects
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'
    """

    modelname_p0b0 = loader + "20pup0pre0beh_stategain4_" + fitter
    modelname_p0b = loader + "20pup0prebeh_stategain4_" + fitter
    modelname_pb0 = loader + "20puppre0beh_stategain4_" + fitter
    modelname_pb = loader + "20pupprebeh_stategain4_" + fitter

    factor0 = "baseline"
    factor1 = "pupil"
    factor2 = "PRE_PASSIVE"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]
    state_colors = [[line_colors['small'], line_colors['large']],
                    [line_colors['pre'], line_colors['post']],
                    [line_colors['passive'], line_colors['active']]]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors,
                                 state_colors=state_colors)

    plt.tight_layout()

    return fh, stats


def ppas_model_plot(cellid='TAR010c-06-1', batch=301,
                    loader="psth.fs20", basemodel="stategain.S",
                    fitter="basic.st.nf10"):
    """
    test for pre-post effects -- passive only data
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'
    """

    # psth.fs20-st.pup0.pas0-pas_stategain.N_basic.st.nf10
    modelname_p0b0 = loader + "-ld-st.pup0.pas0-ref-pas-" + basemodel + "_" + fitter
    modelname_p0b = loader + "-ld-st.pup0.pas-ref-pas-" + basemodel + "_" + fitter
    modelname_pb0 = loader + "-ld-st.pup.pas0-ref-pas-" + basemodel + "_" + fitter
    modelname_pb = loader + "-ld-st.pup.pas-ref-pas-" + basemodel + "_" + fitter

    factor0 = "baseline"
    factor1 = "pupil"
    factor2 = "each_passive"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]
    state_colors = [[line_colors['small'], line_colors['large']],
                    [line_colors['pas1'], line_colors['post']],
                    [line_colors['pas2'], line_colors['post']],
                    [line_colors['pas3'], line_colors['post']],
                    [line_colors['pas4'], line_colors['post']],
                    [line_colors['pas5'], line_colors['post']],
                    [line_colors['pas6'], line_colors['post']]]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors,
                                 state_colors=state_colors)

    plt.tight_layout()

    return fh, stats


def psth_per_file(rec):

    raise NotImplementedError

    resp = rec['resp'].rasterize()

    file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")

    epoch_regex = "^STIM_"
    stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

    r = []
    max_rep_id = np.zeros(len(file_epochs))
    for f in file_epochs:

        r.append(resp.as_matrix(stim_epochs, overlapping_epoch=f) * resp.fs)

    repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
    max_rep_id, = np.where(repcount == np.max(repcount))

    t = np.arange(r.shape[-1]) / resp.fs

    plt.figure()

    ax = plt.subplot(3, 1, 1)
    nplt.plot_spectrogram(s[max_rep_id[-1],0,:,:], fs=stim.fs, ax=ax,
                          title="cell {} - stim".format(cellid))

    ax = plt.subplot(3, 1, 2)
    nplt.raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster')

    ax = plt.subplot(3, 1, 3);
    nplt.psth_from_raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster',
                          ylabel='spk/s')

    plt.tight_layout()
