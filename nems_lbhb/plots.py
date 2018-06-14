#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:05:34 2018

@author: svd
"""
import numpy as np
import matplotlib.pyplot as plt

import nems_db.xform_wrappers as nw
import nems.plots.api as nplt
import nems.xforms as xforms
import nems.epoch as ep
from nems.utils import find_module

params = {'legend.fontsize': 8,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8}
plt.rcParams.update(params)


def get_model_preds(cellid, batch, modelname):
    xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname,
                                        eval_model=False)
    ctx, l = xforms.evaluate(xf, ctx, stop=-1)

    return xf, ctx


def compare_model_preds(cellid, batch, modelname1, modelname2):
    """
    compare prediction accuracy of two models on validation stimuli

    borrows a lot of functionality from nplt.quickplot()

    """
    xf1, ctx1 = get_model_preds(cellid, batch, modelname1)
    xf2, ctx2 = get_model_preds(cellid, batch, modelname2)

    rec = ctx1['rec']
    val1 = ctx1['val'][0]
    val2 = ctx2['val'][0]

    stim = rec['stim'].rasterize()
    resp = rec['resp'].rasterize()
    pred1 = val1['pred']
    pred2 = val2['pred']

    d = resp.get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d))
    d = resp.get_epoch_bounds('PostStimSilence')
    PostStimSilence = np.mean(np.diff(d))

    epoch_regex = "^STIM_"
    stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

    r = resp.as_matrix(stim_epochs)
    s = stim.as_matrix(stim_epochs)
    repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
    max_rep_id, = np.where(repcount == np.max(repcount))

    # keep a max of two stimuli
    stim_ids = max_rep_id[:2]
    stim_count = len(stim_ids)
    # print(max_rep_id)

    # stim_i=max_rep_id[-1]
    # print("Max rep stim={} ({})".format(stim_i, stim_epochs[stim_i]))

    p1 = pred1.as_matrix(stim_epochs)
    p2 = pred2.as_matrix(stim_epochs)

    ms1 = ctx1['modelspecs'][0]
    ms2 = ctx2['modelspecs'][0]
    r_test1 = ms1[0]['meta']['r_test']
    r_test2 = ms2[0]['meta']['r_test']

    fh = plt.figure(figsize=(16, 6))
    ax = plt.subplot(5, 2, 1)
    nplt.strf_timeseries(ms1, ax=ax, clim=None, show_factorized=True,
                         title="{}/{} rtest={:.3f}".format(cellid,modelname1,r_test1),
                         fs=resp.fs)
    ax = plt.subplot(5, 2, 2)
    nplt.strf_timeseries(ms2, ax=ax, clim=None, show_factorized=True,
                      title="{}/{} rtest={:.3f}".format(cellid,modelname2,r_test2),
                      fs=resp.fs)

    if find_module('stp', ms1):
        ax = plt.subplot(5, 2, 3)
        nplt.before_and_after_stp(ms1, sig_name='pred', ax=ax, title='',
                                  channels=0, xlabel='Time (s)', ylabel='STP',
                                  fs=resp.fs)
    if find_module('stp', ms2):
        ax = plt.subplot(5, 2, 4)
        nplt.before_and_after_stp(ms2, sig_name='pred', ax=ax, title='',
                                  channels=0, xlabel='Time (s)', ylabel='STP',
                                  fs=resp.fs)

    for i, stim_i in enumerate(stim_ids):

        ax = plt.subplot(5, 2, 5+i)
        if s.shape[2] <= 2:
            nplt.timeseries_from_vectors(
                    [s[stim_i, 0, 0, :], s[max_rep_id[-1], 0, 1, :]],
                    fs=stim.fs, time_offset=PreStimSilence, ax=ax,
                    title="{}/{} rfit={:.3f}/{:.3f}".format(cellid,
                           stim_epochs[stim_i], r_test1, r_test2))
        else:
            nplt.plot_spectrogram(
                    s[stim_i, 0, :, :],
                    fs=stim.fs, time_offset=PreStimSilence, ax=ax,
                    title="{}/{} rfit={:.3f}/{:.3f}".format(
                            cellid, stim_epochs[stim_i], r_test1, r_test2))
        ax.get_xaxis().set_visible(False)

        ax = plt.subplot(5, 2, 7+i)
        _r = r[stim_i, :, 0, :]
        t = np.arange(_r.shape[-1]) / resp.fs - PreStimSilence - 0.5/resp.fs
        nplt.raster(t, _r)
        ax.get_xaxis().set_visible(False)

        ax = plt.subplot(5, 2, 9+i)
        nplt.timeseries_from_vectors(
                [np.nanmean(_r, axis=0), p1[stim_i, 0, 0, :],
                 p2[stim_i, 0, 0, :]],
                fs=resp.fs, time_offset=PreStimSilence, ax=ax)

    # plt.tight_layout()
    return fh, ctx2


def scatter_comp(beta1, beta2, n1='model1', n2='model2', hist_bins=20,
                 hist_range=[-1, 1], title='modelname/batch',
                 highlight=None):
    """
    beta1, beta2 are T x 1 vectors
    scatter plot comparing beta1 vs. beta2
    histograms of marginals
    """
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)

    # exclude cells without prepassive
    outcells = ((beta1 > hist_range[1]) | (beta1 < hist_range[0]) |
                (beta2 > hist_range[1]) | (beta2 < hist_range[0]))
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    beta1[beta1 > hist_range[1]] = hist_range[1]
    beta1[beta1 < hist_range[0]] = hist_range[0]
    beta2[beta2 > hist_range[1]] = hist_range[1]
    beta2[beta2 < hist_range[0]] = hist_range[0]

    if highlight is None:
        set1 = goodcells
        set2 = []
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    fh = plt.figure(figsize=(8, 6))

    plt.subplot(2, 2, 3)
    plt.plot(beta1[outcells], beta2[outcells], '.', color='red')
    plt.plot(beta1[set2], beta2[set2], '.', color='lightgray')
    plt.plot(beta1[set1], beta2[set1], 'k.')
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(np.array(hist_range), np.array(hist_range), 'k--')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)

    ax = plt.subplot(2, 2, 2)
    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
             bins=hist_bins-1, range=[-hist_range[1]/2, hist_range[1]/2],
             histtype='bar', stacked=True,
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} sterr={:.3f}'.
              format(np.mean(beta2[goodcells]-beta1[goodcells]),
                     np.std(beta2[goodcells]-beta1[goodcells])/np.sqrt(np.sum(goodcells))))
    plt.xlabel('difference')

    plt.tight_layout()

    return fh
