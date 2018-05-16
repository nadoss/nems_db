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

params = {'legend.fontsize': 'small',
          'figure.figsize': (8, 6),
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small'}
params = {'legend.fontsize': 8,
          'figure.figsize': (8, 6),
         'axes.labelsize': 8,
         'axes.titlesize': 8,
         'xtick.labelsize': 8,
         'ytick.labelsize': 8}
plt.rcParams.update(params)

def beta_comp(beta1, beta2, n1='model1', n2='model2', hist_bins=20,
              hist_range=[-1, 1], title='modelname/batch',
              highlight=None):
    """
    beta1, beta2 are T x 1 vectors
    scatter plot comparing beta1 vs. beta2
    histograms of marginals
    """

    # exclude cells without prepassive
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    if highlight is None:
        set1 = goodcells
        set2 = goodcells * 0
    else:
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 1)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(np.array(hist_range), np.array(hist_range), 'k--')
    plt.plot(beta1[set1], beta2[set1], 'k.')
    plt.plot(beta1[set2], beta2[set2], '.', color='gray')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    plt.subplot(2, 2, 3)
    plt.hist([beta1[set2], beta1[set1]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True)
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)

    plt.subplot(2, 2, 2)
    plt.hist([beta2[set2], beta2[set1]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True)
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)

    plt.tight_layout()


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


def pb_model_plot(cellid='TAR010c-06-1', batch=301):
    """ test for pupil-behavior interaction """

    fittype = "basic-nf"
    pretype = "psth"
    # pretype = "psths"
    modelname_p0b0 = pretype + "20pup0beh0_stategain3_" + fittype
    modelname_p0b = pretype + "20pup0beh_stategain3_" + fittype
    modelname_pb0 = pretype + "20pupbeh0_stategain3_" + fittype
    modelname_pb = pretype + "20pupbeh_stategain3_" + fittype

    # xf_p0b0, ctx_p0b0 = nw.load_model_baphy_xform(cellid, batch, modelname_p0b0,
    #                                               eval_model=False)
    # ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, stop=-2)

    xf_p0b, ctx_p0b = nw.load_model_baphy_xform(cellid, batch, modelname_p0b,
                                                eval_model=False)
    ctx_p0b, l = xforms.evaluate(xf_p0b, ctx_p0b, stop=-2)

    xf_pb0, ctx_pb0 = nw.load_model_baphy_xform(cellid, batch, modelname_pb0,
                                                eval_model=False)
    ctx_pb0, l = xforms.evaluate(xf_pb0, ctx_pb0, stop=-2)

    xf_pb, ctx_pb = nw.load_model_baphy_xform(cellid, batch, modelname_pb,
                                              eval_model=False)
    ctx_pb, l = xforms.evaluate(xf_pb, ctx_pb, stop=-2)

    val = ctx_pb['val'][0].copy()

    # val['pred_p0b0'] = ctx_p0b0['val'][0]['pred'].copy()
    val['pred_p0b'] = ctx_p0b['val'][0]['pred'].copy()
    val['pred_pb0'] = ctx_pb0['val'][0]['pred'].copy()

    plt.figure(figsize=(6,6))
    ax = plt.subplot(4, 1, 1)
    nplt.state_vars_timeseries(val, ctx_pb['modelspecs'][0])
    ax.set_title("{}/{} - {}".format(cellid, batch, modelname_pb))

    state_var_list = val['state'].chans
    for i, var in enumerate(state_var_list):
        ax = plt.subplot(4, len(state_var_list), 4+i)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred_p0b",
                                       state_sig=var, ax=ax)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        if i == 0:
            ax.set_ylabel('Behavior-only', fontsize=10)
        else:
            ax.yaxis.label.set_visible(False)
        ax.set_title("{} g={:.3f} b={:.3f}".format(var.lower(),
                     ctx_p0b['modelspecs'][0][0]['phi']['g'][i],
                     ctx_p0b['modelspecs'][0][0]['phi']['d'][i]),
                     fontsize=10)

        ax = plt.subplot(4, len(state_var_list), 7+i)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred_pb0",
                                       state_sig=var, ax=ax)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        if i == 0:
            ax.set_ylabel('Pupil-only', fontsize=10)
        else:
            ax.yaxis.label.set_visible(False)
        ax.set_title("{} g={:.3f} b={:.3f}".format(var.lower(),
                     ctx_pb0['modelspecs'][0][0]['phi']['g'][i],
                     ctx_pb0['modelspecs'][0][0]['phi']['d'][i]),
                     fontsize=10)

        ax = plt.subplot(4, len(state_var_list), 10+i)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred",
                                       state_sig=var, ax=ax)
        if i == 0:
            ax.set_ylabel('Full', fontsize=10)
        else:
            ax.yaxis.label.set_visible(False)
        if var == 'active':
            ax.legend(('pas', 'act'))
        ax.set_title("{} g={:.3f} b={:.3f}".format(var.lower(),
                     ctx_pb['modelspecs'][0][0]['phi']['g'][i],
                     ctx_pb['modelspecs'][0][0]['phi']['d'][i]),
                     fontsize=10)

    plt.tight_layout()


def pp_model_plot(cellid='TAR010c-06-1', batch=301):
    """ test for pre-post effects """
    fittype = "basic-nf"
    pretype = "psth"
    # pretype = "psths"

    modelname_p0b0 = pretype + "20pup0pre0beh_stategain4_" + fittype
    modelname_p0b = pretype + "20pup0prebeh_stategain4_" + fittype
    modelname_pb0 = pretype + "20puppre0beh_stategain4_" + fittype
    modelname_pb = pretype + "20pupprebeh_stategain4_" + fittype

    factor1 = "Act+Pre"
    factor2 = "Act+Pup"

    # xf_p0b0, ctx_p0b0 = nw.load_model_baphy_xform(cellid, batch, modelname_p0b0,
    #                                               eval_model=False)
    # ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, stop=-2)

    xf_p0b, ctx_p0b = nw.load_model_baphy_xform(cellid, batch, modelname_p0b,
                                                eval_model=False)
    ctx_p0b, l = xforms.evaluate(xf_p0b, ctx_p0b, stop=-2)

    xf_pb0, ctx_pb0 = nw.load_model_baphy_xform(cellid, batch, modelname_pb0,
                                                eval_model=False)
    ctx_pb0, l = xforms.evaluate(xf_pb0, ctx_pb0, stop=-2)

    xf_pb, ctx_pb = nw.load_model_baphy_xform(cellid, batch, modelname_pb,
                                              eval_model=False)
    ctx_pb, l = xforms.evaluate(xf_pb, ctx_pb, stop=-2)

    val = ctx_pb['val'][0].copy()

    # val['pred_p0b0'] = ctx_p0b0['val'][0]['pred'].copy()
    val['pred_p0b'] = ctx_p0b['val'][0]['pred'].copy()
    val['pred_pb0'] = ctx_pb0['val'][0]['pred'].copy()

    plt.figure(figsize=(8,6))
    ax = plt.subplot(4, 1, 1)
    nplt.state_vars_timeseries(val, ctx_pb['modelspecs'][0])
    ax.set_title("{}/{} - {}".format(cellid, batch, modelname_pb))

    state_var_list = val['state'].chans
    L = len(state_var_list)
    for i, var in enumerate(state_var_list):

        # FACTOR 1 (excluding FACTOR 2)
        ax = plt.subplot(4, L, L+1+i)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred_p0b",
                                       state_sig=var, ax=ax)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        if i == 0:
            ax.set_ylabel("{} r={:.3f}".format(factor1,
                          ctx_p0b['modelspecs'][0][0]['meta']['r_test']),
                          fontsize=8)
        else:
            ax.yaxis.label.set_visible(False)
        ax.set_title("{} g={:.3f} b={:.3f}".format(var.lower(),
                     ctx_p0b['modelspecs'][0][0]['phi']['g'][i],
                     ctx_p0b['modelspecs'][0][0]['phi']['d'][i]),
                     fontsize=8)

        # FACTOR 2 (excluding FACTOR 1)
        ax = plt.subplot(4, L, 2*L+1+i)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred_pb0",
                                       state_sig=var, ax=ax)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        if i == 0:
            ax.set_ylabel("{} r={:.3f}".format(factor2,
                          ctx_pb0['modelspecs'][0][0]['meta']['r_test']),
                          fontsize=8)
        else:
            ax.yaxis.label.set_visible(False)
        ax.set_title("{} g={:.3f} b={:.3f}".format(var.lower(),
                     ctx_pb0['modelspecs'][0][0]['phi']['g'][i],
                     ctx_pb0['modelspecs'][0][0]['phi']['d'][i]),
                     fontsize=8)

        # FULL MODEL
        ax = plt.subplot(4, L, 3*L+1+i)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred",
                                       state_sig=var, ax=ax)
        if i == 0:
            ax.set_ylabel("{} r={:.3f}".format('Full',
                          ctx_pb['modelspecs'][0][0]['meta']['r_test']),
                          fontsize=8)
        else:
            ax.yaxis.label.set_visible(False)
        if var == 'active':
            ax.legend(('pas', 'act'), fontsize=8)
        ax.set_title("{} g={:.3f} b={:.3f}".format(var.lower(),
                     ctx_pb['modelspecs'][0][0]['phi']['g'][i],
                     ctx_pb['modelspecs'][0][0]['phi']['d'][i]),
                     fontsize=8)

    plt.tight_layout()
