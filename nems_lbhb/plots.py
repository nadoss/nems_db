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


def plot_weights_64D(h, cellids, cbar=True):
    
    '''
    given a weight vector, h, plot the weights on the appropriate electrode channel
    mapped based on the cellids supplied. Weight vector must be sorted the same as
    cellids. Channels without weights will be plotted as empty dots. For cases
    where there are more than one unit on a given electrode, additional units will
    be "offset" from the array geometry as additional electrodes.
    '''
    

    if type(cellids) is not np.ndarray:
        cellids = np.array(cellids)
    
    if type(h) is not np.ndarray:
        h = np.array(h)
        vmin = np.min(h)
        vmax = np.max(h)
    else:
        vmin = np.min(h)
        vmax = np.max(h)
     # Make a vector for each column of electrodes
    
    # left column + right column are identical
    lr_col = np.arange(0,21*0.25,0.25)  # 25 micron vertical spacing
    left_ch_nums = np.arange(3,64,3)
    right_ch_nums = np.arange(4,65,3)
    center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)
    center_col = np.arange(-0.25,20.25*.25,0.25)-0.125
    ch_nums = np.hstack((left_ch_nums, center_ch_nums, right_ch_nums))
    sort_inds = np.argsort(ch_nums)
    
    
    
    l_col = np.vstack((np.ones(21)*-0.2,lr_col))
    r_col = np.vstack((np.ones(21)*0.2,lr_col))
    c_col = np.vstack((np.zeros(22),center_col))
    locations = np.hstack((l_col,c_col,r_col))[:,sort_inds]
    plt.figure()
    plt.scatter(locations[0,:],locations[1,:],facecolor='none',edgecolor='k',s=50)
    
    
   
    # Now, color appropriately
    electrodes = np.zeros(len(cellids))
    
    for i in range(0, len(cellids)):
        electrodes[i] = int(cellids[i][-4:-2])
        
    # Add locations for cases where two or greater units on an electrode
    electrodes=list(electrodes-1)  # cellids labeled 1-64, python counts 0-63
    dupes = list(set([int(x) for x in electrodes if electrodes.count(x)>1]))
    print('electrodes with multiple units:')
    print([d+1 for d in dupes])

    num_of_dupes = [electrodes.count(x) for x in electrodes]
    num_of_dupes = list(set([x for x in num_of_dupes if x>1]))
    #max_duplicates = np.max(np.array(num_of_dupes))
    dup_locations=np.empty((2,np.sum(num_of_dupes)*len(dupes)))
    max_=0
    count = 0
    x_shifts = dict.fromkeys([str(i) for i in dupes])
    for i in np.arange(0,len(dupes)):
        loc_x = locations[0,dupes[i]]
        
        x_shifts[str(dupes[i])]=[]
        x_shifts[str(dupes[i])].append(loc_x)
        
        n_dupes = electrodes.count(dupes[i])-1
        shift = 0
        for d in range(0,n_dupes):
            if loc_x < 0:
                shift -= 0.2
            elif loc_x == 0:
                shift += 0.4
            elif loc_x > 0:
                shift += 0.2
            
            m = shift
            if m > max_:
                max_=m
            
            x_shifts[str(dupes[i])].append(loc_x+shift)
      
            count += 1
    count+=len(dupes)
    dup_locations = np.empty((2, count))
    c=0
    h_dupes = []
    for k in x_shifts.keys():
        index = np.argwhere(np.array(electrodes) == int(k))
        for i in range(0, len(x_shifts[k])):
            dup_locations[0,c] = x_shifts[k][i]
            dup_locations[1,c] = locations[1,int(k)]
            h_dupes.append(h[index[i][0]])
            c+=1
        
    plt.scatter(dup_locations[0,:],dup_locations[1,:],facecolor='none',edgecolor='k',s=50)

    plt.axis('scaled')
    plt.xlim(-max_-.3,max_+.3)

    c_id = np.sort([int(x) for x in electrodes if electrodes.count(x)==1])
    electrodes = [int(x) for x in electrodes]

    # find the indexes of the unique cellids
    indexes = np.argwhere(np.array([electrodes.count(x) for x in electrodes])==1)
    indexes2 = np.argwhere(np.array([electrodes.count(x) for x in electrodes])!=1)
    indexes=[x[0] for x in indexes]
    indexes2=[x[0] for x in indexes2]
    
    # make an inverse mask of the unique indexes
    mask = np.ones(len(h),np.bool)
    mask[indexes]=0


    # plot the unique ones
    import matplotlib
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h[indexes])
    #mappable.set_cmap('jet')
    colors = mappable.to_rgba(h[indexes])
    plt.scatter(locations[:,c_id][0,:],locations[:,c_id][1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=50,edgecolor='none')
    # plot the duplicates
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h[mask])
    #mappable.set_cmap('jet')
    #colors = mappable.to_rgba(h[mask])
    colors = mappable.to_rgba(h_dupes)
    plt.scatter(dup_locations[0,:],dup_locations[1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=50,edgecolor='none')
    if cbar is True:
        plt.colorbar(mappable)
