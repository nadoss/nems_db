#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))

import nems_db.db as nd
import nems_db.params
import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems_db.xform_wrappers as nw
import nems.recording as recording
import nems.epoch as ep
import nems.plots.api as nplt

# PSTH figures

nems_dir = os.path.abspath(os.path.dirname(recording.__file__) + '/..')
signals_dir = nems_dir + '/recordings'

# without DB
# uri = signals_dir + "/BRT026c-02-1.tgz"

# with DB
cellid = "TAR010c-06-1"
batch = 301
loader = "psth200pup0beh0"
uri = nw.generate_recording_uri(cellid, batch, loader)


rec = recording.load_recording(uri)

# move code below to
# stateplots.psth_per_file(rec)


file_epochs = ep.epoch_names_matching(rec.epochs, "^FILE_")

epoch_regex = "^STIM_"
stim_epochs = ep.epoch_names_matching(rec.epochs, epoch_regex)

r = []
max_rep_id = []
for i, f in enumerate(file_epochs):
    resp = rec['resp'].rasterize()

    epoch_indices0 = ep.epoch_intersection(
            resp.get_epoch_indices(f),
            resp.get_epoch_indices('PASSIVE_EXPERIMENT'))
    if epoch_indices0.size == 0:
        epoch_indices0 = ep.epoch_intersection(
                resp.get_epoch_indices('HIT_TRIAL'),
                resp.get_epoch_indices(f))

    epoch_indices = ep.epoch_intersection(
            resp.get_epoch_indices('REFERENCE'),
            epoch_indices0)

    # Only takes the first of any conflicts (don't think I actually need this)
    epoch_indices = ep.remove_overlap(epoch_indices)
    epoch_times = epoch_indices / resp.fs

    # add adjusted signals to the recording
    new_resp = resp.select_times(epoch_times)

    r.append(new_resp.as_matrix(stim_epochs) * resp.fs)

    repcount = np.sum(np.isfinite(r[-1][:, :, 0, 0]), axis=1)
    n, = np.where(repcount == np.max(repcount))
    max_rep_id.append(n[-1])

plt.figure()

d = rec['resp'].get_epoch_bounds('PreStimSilence')
PreStimSilence = np.mean(np.diff(d))
d = rec['resp'].get_epoch_bounds('PostStimSilence')
PostStimSilence = np.mean(np.diff(d))

for i, _r in enumerate(r):
    repcount = np.sum(np.isfinite(r[-1][:, :, 0, 0]), axis=1)
    med_rep = int(np.median(repcount))

    t = np.arange(_r.shape[-1]) / resp.fs - PreStimSilence - 0.5/resp.fs

    ax = plt.subplot(4, len(r), i+1)
    nplt.raster(t, _r[max_rep_id[0], :, 0, :], ax=ax, title=file_epochs[i])
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'g--')
    ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs), ylim, 'g--')

    ax = plt.subplot(4, len(r), len(r)+i+1)
    nplt.psth_from_raster(t, _r[max_rep_id[0], :, 0, :], ax=ax, title='raster',
                          ylabel='spk/s', binsize=10)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'k--')
    ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs*10), ylim, 'k--')

    ax = plt.subplot(4, len(r), len(r)*2+i+1)
    rall = _r[:, :med_rep, 0, :]
    rall = np.reshape(rall, [rall.shape[0]*med_rep, rall.shape[2]])
    nplt.raster(t, rall, ax=ax, title='raster')
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'g--')
    ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs), ylim, 'g--')

    ax = plt.subplot(4, len(r), len(r)*3+i+1)
    rall = _r[:, :med_rep, 0, :]
    rall = np.reshape(rall, [rall.shape[0] * med_rep, rall.shape[2]])
    nplt.psth_from_raster(t, rall, ax=ax, title='raster',
                          ylabel='spk/s', binsize=10)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'k--')
    ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs*10), ylim, 'k--')

plt.tight_layout()
