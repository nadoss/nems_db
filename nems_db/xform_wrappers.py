# -*- coding: utf-8 -*-
# wrapper code for fitting models

import os
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

import nems
import nems.initializers
import nems.epoch as ep
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.metrics.api
import nems.analysis.api
import nems.utils
import nems_db.baphy as nb
import nems_db.db as nd
from nems.recording import Recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems_lbhb.old_xforms.xform_wrappers import generate_recording_uri as ogru
import nems_lbhb.old_xforms.xforms as oxf
import nems_lbhb.old_xforms.xform_helper as oxfh

import logging
log = logging.getLogger(__name__)


def get_recording_file(cellid, batch, options={}):

    options["batch"] = batch
    options["cellid"] = cellid
    uri = nb.baphy_data_path(**options)

    return uri


def get_recording_uri(cellid, batch, options={}):

    opts = []
    for i, k in enumerate(options):
        if type(options[k]) is bool:
            opts.append(k+'='+str(int(options[k])))
        elif type(options[k]) is list:
            pass
        else:
            opts.append(k+'='+str(options[k]))
    optstring = "&".join(opts)

    url = "http://hyrax.ohsu.edu:3000/baphy/{0}/{1}?{2}".format(
                batch, cellid, optstring)
    return url


def generate_recording_uri(cellid=None, batch=None, loadkey=None, siteid=None):
    """
    required parameters (passed through to nb.baphy_data_path):
        cellid: string or list
            string can be a valid cellid or siteid
            list is a list of cellids from a single(?) site
        batch: integer

    figure out filename (or eventually URI) of pre-generated
    NEMS-format recording for a given cell/batch/loader string

    very baphy-specific. Needs to be coordinated with loader processing
    in nems.xform_helper
    """

    # TODO: A lot of the parsing is copy-pasted from nems_lbhb/loaders/,
    #       need to figure out which role goes where and delete the
    #       repeated code.

    options = {}

    def _parm_helper(fs, pupil):
        options = {'rasterfs': fs, 'stimfmt': 'parm',
                   'chancount': 0, 'stim': False}

        if pupil:
            pup_med = 2.0 if fs == 10 else 0.5
            options.update({'pupil': True, 'pupil_deblink': True,
                            'pupil_median': pup_med})
        else:
            options['pupil'] = False

        return options

    # remove any preprocessing keywords in the loader string.
    loader = nems.utils.escaped_split(loadkey, '-')[0]
    log.info('loader=%s',loader)

    if loader.startswith('ozgf'):
        pattern = re.compile(r'^ozgf\.fs(\d{1,})\.ch(\d{1,})([a-zA-Z\.]*)$')
        parsed = re.match(pattern, loader)
        # TODO: fs and chans useful for anything for the loader? They don't
        #       seem to be used here, only in the baphy-specific stuff.
        fs = int(parsed.group(1))
        chans = int(parsed.group(2))
        ops = parsed.group(3)
        pupil = ('pup' in ops) if ops is not None else False

        options = {'rasterfs': fs, 'includeprestim': True,
                   'stimfmt': 'ozgf', 'chancount': chans}

        if pupil:
            options.update({'pupil': True, 'stim': True, 'pupil_deblink': True,
                            'pupil_median': 2})

    elif loader.startswith('nostim'):
        raise(DeprecationWarning)
        pattern = re.compile(r'^nostim\.fs(\d{1,})([a-zA-Z\.]*)?$')
        parsed = re.match(pattern, loader)
        fs = parsed.group(1)
        ops = parsed.group(2)
        pupil = ('pup' in ops)

    elif loader.startswith('parm'):
        pattern = re.compile(r'^parm\.fs(\d{1,})([a-zA-Z\.]*)?$')
        parsed = re.match(pattern, loader)
        fs = parsed.group(1)
        ops = parsed.group(2)
        pupil = ('pup' in ops)

        options.update(_parm_helper(fs, pupil))
        options['stimfmt'] = 'parm'
        options['stim'] = True

    elif loader.startswith('ns'):
        pattern = re.compile(r'^ns\.fs(\d{1,})')
        parsed = re.match(pattern, loader)
        fs = parsed.group(1)
        pupil = ('pup' in loadkey)

        options.update(_parm_helper(fs, pupil))

    elif loader.startswith('psth'):
        pattern = re.compile(r'^psth\.fs(\d{1,})([a-zA-Z0-9\.]*)?$')
        parsed = re.match(pattern, loader)
        fs = parsed.group(1)
        ops = parsed.group(2)
        pupil = ('pup' in ops)

        options.update(_parm_helper(fs, pupil))

    elif loader.startswith('evt'):
        pattern = re.compile(r'^evt\.fs(\d{1,})([a-zA-Z0-9\.]*)?$')
        parsed = re.match(pattern, loader)
        fs = parsed.group(1)
        ops = parsed.group(2)
        pupil = ('pup' in ops)

        options.update(_parm_helper(fs, pupil))

    elif loader.startswith('env'):
        pattern = re.compile(r'^env\.fs(\d{1,})([a-zA-Z0-9\.]*)$')
        parsed = re.match(pattern, loader)
        fs = parsed.group(1)
        ops = parsed.group(2)  # nothing relevant here yet?

        options.update({'rasterfs': fs, 'stimfmt': 'envelope', 'chancount': 0})

    else:
        raise ValueError('unknown loader string: %s' % loader)

    if siteid is not None:
        options['siteid'] = siteid

    # check for use of new loading key (ldb - load baphy) - recording_uri
    # will point towards cached recording holding all stable cells at that
    # site/batch
    # else will load the rec_uri for the single cell specified in fn args
    if 'ldb' in loadkey:
        options['batch'] = batch
        options['recache'] = options.get('recache', False)
        if type(cellid) is not list:
            cellid = [cellid]
        if re.search(r'\d+$', cellid[0]) is None:
            options['site'] = cellid[0]
        else:
            options['site'] = cellid[0][:-5]
        recording_uri = nb.baphy_load_multichannel_recording(**options)
    else:
        recording_uri = get_recording_file(cellid, batch, options)
        #recording_uri = get_recording_uri(cellid, batch, options)

    return recording_uri


def fit_model_xforms_baphy(cellid, batch, modelname,
                           autoPlot=True, saveInDB=False):
    """
    Fits a single NEMS model using data from baphy/celldb
    eg, 'ozgf100ch18_wc18x1_lvl1_fir15x1_dexp1_fit01'
    generates modelspec with 'wc18x1_lvl1_fir15x1_dexp1'

    based on this function in nems/scripts/fit_model.py
       def fit_model(recording_uri, modelstring, destination):

     xfspec = [
        ['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
        ['nems.xforms.add_average_sig', {'signal_to_average': 'resp',
                                         'new_signalname': 'resp',
                                         'epoch_regex': '^STIM_'}],
        ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
        ['nems.xforms.init_from_keywords', {'keywordstring': modelspecname}],
        ['nems.xforms.set_random_phi',  {}],
        ['nems.xforms.fit_basic',       {}],
        # ['nems.xforms.add_summary_statistics',    {}],
        ['nems.xforms.plot_summary',    {}],
        # ['nems.xforms.save_recordings', {'recordings': ['est', 'val']}],
        ['nems.xforms.fill_in_default_metadata',    {}],
    ]
    """

    log.info('Initializing modelspec(s) for cell/batch %s/%d...',
             cellid, int(batch))

    # Segment modelname for meta information
    kws = nems.utils.escaped_split(modelname, '_')
    old = False
    if (len(kws) > 3) or ((len(kws) == 3) and kws[1].startswith('stategain')
                          and not kws[1].startswith('stategain.')):
        # Check if modelname uses old format.
        log.info("Using old modelname format ... ")
        old = True
        modelspecname = nems.utils.escaped_join(kws[1:-1], '_')
    else:
        modelspecname = nems.utils.escaped_join(kws[1:-1], '-')
    loadkey = kws[0]
    fitkey = kws[-1]

    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loadkey}

    if old:
        recording_uri = ogru(cellid, batch, loadkey)
        xfspec = oxfh.generate_loader_xfspec(loadkey, recording_uri)
        xfspec.append(['nems_lbhb.old_xforms.xforms.init_from_keywords',
                       {'keywordstring': modelspecname, 'meta': meta}])
        xfspec.extend(oxfh.generate_fitter_xfspec(fitkey))
        xfspec.append(['nems.analysis.api.standard_correlation', {},
                       ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])
        if autoPlot:
            log.info('Generating summary plot ...')
            xfspec.append(['nems.xforms.plot_summary', {}])
    else:
        uri_key = nems.utils.escaped_split(loadkey, '-')[0]
        recording_uri = generate_recording_uri(cellid, batch, uri_key)
        xfspec = xhelp.generate_xforms_spec(recording_uri, modelname, meta)

    # actually do the fit
    ctx, log_xf = xforms.evaluate(xfspec)

    # save some extra metadata
    modelspecs = ctx['modelspecs']

    destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
            batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
    modelspecs[0][0]['meta']['modelpath'] = destination
    modelspecs[0][0]['meta']['figurefile'] = destination+'figure.0000.png'
    modelspecs[0][0]['meta'].update(meta)

    # save results
    log.info('Saving modelspec(s) to {0} ...'.format(destination))
    save_data = xforms.save_analysis(destination,
                                     recording=ctx['rec'],
                                     modelspecs=modelspecs,
                                     xfspec=xfspec,
                                     figures=ctx['figures'],
                                     log=log_xf)
    savepath = save_data['savepath']

    # save in database as well
    if saveInDB:
        # TODO : db results finalized?
        nd.update_results_table(modelspecs[0])

    return savepath


def fit_pop_model_xforms_baphy(cellid, batch, modelname, saveInDB=False):
    """
    Fits a NEMS population model using baphy data
    """

    log.info("Preparing pop model: ({0},{1},{2})".format(
            cellid, batch, modelname))

    # Segment modelname for meta information
    kws = modelname.split("_")
    modelspecname = "-".join(kws[1:-1])

    loadkey = kws[0]
    fitkey = kws[-1]
    if type(cellid) is list:
        disp_cellid="_".join(cellid)
    else:
        disp_cellid=cellid

    meta = {'batch': batch, 'cellid': disp_cellid, 'modelname': modelname,
            'loader': loadkey, 'fitkey': fitkey,
            'modelspecname': modelspecname,
            'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loadkey}

    uri_key = nems.utils.escaped_split(loadkey, '-')[0]
    recording_uri = generate_recording_uri(cellid, batch, uri_key)

    # pass cellid information to xforms so that loader knows which cells
    # to load from recording_uri
    xfspec = xhelp.generate_xforms_spec(recording_uri, modelname, meta,
                                        xforms_kwargs={'cellid': cellid})

    # actually do the fit
    ctx, log_xf = xforms.evaluate(xfspec)

    # save some extra metadata
    modelspecs = ctx['modelspecs']

    destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
            batch, disp_cellid, ms.get_modelspec_longname(modelspecs[0]))
    modelspecs[0][0]['meta']['modelpath'] = destination
    modelspecs[0][0]['meta']['figurefile'] = destination+'figure.0000.png'
    modelspecs[0][0]['meta'].update(meta)

    # extra thing to save for pop model
    modelspecs[0][0]['meta']['cellids'] = ctx['val'][0]['resp'].chans

    # save results
    log.info('Saving modelspec(s) to {0} ...'.format(destination))
    save_data = xforms.save_analysis(destination,
                                     recording=ctx['rec'],
                                     modelspecs=modelspecs,
                                     xfspec=xfspec,
                                     figures=ctx['figures'],
                                     log=log_xf)
    savepath = save_data['savepath']

    if saveInDB:
        # save in database as well
        nd.update_results_table(modelspecs[0])

    return savepath


def load_model_baphy_xform(cellid, batch=271,
        modelname="ozgf100ch18_wcg18x2_fir15x2_lvl1_dexp1_fit01",
        eval_model=True):

    kws = nems.utils.escaped_split(modelname, '_')
    old = False
    if (len(kws) > 3) or ((len(kws) == 3) and kws[1].startswith('stategain')
                          and not kws[1].startswith('stategain.')):
        # Check if modelname uses old format.
        log.info("Using old modelname format ... ")
        old = True

    d = nd.get_results_file(batch, [modelname], [cellid])
    filepath = d['modelpath'][0]

    if old:
        return oxf.load_analysis(filepath, eval_model=eval_model)
    else:
        return xforms.load_analysis(filepath, eval_model=eval_model)


def model_pred_comp(cellid, batch, modelnames, occurrence=0,
                    pre_dur=None, dur=None):

    modelcount = len(modelnames)
    epoch = 'REFERENCE'
    c = 0

    legend = ['1','2','3','act']
    times = []
    values = []
    values_all = []
    r_test = []
    for i, m in enumerate(modelnames):
        xf, ctx = load_model_baphy_xform(cellid, batch, m)

        val = ctx['val'][0]

        if i==0:
            d = val['resp'].get_epoch_bounds('PreStimSilence')
            if len(d):
                PreStimSilence = np.mean(np.diff(d))
            else:
                PreStimSilence = 0
            if pre_dur is None:
                pre_dur = PreStimSilence

            # Get values from specified occurrence and channel
            extracted = val['resp'].extract_epoch(epoch)
            r_vector = extracted[occurrence][c]
            r_vector = nems.utils.smooth(r_vector, window_len=7)[3:-3]

            r_all = val['resp'].as_continuous()

            # Convert bins to time (relative to start of epoch)
            # TODO: want this to be absolute time relative to start of data?
            time_vector = np.arange(0, len(r_vector)) / val['resp'].fs - PreStimSilence

            # limit time range if specified
            good_bins = (time_vector >= -pre_dur)
            if dur is not None:
                good_bins[time_vector > dur] = False

        extracted = val['pred'].extract_epoch(epoch)
        p_vector = extracted[occurrence][c] + i + 1
        p_all = val['pred'].as_continuous()
        p_all = p_all[0,np.isfinite(r_all[0,:])]

        times.append(time_vector[good_bins])
        values.append(p_vector[good_bins])
        values_all.append(p_all)

        r_test.append(ctx['modelspecs'][0][0]['meta']['r_test'][0])

    times.append(time_vector[good_bins])
    values.append(r_vector[good_bins])
    values_all.append(r_all[0,np.isfinite(r_all[0,:])])

    cc12 = np.corrcoef(values_all[0], values_all[1])[0, 1]
    cc13 = np.corrcoef(values_all[0], values_all[2])[0, 1]
    cc23 = np.corrcoef(values_all[1], values_all[2])[0, 1]

    ccd12 = np.corrcoef(values_all[0]-values_all[3],
                        values_all[1]-values_all[3])[0, 1]
    ccd13 = np.corrcoef(values_all[0]-values_all[3],
                        values_all[2]-values_all[3])[0, 1]
    ccd23 = np.corrcoef(values_all[1]-values_all[3],
                        values_all[2]-values_all[3])[0, 1]

    print("CC LN-GC: {:.3f}  LN-STP: {:.3f} STP-GC: {:.3f}".format(
            cc12,cc13,cc23))
    print("CCd LN-GC: {:.3f}  LN-STP: {:.3f} STP-GC: {:.3f}".format(
            ccd12,ccd13,ccd23))

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    extracted = val['stim'].extract_epoch(epoch)
    sg = extracted[occurrence]
    nplt.plot_spectrogram(sg, val['resp'].fs, ax=ax,
                          title='{} Stim {}'.format(cellid, occurrence),
                          time_offset=PreStimSilence)

    ax = plt.subplot(2, 1, 2)
    title = 'Preds LN {:.3f} GC {:.3f} STP {:.3f} /CC LN-GC: {:.3f}  LN-STP: {:.3f} STP-GC: {:.3f}'.format(
            r_test[0],r_test[1],r_test[2],cc12,cc13,cc23)
    nplt.plot_timeseries(times, values, ax=ax, legend=legend, title=title)

    plt.tight_layout()
"""

    legend = [s.name for s in signals]
    times = []
    values = []
    for s, o, c in zip(signals, occurrences, channels):
        # Get occurrences x chans x time
        extracted = s.extract_epoch(epoch)
        # Get values from specified occurrence and channel
        value_vector = extracted[o][c]
        # Convert bins to time (relative to start of epoch)
        # TODO: want this to be absolute time relative to start of data?
        time_vector = np.arange(0, len(value_vector)) / s.fs - PreStimSilence

        # limit time range if specified
        good_bins = (time_vector >= -PreStimSilence)
        if dur is not None:
            good_bins[time_vector > dur] = False

        times.append(time_vector[good_bins])
        values.append(value_vector[good_bins])

    plot_timeseries(times, values, xlabel, ylabel, legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)
"""



def load_batch_modelpaths(batch, modelnames, cellids=None, eval_model=True):
    d = nd.get_results_file(batch, [modelnames], cellids=cellids)
    return d['modelpath'].tolist()


def quick_inspect(cellid="chn020f-b1", batch=271,
                  modelname="ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"):

    xf, ctx = load_model_baphy_xform(cellid, batch, modelname, eval_model=True)

    modelspecs = ctx['modelspecs']
    est = ctx['est']
    val = ctx['val']
    nplt.quickplot(ctx)

    return modelspecs, est, val
