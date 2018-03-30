# -*- coding: utf-8 -*-
# wrapper code for fitting models

import os
import random
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

import logging
log = logging.getLogger(__name__)


def get_recording_file(cellid,batch,options={}):

    options["batch"]=batch
    options["cellid"]=cellid
    uri = nb.baphy_data_path(options)
    return uri

def get_recording_uri(cellid,batch,options={}):

    opts=[]
    for i,k in enumerate(options):
        if type(options[k]) is bool:
            opts.append(k+'='+str(int(options[k])))
        elif type(options[k]) is list:
            pass
        else:
            opts.append(k+'='+str(options[k]))
    optstring="&".join(opts)

    url="http://hyrax.ohsu.edu:3000/baphy/{0}/{1}?{2}".format(
                batch, cellid, optstring)
    return url

def generate_loader_xfspec(cellid,batch,loader):

    options = {}
    if loader == "ozgf100ch18":
        options["stimfmt"] = "ozgf"
        options["chancount"] = 18
        options["rasterfs"] = 100
        options['includeprestim'] = 1
        options["average_stim"]=True
        options["state_vars"]=[]
        #recording_uri = get_recording_uri(cellid,batch,options)
        recording_uri = get_recording_file(cellid,batch,options)
        recordings = [recording_uri]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences',{}]]

    elif loader == "ozgf100ch18pup":
        options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf',
          'chancount': 18, 'pupil': True, 'stim': True,
          'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"]=False
        options["state_vars"]=['pupil']
        recording_uri = get_recording_file(cellid,batch,options)
        recordings = [recording_uri]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.xforms.make_state_signal', {'state_signals': ['pupil'], 'permute_signals': [], 'new_signalname': 'state'}]]

    elif loader == "nostim10pup":
        options={'rasterfs': 10, 'includeprestim': True, 'stimfmt': 'parm',
          'chancount': 0, 'pupil': True, 'stim': False,
          'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"]=False
        options["state_vars"]=['pupil']
        recording_uri = get_recording_file(cellid,batch,options)
        recordings = [recording_uri]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.preprocessing.make_state_signal', {'state_signals': ['pupil'], 'permute_signals': [], 'new_signalname': 'state'},['rec'],['rec']]]

    elif loader in ["nostim10pup0beh0","nostim10pup0beh","nostim10pupbeh0","nostim10pupbeh"]:
        options={'rasterfs': 10, 'includeprestim': True, 'stimfmt': 'parm',
          'chancount': 0, 'pupil': True, 'stim': False,
          'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"]=False
        options["state_vars"]=['pupil']

        state_signals=['pupil','behavior_state']
        if loader=="nostim10pup0beh0":
            permute_signals=['pupil','behavior_state']
        elif loader=="nostim10pup0beh":
            permute_signals=['pupil']
        elif loader=="nostim10pupbeh0":
            permute_signals=['behavior_state']
        else:
            permute_signals=['']

        recording_uri = get_recording_file(cellid,batch,options)
        recordings = [recording_uri]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.xforms.make_state_signal', {'state_signals': state_signals, 'permute_signals': permute_signals, 'new_signalname': 'state'}]]

    elif loader in ["nostim20pup0beh0","nostim20pup0beh","nostim20pupbeh0","nostim20pupbeh"]:
        options={'rasterfs': 20, 'includeprestim': True, 'stimfmt': 'parm',
          'chancount': 0, 'pupil': True, 'stim': False,
          'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"]=False
        options["state_vars"]=['pupil']

        state_signals=['pupil','behavior_state']
        if loader=="nostim20pup0beh0":
            permute_signals=['pupil','behavior_state']
        elif loader=="nostim20pup0beh":
            permute_signals=['pupil']
        elif loader=="nostim20pupbeh0":
            permute_signals=['behavior_state']
        else:
            permute_signals=['']

        recording_uri = get_recording_file(cellid,batch,options)
        recordings = [recording_uri]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.preprocessing.make_state_signal', {'state_signals': state_signals, 'permute_signals': permute_signals, 'new_signalname': 'state'},['rec'],['rec']]]

    elif loader == "env100":
        options["stimfmt"] = "envelope"
        options["chancount"] = 0
        options["rasterfs"] = 100
        options['includeprestim'] = 1
        options["average_stim"]=True
        options["state_vars"]=[]
        recording_uri = get_recording_file(cellid,batch,options)
        recordings = [recording_uri]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences',{}]]

    else:
        raise ValueError('unknown loader string')

    return xfspec


def generate_fitter_xfspec(cellid,batch,fitter):

    xfspec=[]

    # parse the fit spec: Use gradient descent on whole data set(Fast)
    if fitter == "fit01":
        # prefit strf
        log.info("Prefitting STRF without other modules...")
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fitjk01":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife', {'njacks': 5}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fitpjk01":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife', {'njacks': 10}])
        xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold',  {}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fit02":
        # no pre-fit
        log.info("Performing full fit...")
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])
    else:
        raise ValueError('unknown fitter string ' + fitter)

    return xfspec



def fit_model_xforms_baphy(cellid,batch,modelname,
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

    log.info('Initializing modelspec(s) for cell/batch {0}/{1}...'.format(cellid,batch))

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitter = kws[-1]

    if 'CODEHASH' in os.environ.keys():
        githash=os.environ['CODEHASH']
    else:
        githash=""
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitter': fitter, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': githash, 'recording': loader}

    # generate xfspec, which defines sequence of events to load data,
    # generate modelspec, fit data, plot results and save
    xfspec = generate_loader_xfspec(cellid,batch,loader)

    xfspec.append(['nems.xforms.init_from_keywords', {'keywordstring': modelspecname, 'meta': meta}])
    #xfspec.append(['nems.initializers.from_keywords_as_list',
    #               {'keyword_string': modelspecname, 'meta': meta},
    #               [],['modelspecs']])

    xfspec+=generate_fitter_xfspec(cellid,batch,fitter)

    #xfspec.append(['nems.xforms.add_summary_statistics',    {}])
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs'],['modelspecs']])

    if autoPlot:
        # GENERATE PLOTS
        log.info('Generating summary plot...')
        xfspec.append(['nems.xforms.plot_summary',    {}])

    # actually do the fit
    ctx, log_xf = xforms.evaluate(xfspec)

    # save some extra metadata
    modelspecs=ctx['modelspecs']

    destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
            batch,cellid,ms.get_modelspec_longname(modelspecs[0]))
    modelspecs[0][0]['meta']['modelpath']=destination
    modelspecs[0][0]['meta']['figurefile']=destination+'figure.0000.png'

    # save results

    log.info('Saving modelspec(s) to {0} ...'.format(destination))
    xforms.save_analysis(destination,
                         recording=ctx['rec'],
                         modelspecs=modelspecs,
                         xfspec=xfspec,
                         figures=ctx['figures'],
                         log=log_xf)

    # save in database as well
    if saveInDB:
        # TODO : db results
        nd.update_results_table(modelspecs[0])

    return ctx


def load_model_baphy_xform(cellid, batch=271,
               modelname="ozgf100ch18_wcg18x2_fir15x2_lvl1_dexp1_fit01",eval_model=True):

    d=nd.get_results_file(batch,[modelname],[cellid])
    filepath=d['modelpath'][0]
    print("Loading from " + filepath)
    return xforms.load_analysis(filepath,eval_model=eval_model)


def quick_inspect(cellid="chn020f-b1", batch=271,
               modelname="ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"):

    ctx=load_model_baphy_xform(cellid, batch,
               modelname,eval=True)

    modelspecs=ctx['modelspecs']
    est=ctx['est']
    val=ctx['val']
    nplt.plot_summary(val, modelspecs)

    return modelspecs,est,val

"""
# SPN example
cellid='btn144a-c1'
batch=259
modelname="env100_fir15x2_dexp1_fit01"

# A1 RDT example
cellid = 'zee021e-c1'
batch=269
modelname = "ozgf100ch18pt_wc18x1_fir15x1_lvl1_dexp1_fit01"
savepath = fit_model_baphy(cellid=cellid, batch=batch, modelname=modelname,
                           autoPlot=False, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)

# A1 VOC+pupil example
cellid = 'eno053f-a1'
batch=294
modelname = "ozgf100ch18pup_pup_psth_stategain2_fit02"
savepath = fit_model_baphy(cellid=cellid, batch=batch, modelname=modelname,
                           autoPlot=False, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)


# A1 NAT + pupil example
cellid = 'TAR010c-18-1'
batch=289
modelname = "ozgf100ch18_wcg18x2_fir15x2_lvl1_dexp1_fit01"

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)

# IC NAT example
cellid = "bbl031f-a1"
batch=291
modelname = "ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)
"""


# A1 NAT + pupil example
"""
cellid = 'TAR010c-18-1'
batch=289
modelname = "ozgf100ch18pup_wcg18x1_fir1x15_lvl1_stategain2_fitjk01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)

"""


# A1 NAT example
"""

cellid = 'zee019b-b1'
batch=271
modelname = "ozgf100ch18_dlog_wcg18x1_fir1x15_lvl1_dexp1_fit01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)

modelspec,est,val=load_model_baphy(savepath)

"""

# A1 VOC+pup example
"""
cellid = 'eno052d-a1'
batch=294
modelname = "nostim10pup_stategain2_fitpjk01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
"""

"""
cellid = 'TAR010c-06-1'
batch=301
modelname = "nostim10pup_stategain2_fitpjk01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
"""
