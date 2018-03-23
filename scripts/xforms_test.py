#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import matplotlib.pyplot as plt
import numpy as np
import os

#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms

#import nems_db.baphy as nb
import nems_db.db as nd
import nems_db.xform_wrappers as nw

import logging
log = logging.getLogger(__name__)

"""
#cellid = 'zee021e-c1'
cellid = 'TAR010c-18-1'
#cellid = 'BRT033b-01-1'

example_list=['TAR010c-06-1','TAR010c-22-1',
              'TAR010c-60-1','TAR010c-44-1',
              'bbl074g-a1','bbl081d-a1','BRT006d-a1','BRT007c-a1',
              'BRT009a-a1','BRT015c-a1','BRT016f-a1','BRT017g-a1']

cellid='TAR010c-22-1'
batch=301
modelname = "nostim20pup0beh0_stategain3_fitpjk01"

ctx=load_model_baphy_xform(cellid, batch,modelname)

"""
cellid = 'BRT034f-01-1'
batch=271
modelname = "ozgf100ch18_wcg18x1_fir1x15_lvl1_dexp1_fit01"


autoPlot=True
saveInDB=True

log.info('Initializing modelspec(s) for cell/batch {0}/{1}...'.format(cellid,batch))

# parse modelname
kws = modelname.split("_")
loader = kws[0]
modelspecname = "_".join(kws[1:-1])
fitter = kws[-1]

# figure out some meta data to save in the model spec
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
xfspec = nw.generate_loader_xfspec(cellid,batch,loader)

xfspec.append(['nems.xforms.init_from_keywords', {'keywordstring': modelspecname, 'meta': meta}])

xfspec += nw.generate_fitter_xfspec(cellid,batch,fitter)

xfspec.append(['nems.xforms.add_summary_statistics',    {}])

if autoPlot:
    # GENERATE PLOTS
    log.info('Generating summary plot...')
    xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
#ctx, log_xf = xforms.evaluate(xfspec)
# Evaluate the xforms
ctx={}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

# save some extra metadata
modelspecs=ctx['modelspecs']

destination = '/auto/data/tmp/modelspecs/{0}/{1}/{2}/'.format(
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

# save some extra metadata
modelspecs=ctx['modelspecs']
val=ctx['val'][0]

#plt.figure();
#plt.plot(val['resp'].as_continuous().T)
#plt.plot(val['pred'].as_continuous().T)
#if 'state' in val.signals.keys():
#    plt.plot(val['state'].as_continuous().T/100)
#


