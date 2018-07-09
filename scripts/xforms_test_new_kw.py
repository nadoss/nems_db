#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import io

#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils

#import nems_db.baphy as nb
import nems_db.db as nd
import nems_db.xform_wrappers as nw
from nems_lbhb.old_xforms.xform_wrappers import generate_recording_uri as ogru
import nems_lbhb.old_xforms.xform_helper as oxfh
import logging
log = logging.getLogger(__name__)


#cellid = 'por016d-b1'
#batch = 259
#modelname = 'env.100_dlog-wc.2x1.c-fir.1x15-lvl.1-dexp.1_basic'

cellid = 'TAR010c-18-1'
batch = 271
modelname = ('ozgf.fs100.ch18-ld_'
             'wc.18x1.g-fir.1x15-lvl.1-dexp.1_'
             'mt.shr-jk.nf5-iter-pred')

#cellid = 'ley046f-01-1'
#batch = 309
#modelname = "psth.20-pupbeh_stategain3_basic-nf"

#cellid = 'sti019b-d1'
#batch = 274

#modelname = 'env.100.m.beh_wc.2x15-fir.15x2-lvl.2-mrg_basic.st.nf5'
#modelname = 'envm100beh_rep2_fir2x2x15_lvl2_mrg_state01-jkm'
#modelname = 'envm100beh_fir2x15_lvl1_rep2_dexp2_mrg_state01-jkm'
#modelname = 'env100beh_fir2x15_lvl1_rep2_dexp2_mrg_state01-jk'
#modelname = 'env100beh_dlogn2_wcc2x1_rep2_stp2_fir2x1x15_lvl2_dexp2_mrg_state01-jk'

#cellid = 'ley041l-a1'
#batch = 303
#modelname = "evt20pupbehtarlic_firNx40_lvl1_stategain3_basic-nf"

#cellid = 'TAR010c-18-1'
#batch = 289
#modelname = 'ozgf100ch18pup_dlog_wcg18x1_fir1x15_lvl1_dexp1_basic-nf'

#batch = 307
#cellid = 'TAR010c-60-1'
#cellid = 'BRT026c-16-1'
#modelname = 'evt20pup0beh0tarlic_firNx40_lvl1_stategain3_basic-nf'
#modelname = 'evt20pupbehtarlic0_firNx40_lvl1_stategain3_basic-nf'


autoPlot = True
saveInDB = False

log.info('Initializing modelspec(s) for cell/batch %s/%d...',
         cellid, int(batch))

# Segment modelname for meta information
kws = modelname.split("_")
old = False
if (len(kws) > 3) or ((len(kws) == 3) and kws[1].startswith('stategain')):
    # Check if modelname uses old format.
    log.info("Using old modelname format ... ")
    old = True
    modelspecname = '_'.join(kws[1:-1])
else:
    modelspecname = "-".join(kws[1:-1])
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
    recording_uri = nw.generate_recording_uri(cellid, batch, loadkey)
    xfspec = xhelp.generate_xforms_spec(recording_uri, modelname, meta)


# Create a log stream set to the debug level; add it as a root log handler
log_stream = io.StringIO()
ch = logging.StreamHandler(log_stream)
ch.setLevel(logging.DEBUG)
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
ch.setFormatter(formatter)
rootlogger = logging.getLogger()
rootlogger.addHandler(ch)

ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

# Close the log, remove the handler, and add the 'log' string to context
log.info('Done (re-)evaluating xforms.')
ch.close()
rootlogger.removeFilter(ch)

log_xf = log_stream.getvalue()


# save some extra metadata
modelspecs = ctx['modelspecs']

destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
        batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
modelspecs[0][0]['meta']['modelpath'] = destination
modelspecs[0][0]['meta']['figurefile'] = destination+'figure.0000.png'

# save results
#log.info('Saving modelspec(s) to {0} ...'.format(destination))
#xforms.save_analysis(destination,
#                     recording=ctx['rec'],
#                     modelspecs=modelspecs,
#                     xfspec=xfspec,
#                     figures=ctx['figures'],
#                     log=log_xf)

# save in database as well
if saveInDB:
    # TODO : db results finalized?
    nd.update_results_table(modelspecs[0])




