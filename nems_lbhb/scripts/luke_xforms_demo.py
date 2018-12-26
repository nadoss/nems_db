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


import nems.xforms as xforms
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import default_keywords
from nems.plugins import default_loaders
from nems.plugins import default_initializers
from nems.plugins import default_fitters
from nems.gui.recording_browser import browse_recording, browse_context


batch = 289
cellid ='BRT026c-46-1'
modelname = "ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1_init-basic"

batch=307
cellid="TAR010c-06-1"

modelname = "psth.fs20.pup-ld-st.pup0.beh0-ref-psthfr_stategain.S_jk.nf10-basic"
modelname = "psth.fs20.pup-ld-st.pup.beh-ref-psthfr_stategain.S_jk.nf10-basic"

autoPlot = True
saveInDB = False
browse_results = False

log.info('Initializing modelspec(s) for cell/batch %s/%d...',
         cellid, int(batch))

# Segment modelname for meta information
kws = modelname.split("_")
modelspecname = "-".join(kws[1:-1])
loadkey = kws[0]
fitkey = kws[-1]

meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
        'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
        'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
        'githash': os.environ.get('CODEHASH', ''),
        'recording': loadkey}

recording_uri = nw.generate_recording_uri(cellid, batch, loadkey)
# code from
# xfspec = xhelp.generate_xforms_spec(recording_uri, modelname, meta)
"""
{'stim': 0, 'chancount': 0, 'pupil': 1, 'rasterfs': 20, 'rawid': None, 'cellid': 'BRT026c-15-1', 'pupil_median': 0, 'pertrial': 0, 'pupil_deblink': 1, 'stimfmt': 'parm', 'runclass': None, 'includeprestim': 1, 'batch': 307}
{'stimfmt': 'parm', 'chancount': 0, 'pupil': 1, 'rasterfs': 20, 'rawid': None, 'cellid': 'BRT026c-15-1', 'pupil_median': 0, 'pertrial': 0, 'pupil_deblink': 1, 'stim': 0, 'runclass': None, 'includeprestim': 1, 'batch': 307}
"""
log.info('Initializing modelspec(s) for recording/model {0}/{1}...'
         .format(recording_uri, modelname))

# parse modelname and assemble xfspecs for loader and fitter
load_keywords, model_keywords, fit_keywords = modelname.split("_")

xforms_kwargs = {'cellid': cellid, 'batch': int(batch)}
xforms_lib = KeywordRegistry(**xforms_kwargs)
xforms_lib.register_modules([default_loaders, default_fitters,
                             default_initializers])
xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

keyword_lib = KeywordRegistry()
keyword_lib.register_module(default_keywords)
keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

# Generate the xfspec, which defines the sequence of events
# to run through (like a packaged-up script)
xfspec = []

# 1) Load the data
xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

# 2) generate a modelspec
xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': model_keywords, 'meta': meta,
                'registry': keyword_lib}])

# 3) fit the data
xfspec.extend(xhelp._parse_kw_string(fit_keywords, xforms_lib))

# 4) add some performance statistics
xfspec.append(['nems.xforms.predict', {}])
xfspec.append(['nems.xforms.add_summary_statistics', {}])

# 5) generate plots
if autoPlot:
    log.info('Generating summary plot...')
    xfspec.append(['nems.xforms.plot_summary', {}])

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
# log.info('Saving modelspec(s) to {0} ...'.format(destination))
# xforms.save_analysis(destination,
#                      recording=ctx['rec'],
#                      modelspecs=modelspecs,
#                      xfspec=xfspec,
#                      figures=ctx['figures'],
#                      log=log_xf)

# save in database as well
if saveInDB:
    # TODO : db results finalized?
    nd.update_results_table(modelspecs[0])

if browse_results:
    aw = browse_context(ctx, signals=['stim', 'pred', 'resp'])
