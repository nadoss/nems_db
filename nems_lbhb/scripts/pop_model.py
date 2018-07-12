# A Template NEMS Script that demonstrates use of xforms for generating
# models that are easy to reload

import os
import logging
import nems
import nems.initializers
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
import nems.uri
import nems.xforms as xforms

from nems.recording import Recording, load_recording, get_demo_recordings
from nems.fitters.api import scipy_minimize


site = "TAR010c"
# site = "BRT033b"
# site = "bbl099g"

# use this line to load recording from server.
#uri = 'http://hearingbrain.org/tmp/'+site+'.tgz'

# alternatively download the file, save and load from local file:
filename=site+'.NAT.fs200.tgz'
recording_path=get_demo_recordings(name=filename)

# uri = '/path/to/recording/' + site + '.tgz'
uri = os.path.join(recording_path, filename)

recordings = [uri]

xfspec = []
xfspec.append(['nems.xforms.load_recordings',
               {'recording_uri_list': recordings}])

xfspec.append(['nems.preprocessing.signal_select_channels',
               {'sig_name': "resp",
                'chans': ['TAR010c-13-1', 'TAR010c-18-2']},
               ['rec'], ['rec']])

xfspec.append(['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': '^STIM_'}])
xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])

# MODEL SPEC
modelspecname = 'wc.18x1.g-fir.1x15x2-lvl.2'

meta = {'cellid': 'TAR010c-18-1', 'batch': 271, 'modelname': modelspecname}

xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])
xfspec.append(['nems.xforms.predict', {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])
xfspec.append(['nems.xforms.plot_summary',    {}])

ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

