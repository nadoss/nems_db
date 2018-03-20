#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import nems_db.baphy as nb
import nems_db.wrappers as nw
import matplotlib.pyplot as plt
import numpy as np
import nems.recording
import nems.xforms as xforms


cellid='eno052d-a1'
batch=294
modelname='nostim100pup_stategain2_fitjk01'
print('Initializing modelspec(s) for cell/batch {0}/{1}...'.format(cellid,batch))

# parse modelname
kws = modelname.split("_")
loader = kws[0]
modelspecname = "_".join(kws[1:-1])
fitter = kws[-1]

options={'rasterfs': 10, 'includeprestim': True, 'stimfmt': 'parm',
      'chancount': 0, 'pupil': True, 'stim': False,
      'pupil_deblink': True, 'pupil_median': 1}
recording_uri = nw.get_recording_uri(cellid,batch,options)
recordings = [recording_uri]

# generate xfspec, which defines sequence of events to load data,
# generate modelspec, fit data, plot results and save

xfspec=[]
xfspec.append(['nems.xforms.load_recordings', {'recording_uri_list': recordings}])

xfspec.append(['nems.xforms.init_from_keywords', {'keywordstring': modelspecname}])

xfspec.append(['nems.xforms.make_state_signal', {'state_signals': ['pupil'], 'permute_signals': [], 'new_signalname': 'state'}])

xfspec.append(['nems.xforms.split_for_jackknife', {'njacks': 10}])

xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold',  {}])

#xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_nfold', {}])

xfspec.append(['nems.xforms.add_summary_statistics',    {}])

#xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
ctx, log_xf = xforms.evaluate(xfspec)

# save some extra metadata
modelspecs=ctx['modelspecs']
val=ctx['val']

plt.figure();
plt.plot(val['resp'].as_continuous().T)
plt.plot(val['pred'].as_continuous().T)
plt.plot(val['state'].as_continuous().T/100)


#if 'CODEHASH' in os.environ.keys():
#    githash=os.environ['CODEHASH']
#else:
#    githash=""
#meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
#        'loader': loader, 'fitter': fitter, 'modelspecname': modelspecname,
#        'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
#        'githash': githash, 'recording': loader}
#if not 'meta' in modelspecs[0][0].keys():
#    modelspecs[0][0]['meta'] = meta
#else:
#    modelspecs[0][0]['meta'].update(meta)
#destination = '/auto/data/tmp/modelspecs/{0}/{1}/{2}/'.format(
#        batch,cellid,ms.get_modelspec_longname(modelspecs[0]))
#modelspecs[0][0]['meta']['modelpath']=destination
#modelspecs[0][0]['meta']['figurefile']=destination+'figure.0000.png'

## save results
#xforms.save_analysis(destination,
#                     recording=ctx['rec'],
#                     modelspecs=modelspecs,
#                     xfspec=xfspec,
#                     figures=ctx['figures'],
#                     log=log_xf)
#log.info('Saved modelspec(s) to {0} ...'.format(destination))
#
## save in database as well
#if saveInDB:
#    # TODO : db results
#    nd.update_results_table(modelspecs[0])
#
#return ctx



