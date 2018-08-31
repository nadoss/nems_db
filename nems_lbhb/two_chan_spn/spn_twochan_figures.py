#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
import matplotlib.pyplot as plt

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems_db.db as nd
import nems_db.params
import numpy as np

import nems_lbhb.stateplots as stateplots
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
import nems_db.xform_wrappers as nw

cellid="chn002h-a1"
batch=259
uri=nw.generate_recording_uri(cellid,batch,"env100")

modelname1="env100_dlog_fir2x15_lvl1_dexp1_basic"
#modelname2="env100_dlog_stp2_fir2x15_lvl1_dexp1_basic"
modelname2="env100_dlog_wcc2x2_stp2_fir2x15_lvl1_dexp1_basic"
modelname2="env100_dlog_wcc2x3_stp3_fir3x15_lvl1_dexp1_basic-shr"
xf1, ctx1 = nw.load_model_baphy_xform(cellid, batch, modelname1,
                                      eval_model=False)
ctx1, l = xforms.evaluate(xf1, ctx1, stop=-1)
xf2, ctx2 = nw.load_model_baphy_xform(cellid, batch, modelname2,
                                      eval_model=False)
ctx2, l = xforms.evaluate(xf2, ctx2, stop=-1)

rec=ctx1['rec']
val1=ctx1['val'][0]
val2=ctx2['val'][0]

resp = rec['resp'].rasterize()
pred1 = val1['pred']
pred2 = val2['pred']

epoch_regex = "^STIM_"

stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

r = resp.as_matrix(stim_epochs)
repcount = np.sum(np.isfinite(r[:,:,0,0]), axis=1)
max_rep_id, = np.where(repcount == np.max(repcount))

p1 = pred1.as_matrix(stim_epochs)
p2 = pred2.as_matrix(stim_epochs)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(r[max_rep_id[-1],:,0,:], aspect='auto')
plt.title("{}/{}".format(cellid,batch))
plt.subplot(2,1,2);
plt.plot(np.nanmean(r[max_rep_id[-1],:,0,:], axis=0))
plt.plot(p1[max_rep_id[-1],0,0,:])
plt.plot(p2[max_rep_id[-1],0,0,:])
plt.legend(('actual',modelname1,modelname2))

