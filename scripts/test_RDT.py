#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:26:48 2018

@author: svd
"""

import nems_db.baphy as nb
import nems_db.wrappers as nw
import matplotlib.pyplot as plt
import numpy as np


cellid='zee021e-c1'
batch=269
options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 
         'chancount': 18, 'pupil': False, 'stim': True,
         'pupil_deblink': True, 'pupil_median': 1,
         'plot_results': True, 'plot_ax': None}

options['pertrial']=True
options['runclass']='RDT'
options['cellid']=cellid

rec=nb.baphy_load_recording(cellid,batch,options)

tresp=rec['resp'].extract_epoch('TRIAL')
tstate=rec['state'].extract_epoch('TRIAL')
plt.figure()
plt.plot(np.nanmean(tresp[:,0,:],axis=0))
plt.plot(np.nanmean(tstate[:,0,:],axis=0))

# do a model fit
cellid='zee021e-c1'
batch=269
modelname = "ozgf100ch18pt_wc18x1_fir15x1_lvl1_dexp1_fit01"
savepath = nw.fit_model_baphy(cellid=cellid, batch=batch, modelname=modelname, 
                           autoPlot=True, saveInDB=True)
modelspec,est,val=nw.quick_inspect(cellid,batch,modelname)
