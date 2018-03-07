#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:26:48 2018

@author: svd
"""

import nems_db.baphy
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

rec=nems_db.baphy.baphy_load_recording(cellid,batch,options)

tresp=rec['resp'].extract_epoch('TRIAL')
tstate=rec['state'].extract_epoch('TRIAL')
plt.figure()
plt.plot(np.nanmean(tresp[:,0,:],axis=0))
plt.plot(np.nanmean(tstate[:,0,:],axis=0))

