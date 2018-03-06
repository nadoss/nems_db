#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:26:48 2018

@author: svd
"""

import nems_db.baphy

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