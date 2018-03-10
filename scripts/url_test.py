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

#options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf',
#         'chancount': 18, 'pupil': False, 'stim': True,
#         'plot_results': True, 'plot_ax': None}
#cellid = 'TAR010c-18-1'
#batch=271

cellid='zee021e-c1'
#cellid='zee022c-a1'
batch=269
options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf',
         'chancount': 18, 'pupil': False, 'stim': True,
         'pertrial': True, 'runclass': 'RDT'}

#options['cellid']=cellid

opts=[]
for i,k in enumerate(options):
    if type(options[k]) is bool:
        opts.append(k+'='+str(int(options[k])))
    else:
        opts.append(k+'='+str(options[k]))
optstring="&".join(opts)

url="http://hyrax.ohsu.edu:3000/baphy/{0}/{1}?{2}".format(batch, cellid, optstring)
print(url)

rec = nems.recording.Recording.load_url(url)
