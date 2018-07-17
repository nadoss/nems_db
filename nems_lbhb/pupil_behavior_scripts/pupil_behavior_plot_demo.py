#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:59:40 2018

@author: hellerc
"""

''' 
Demo plotting pupil-behavior results from fit results folder
click event on scatter plot will load png for that cell
'''


import nems_lbhb.stateplots as sp
import pandas as pd

results_path = '/auto/users/svd/projects/pupil-behavior/pb_AC_301_psth.fs20_psthfr_sdexp.S_jk.nf10-init.st-basic'
df = pd.read_csv(results_path+'/results.csv')
beta1 = 'r_pup'
beta2 = 'r_beh'
n1 = 'pupil'
n2 = 'active'
hist_range = [-0.05, .1]
title = 'unique pred'


cellids = ['BRT036b-38-1']
highlight = []
for c in df['cellid']:
    if c in cellids:
        highlight.append(1)
    else:
        highlight.append(0)

fh = sp.beta_comp_from_folder(beta1=beta1, beta2=beta2, n1=n1, n2=n2, title=title,
                              hist_range=hist_range, folder=results_path, highlight=highlight)

