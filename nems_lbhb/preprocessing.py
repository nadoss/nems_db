#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nems_lbhb. state initializers

Created on Fri Aug 31 12:50:49 2018

@author: svd
"""
import re


def append_difficulty(rec, **kwargs):

    newrec = rec.copy()

    newrec['puretone_trials'] = resp.epoch_to_signal('PURETONE_BEHAVIOR')
    newrec['puretone_trials'].chans = ['puretone_trials']
    newrec['easy_trials'] = resp.epoch_to_signal('EASY_BEHAVIOR')
    newrec['easy_trials'].chans = ['easy_trials']
    newrec['hard_trials'] = resp.epoch_to_signal('HARD_BEHAVIOR')
    newrec['hard_trials'].chans = ['hard_trials']


def mask_high_repetion_stims(rec,epoch_regex='^STIM_'):
    full_rec = rec.copy()
    stims = (full_rec.epochs['name'].value_counts() > 9)
    stims = [stims.index[i] for i, s in enumerate(stims) if bool(re.search(epoch_regex, stims.index[i])) and s == True]
    full_rec = full_rec.or_mask(stims)
    #full_rec = full_rec.apply_mask()
    #full_rec = full_rec.and_mask(['REFERENCE'])
    return full_rec
