#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:13:33 2018

@author: svd
"""

# scripts for showing changes in response across behabior blocks

from nems_lbhb.stateplots import model_per_time_wrapper
#from nems_lbhb.behavior_pupil_scripts.mod_per_state import get_model_results_per_state_model
exec(open("/auto/users/svd/python/nems_db/nems_lbhb/pupil_behavior_scripts/mod_per_state.py").read())


do_single_cell = True  # if False, do pop summary

if do_single_cell:
    # SINGLE CELL EXAMPLES

    # DEFAULTS
    #loader= "psth.fs20.pup-ld-"
    #fitter = "_jk.nf20-basic"
    basemodel = "-ref-psthfr_stategain.S"

    # alternative state / file breakdowns
    state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
    #state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
    #state_list = ['st.pup0.far0.hit0.hlf0','st.pup0.far0.hit0.hlf',
    #              'st.pup.far.hit.hlf0','st.pup.far.hit.hlf']

    # individual cells - by default

    cellid="TAR010c-27-2"  # behavior
    cellid="TAR010c-06-1"  # pupil cell
    cellid="BRT026c-20-1"
    cellid="BRT026c-05-2"
    cellid="TAR010c-19-1"  #
    cellid="TAR010c-33-1"
    cellid="bbl102d-01-1"  # maybe good hybrid cell?
    batch=307

    model_per_time_wrapper(cellid, batch, basemodel=basemodel,
                           state_list=state_list)

else:
    # POPULATION SUMMARY CELL EXAMPLES
    # use basemodel = "-ref-psthfr.s_sdexp.S" for better accuracy and
    # statistical power

    #state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
    #states = ['PASSIVE_0',  'ACTIVE_1','PASSIVE_1',  'ACTIVE_2','PASSIVE_2']

    state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
    states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
              'PASSIVE_1_A','PASSIVE_1_B', 'ACTIVE_2_A','ACTIVE_2_B',
              'PASSIVE_2_A','PASSIVE_2_B']
    #states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
    #          'PASSIVE_1_A','PASSIVE_1_B']

    basemodel = "-ref-psthfr.s_sdexp.S"
    batch = 307

    df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

    dMI, dMI0 = hlf_analysis(df, state_list)


