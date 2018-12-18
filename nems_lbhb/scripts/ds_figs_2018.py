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

#batch = 309 # IC cells SUA+MUA
batch = 307 # A1 cells SUA+MUA
do_single_cell = True  # if False, do pop summary

if do_single_cell:

    # DEFAULTS
    #loader= "psth.fs20.pup-ld-"
    #fitter = "_jk.nf20-basic"
    #basemodel = "-ref-psthfr_stategain.S"
    basemodel = "-ref-psthfr_sdexp.S"

    # alternative state / file breakdowns
    #state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
    #state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
    #state_list = ['st.pup0.far0.hit0.hlf0','st.pup0.far0.hit0.hlf',
    #              'st.pup.far.hit.hlf0','st.pup.far.hit.hlf']
    state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    
    # individual cells - by default
    
    if batch == 309:
        # A1 SU
        cellids=['bbl071d-a1', 'bbl071d-a3', 'bbl074g-a1','bbl078k-a1','bbl081d-a1',
             'BRT005c-a1','BRT006d-a1','BRT006d-a2','BRT007c-a1','BRT007c-a3',
             'BRT009a-a1','BRT009c-a1','BRT015b-a1','BRT015c-a1','BRT016f-a1',
             'BRT017g-a1','ley011c-a1','ley026g-a1','ley027f-a1','ley027g-a1',
             'ley034g-a1','ley035c-a1','ley040j-a1','ley040k-a1','ley041l-a1',
             'ley042i-a1','ley046f-01-1','ley046f-21-1','ley046f-22-1','ley046f-30-1',
             'ley046f-38-1','ley046f-60-1','ley046f-62-1','ley046g-11-1','ley046g-23-1',
             'ley046g-30-1','ley046g-31-2','ley046g-47-1','ley049h-a1','ley049h-a2']
        
        path='/auto/users/daniela/docs/AC_IC_project/309_per_file/'
        
        
    if batch == 307:
        # IC SU 
        cellids=['bbl102d-29-1','bbl102d-31-1','bbl102d-33-1','bbl102d-40-1',
                 'bbl102d-52-1','bbl102d-52-2','bbl102d-60-1','BRT026c-02-1',
                 'BRT026c-05-1','BRT026c-05-2','BRT026c-07-1','BRT026c-08-1',
                 'BRT026c-11-1','BRT026c-11-2','BRT026c-12-1','BRT026c-17-1',
                 'BRT026c-18-1','BRT026c-19-1','BRT026c-20-1','BRT026c-27-1',
                 'BRT026c-29-1','BRT026c-31-1','BRT026c-33-1','BRT026c-41-1',
                 'BRT026c-44-1','BRT033b-02-1','BRT033b-03-1','BRT033b-09-1',
                 'BRT033b-12-1','BRT033b-33-1','BRT033b-47-1','BRT033b-50-1',
                 'BRT034f-02-1','BRT034f-07-1','BRT034f-34-1','BRT036b-06-1',
                 'BRT036b-07-1','BRT036b-10-1','BRT036b-12-1','BRT036b-20-1',
                 'BRT036b-28-1','BRT036b-31-1','BRT036b-36-1','BRT036b-38-1',
                 'BRT036b-40-1','BRT036b-45-1','BRT036b-46-1','BRT037b-03-1',
                 'BRT037b-13-1','BRT037b-30-1','BRT037b-33-1','BRT037b-36-1',
                 'BRT037b-63-1','TAR010c-01-1','TAR010c-06-1','TAR010c-12-1',
                 'TAR010c-15-1','TAR010c-18-1','TAR010c-19-1','TAR010c-21-1',
                 'TAR010c-27-1','TAR010c-27-2','TAR010c-30-1','TAR010c-33-1',
                 'TAR010c-58-1']
  
        path='/auto/users/daniela/docs/AC_IC_project/307_per_file/'
        
    for cellid in cellids:
        model_per_time_wrapper(cellid, batch, basemodel=basemodel,
                           state_list=state_list)
        fig = plt.gcf()
        fig.savefig(path+cellid+'.pdf')
        plt.close(fig)
        
    state_list_beh = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
        
    for cellid in cellids:
        model_act_pas_DS(cellid, batch, basemodel=basemodel, state_list=state_list_beh)
        fig = plt.gcf()

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
#    states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
#              'PASSIVE_1_A','PASSIVE_1_B']

    basemodel = "-ref-psthfr.s_sdexp.S"

    df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

    dMI, dMI0 = hlf_analysis(df, state_list, states=states)
    
    
    fig = plt.gcf()
    
    if batch==309:
        path='/auto/users/daniela/docs/AC_IC_project/309_per_file/'
        
    if batch==307:
        path='/auto/users/daniela/docs/AC_IC_project/307_per_file/'
        
    fig.savefig(path+str(batch)+'_allcells.pdf')


