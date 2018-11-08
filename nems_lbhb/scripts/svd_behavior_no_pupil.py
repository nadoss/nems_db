#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:13:33 2018

@author: svd
"""

# scripts for showing changes in response across behabior blocks

from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
#from nems_lbhb.behavior_pupil_scripts.mod_per_state import get_model_results_per_state_model
exec(open("/auto/users/svd/python/nems_db/nems_lbhb/pupil_behavior_scripts/mod_per_state.py").read())


# fil only
state_list = ['st.fil0','st.fil']
basemodel = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20-ld-"

batch = 295  # old (Slee) IC data
d_295_IC_old = get_model_results_per_state_model(
        batch=batch, state_list=state_list,
        basemodel=basemodel, loader=loader)

batch = 305  # new (Saderi) IC data, pupil not required
d_305_IC_new = get_model_results_per_state_model(
        batch=batch, state_list=state_list,
        basemodel=basemodel, loader=loader)

d_IC_fil = pd.concat((d_295_IC_old, d_305_IC_new), axis=0)

_d = d_IC_fil[d_IC_fil["state_chan"]=="ACTIVE_1"]
_d['animal'] = _d['cellid'].apply(lambda x: x[:3])
_d0 = _d[_d["state_sig"]=="st.fil0"]
_d = _d[_d["state_sig"]=="st.fil"]
_d0 = _d0.set_index(['animal','cellid'])
_d = _d.set_index(['animal','cellid'])
_d['sig'] = (_d['r']-_d0['r']) > (_d['r_se'] + _d0['r_se'])
#_d[_d['sig']].mean(level=['animal'])
_d.median(level=['animal'])

save_path="/auto/users/svd/docs/current/conf/sfn2018/"
d_IC_fil.to_csv(save_path+"d_IC_fil.csv")

d_IC_fil=pd.read_csv(save_path+"d_IC_fil.csv", index_col=0)


# summ active vs. passive
state_list = ['st.beh0','st.beh']
basemodel = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20-ld-"

batch = 295  # old (Slee) IC data
d_295_IC_old = get_model_results_per_state_model(
        batch=batch, state_list=state_list,
        basemodel=basemodel, loader=loader)

batch = 305  # new (Saderi) IC data, pupil not required
d_305_IC_new = get_model_results_per_state_model(
        batch=batch, state_list=state_list,
        basemodel=basemodel, loader=loader)

d_IC_AP = pd.concat((d_295_IC_old, d_305_IC_new), axis=0)

_d = d_IC_AP[d_IC_AP["state_chan"]=="active"]
_d['animal'] = _d['cellid'].apply(lambda x: x[:3])
_d0 = _d[_d["state_sig"]=="st.beh0"]
_d = _d[_d["state_sig"]=="st.beh"]
_d0 = _d0.set_index(['animal','cellid'])
_d = _d.set_index(['animal','cellid'])
_d['sig'] = (_d['r']-_d0['r']) > (_d['r_se'] + _d0['r_se'])
_d[_d['sig']].mean(level=['animal'])

save_path="/auto/users/svd/docs/current/conf/sfn2018/"
d_IC_AP.to_csv(save_path+"d_IC_AP.csv")

d_IC_AP=pd.read_csv(save_path+"d_IC_AP.csv", index_col=0)



