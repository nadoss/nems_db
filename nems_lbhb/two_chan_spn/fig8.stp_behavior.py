import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import pandas as pd

import logging
log = logging.getLogger(__name__)
log.disabled = True

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems.db as nd
import nems_db.params

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
#import nems_lbhb.xform_wrappers as nw
import nems.db as nd
import nems.plots.api as nplt
from nems.utils import find_module, ax_remove_box
from nems.metrics.stp import stp_magnitude

def stp_v_beh():

    batch1 = 274
    batch2 = 275
    modelnames=["env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
                "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
                "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
                "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
                "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
                "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
                "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic"]
    fileprefix="fig8.stp_v_beh"
    n1=modelnames[0]
    n2=modelnames[-3]

    xc_range = [-0.05, 0.6]

    df1 = nd.batch_comp(batch1,modelnames,stat='r_test').reset_index()
    df1_e = nd.batch_comp(batch1,modelnames,stat='se_test').reset_index()

    df2 = nd.batch_comp(batch2,modelnames,stat='r_test').reset_index()
    df2_e = nd.batch_comp(batch2,modelnames,stat='se_test').reset_index()

    df = df1.append(df2)
    df_e = df1_e.append(df2_e)

    cellcount = len(df)

    beta1 = df[n1]
    beta2 = df[n2]
    beta1_test = df[n1]
    beta2_test = df[n2]
    se1 = df_e[n1]
    se2 = df_e[n2]

    beta1[beta1>1]=1
    beta2[beta2>1]=1

    # test for significant improvement
    improvedcells = (beta2_test-se2 > beta1_test+se1)

    # test for signficant prediction at all
    goodcells = ((beta2_test > se2*3) | (beta1_test > se1*3))

    fh = plt.figure()
    ax = plt.subplot(2,2,1)
    stateplots.beta_comp(beta1[goodcells], beta2[goodcells],
                         n1='LN STRF', n2='STP+BEH LN STRF',
                         hist_range=xc_range, ax=ax,
                         highlight=improvedcells[goodcells])


    # LN vs. STP:
    beta1b = df[modelnames[3]]
    beta1a = df[modelnames[0]]
    beta1 = beta1b - beta1a
    se1a = df_e[modelnames[3]]
    se1b= df_e[modelnames[0]]

    b1=4
    b0=3
    beta2b = df[modelnames[b1]]
    beta2a = df[modelnames[b0]]
    beta2 = beta2b - beta2a
    se2a = df_e[modelnames[b1]]
    se2b= df_e[modelnames[b0]]

    stpgood = (beta1 > se1a+se1b)
    behgood = (beta2 > se2a+se2b)
    neither_good = np.logical_not(stpgood) & np.logical_not(behgood)
    both_good = stpgood & behgood
    stp_only_good = stpgood & np.logical_not(behgood)
    beh_only_good = np.logical_not(stpgood) & behgood

    xc_range = np.array([-0.05, 0.15])
    beta1[beta1<xc_range[0]]=xc_range[0]
    beta2[beta2<xc_range[0]]=xc_range[0]

    zz = np.zeros(2)
    ax=plt.subplot(2,2,2)
    ax.plot(xc_range,zz,'k--',linewidth=0.5)
    ax.plot(zz,xc_range,'k--',linewidth=0.5)
    ax.plot(xc_range, xc_range, 'k--', linewidth=0.5)
    l = ax.plot(beta1[neither_good], beta2[neither_good], '.', color='lightgray') +\
        ax.plot(beta1[beh_only_good], beta2[beh_only_good], '.', color='purple') +\
        ax.plot(beta1[stp_only_good], beta2[stp_only_good], '.', color='orange') +\
        ax.plot(beta1[both_good], beta2[both_good], '.', color='black')
    ax_remove_box(ax)
    ax.set_aspect('equal', 'box')
    #plt.axis('equal')
    ax.set_xlim(xc_range)
    ax.set_ylim(xc_range)
    ax.set_xlabel('delta(stp)')
    ax.set_ylabel('delta(beh)')

    olap=np.zeros(100)
    a = stpgood.values.copy()
    b = behgood.values.copy()
    for i in range(100):
        np.random.shuffle(a)
        olap[i] = np.sum(a & b)

    ll=[np.sum(neither_good), np.sum(beh_only_good),
        np.sum(stp_only_good), np.sum(both_good)]
    ax.legend(l, ll)

    ax=plt.subplot(2,2,3)
    m = np.array(df.loc[goodcells].mean()[modelnames])
    xc_range = [-0.02, 0.2]
    plt.bar(np.arange(len(modelnames)), m, color='black')
    plt.plot(np.array([-1, len(modelnames)]), np.array([0, 0]), 'k--',
             linewidth=0.5)
    plt.ylim(xc_range)
    plt.title("batch {}, n={}/{} good cells".format(
            batch, np.sum(goodcells), len(goodcells)))
    plt.ylabel('median pred corr')
    plt.xlabel('model architecture')
    ax_remove_box(ax)

    for i in range(len(modelnames)-1):

        d1 = np.array(df[modelnames[i]])
        d2 = np.array(df[modelnames[i+1]])
        s, p = ss.wilcoxon(d1, d2)
        plt.text(i+0.5, m[i+1], "{:.1e}".format(p), ha='center', fontsize=6)

    plt.xticks(np.arange(len(m)),np.round(m,3))

    return fh, df[stpgood]['cellid'].tolist()


# start main code
outpath = "/auto/users/svd/docs/current/two_band_spn/eps/"
save_fig = True
#if save_fig:
plt.close('all')

# figure 8

batch1 = 274
batch2 = 275
batch = batch1

if 1:
    # standard nMSE, tol 10e-7
    modelnames=["env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic"]
elif 0:
    # nMSE, stop at 10^-6
    modelnames=["env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic.t6",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic.t6",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic.t6",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic.t6",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t6",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t6",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t6",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t6"]
elif 0:
    # nMSE, stop at 10^-5
    modelnames=["env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic.t5",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic.t5",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic.t5",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic.t5",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t5",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t5",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t5",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic.t5"]
else:
    # shrinkge MSE
    modelnames=["env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-mt.shr-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-mt.shr-basic",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-mt.shr-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-mt.shr-basic",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-stp.1-rep2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-mt.shr-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-stp.1-rep2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-mt.shr-basic",
             "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-mt.shr-basic",
             "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-mt.shr-basic"]
mlabels= ["ind0","ind","NL0","NL","FIR0","FIR","STP0","STP"]

# prediction analysis
"""
#df = nd.batch_comp(batch,modelnames,stat='r_ceiling')
df = nd.batch_comp(batch,modelnames,stat='r_test')
df_r = nd.batch_comp(batch,modelnames,stat='r_test')
df_e = nd.batch_comp(batch,modelnames,stat='se_test')

i1=4
i2=5
n1=modelnames[i1]
n2=modelnames[i2]
cellcount = len(df)

beta1 = df[n1]
beta2 = df[n2]
se1 = df_e[n1]
se2 = df_e[n2]


# test for significant improvement
improvedcells = (beta2-se2 > beta1+se1)

# test for signficant prediction at all
goodcells = ((beta2 > se2*3) | (beta1 > se1*3))

fh1 = stateplots.beta_comp(beta1, beta2, n1=mlabels[i1], n2=mlabels[i2],
                           hist_range=[-.1, 0.6], highlight=improvedcells)

fh2 = plt.figure(figsize=(4, 4))
m = np.array(df.loc[goodcells].median()[modelnames])
plt.bar(np.arange(len(modelnames)), m, color=['lightgray','black'])
plt.plot(np.array([-1, len(modelnames)]), np.array([0, 0]), 'k--')
plt.ylim((-.05, 0.2))
plt.title("batch {}, n={}/{} good cells".format(
        batch, np.sum(goodcells), len(goodcells)))
plt.ylabel('median pred corr')
plt.xlabel('model architecture')
plt.xticks(np.arange(len(modelnames)), mlabels)
lplt.ax_remove_box()

for i in range(int(len(modelnames)/2)-1):

    d1 = np.array(df[modelnames[i*2+1]])
    d2 = np.array(df[modelnames[i*2+3]])
    s, p = ss.wilcoxon(d1, d2)
    plt.text(i*2+2, m[i*2+3]+0.03, "{:.1e}".format(p), ha='center', fontsize=6)
"""

# STP parameter anlaysis

modelname_amp = modelnames[3]  # only allow dexp parameters to change

modelname0 = modelnames[-2]
modelname = modelnames[-1]
# modelname="env100beh_dlogn2_wcc2x1_rep2_stp2_fir2x1x15_lvl2_dexp2_mrg_state01-jk"

d1 = nems_db.params.fitted_params_per_batch(batch1, modelname, stats_keys=[],
                                           multi='first')
d1_amp = nems_db.params.fitted_params_per_batch(batch1, modelname_amp,
                                               stats_keys=[], multi='first')
d2 = nems_db.params.fitted_params_per_batch(batch2, modelname, stats_keys=[],
                                           multi='first')
d2_amp = nems_db.params.fitted_params_per_batch(batch2, modelname_amp,
                                               stats_keys=[], multi='first')
#d = pd.concat((d1,d2), axis=1)
#d_amp = pd.concat((d1_amp,d2_amp), axis=1)
d=d1
d_amp=d1

fh1, stpcellid = stp_v_beh()
stpgood = np.array([i in stpcellid for i in d.columns])


u_bounds = np.array([-0.6, 2.1])
tau_bounds = np.array([-0.1, 1.5])
str_bounds = np.array([-0.15, 0.5])
#str_bounds = np.array([-0.25, 2])
#amp_bounds = np.array([-0.1, 2.0])
amp_bounds = np.array([-0.1, 2.0])

indices = list(d.index)

for ind in indices:
    if '--u' in ind:
        u_index = ind
    elif '--tau' in ind:
        tau_index = ind
    elif '--fir' in ind:
        fir_index = ind

for ind in list(d_amp.index):
    if '--amplitude' in ind:
        amp_index = ind

u = d.loc[u_index]
tau = d.loc[tau_index]
fir = d.loc[fir_index]
amp = d_amp.loc[amp_index]
r_test = d.loc['meta--r_test']
se_test = d.loc['meta--se_test']

if modelname0 is not None:
    d01 = nems_db.params.fitted_params_per_batch(batch1, modelname0, stats_keys=[], multi='first')
    d02 = nems_db.params.fitted_params_per_batch(batch2, modelname0, stats_keys=[], multi='first')
    d01 = d01[d1.columns]
    d02 = d02[d2.columns]

    #d0 = pd.concat((d01,d02), axis=1)
    d0 = d01
    r0_test = d0.loc['meta--r_test']
    se0_test = d0.loc['meta--se_test']

u_mtx = np.zeros((len(u), 2))
tau_mtx = np.zeros_like(u_mtx)
m_fir = np.zeros_like(u_mtx)
amp_mtx = np.zeros((len(u), 2))
r_test_mtx = np.zeros(len(u))
r0_test_mtx = np.zeros(len(u))
se_test_mtx = np.zeros(len(u))
se0_test_mtx = np.zeros(len(u))
str_mtx = np.zeros_like(u_mtx)

# NOTE that parameter ordering is flipped so that active==1, passive==0

i = 0
for cellid in u.index:
    match=np.argwhere(r0_test.index==cellid)[0][0]
    r_test_mtx[i] = r_test[match]
    se_test_mtx[i] = se_test[match]
    if modelname0 is not None:
        r0_test_mtx[i] = r0_test[match]
        se0_test_mtx[i] = se0_test[match]

    t_fir = fir[match]
    x = np.mean(t_fir, axis=1) / np.std(t_fir)

    # REVERSE ORDER OF PARAMETERS to (PASSIVE, ACTIVE)
    xidx=np.array([0, 1])
    m_fir[i, :] = x[xidx]
    u_mtx[i, :] = u[match][xidx]
    tau_mtx[i, :] = np.abs(tau[match][xidx])
    str_mtx[i,:] = stp_magnitude(tau_mtx[i,:], u_mtx[i,:], fs=100)[0]

    # dexp amplitude for passive, active
    amp_mtx[i, :] = np.absolute(amp[match].T[0][xidx])

    i += 1

amp_mtx_norm = amp_mtx / amp_mtx[:,[0]] # normalize by passive
str_mtx_norm = str_mtx / str_mtx[:,[0]] # normalize by passive

# EI_units = (m_fir[:,1]<0)
#good_pred = (r_test_mtx > se_test_mtx*3) | \
#            (r0_test_mtx > se0_test_mtx*3)
good_pred = (r_test_mtx > se_test_mtx*2)
mod_units = (r_test_mtx-se_test_mtx/2) >(r0_test_mtx+se0_test_mtx/2)
non_suppressed_units=((amp_mtx[:,0]/10 < amp_mtx[:,1]) &
                      (amp_mtx[:,1]/10 < amp_mtx[:,0]) &
                      (r_test_mtx > 0.08))

#show_units = good_pred & stpgood
show_units = mod_units & stpgood
#show_units = stpgood

tau_mtx[tau_mtx > tau_bounds[1]] = tau_bounds[1]
str_mtx[str_mtx < str_bounds[0]] = str_bounds[0]
str_mtx[str_mtx > str_bounds[1]] = str_bounds[1]
amp_mtx[amp_mtx > amp_bounds[1]] = amp_bounds[1]

umean = np.mean(u_mtx[show_units], axis=0)
uerr = np.std(u_mtx[show_units], axis=0) / np.sqrt(np.sum(show_units))
taumean = np.mean(tau_mtx, axis=0)
tauerr = np.std(tau_mtx, axis=0) / np.sqrt(str_mtx.shape[0])
strmean = np.mean(str_mtx[show_units], axis=0)
strerr = np.std(str_mtx[show_units], axis=0) / np.sqrt(np.sum(show_units))
str_norm_mean = np.mean(str_mtx_norm[show_units], axis=0)
str_norm_err = np.std(str_mtx_norm[show_units], axis=0) / np.sqrt(np.sum(show_units))
ampmean = np.mean(amp_mtx[show_units], axis=0)
amperr = np.std(amp_mtx[show_units], axis=0) / np.sqrt(np.sum(show_units))
amp_norm_mean = np.mean(amp_mtx_norm[show_units], axis=0)
amp_norm_err = np.std(amp_mtx_norm[show_units], axis=0) / np.sqrt(np.sum(show_units))

# see note about reversed ordering above
xstr = 'passive'
ystr = 'active'

fh2 = plt.figure(figsize=(8, 5))

dotcolor = 'black'
dotcolor_ns = 'lightgray'
thinlinecolor = 'gray'
barcolors = [(115/255, 200/255, 239/255), (235/255, 47/255, 40/255)]
barwidth = 0.5

ax = plt.subplot(2, 3, 1)
plt.plot(amp_bounds, amp_bounds, 'k--')
plt.plot(amp_mtx[~show_units, 0], amp_mtx[~show_units, 1], '.',
         color=dotcolor_ns)
plt.plot(amp_mtx[show_units, 0], amp_mtx[show_units, 1], '.', color=dotcolor)
plt.title('bat {} n={}/{} good units'.format(
        batch, np.sum(show_units), u_mtx.shape[0]))
plt.xlabel(xstr+' gain')
plt.ylabel(ystr+' gain')
plt.axis('equal')
lplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 2)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
plt.bar(np.arange(2), ampmean, color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), ampmean, yerr=amperr, color='black', linewidth=2)
plt.plot(amp_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)

w, p = ss.wilcoxon(amp_mtx_norm[show_units, 0], amp_mtx_norm[show_units, 1])
plt.ylim(amp_bounds)
plt.ylabel('STRF gain')
plt.xlabel('{} {:.3f} - {} {:.3f} - rat {:.3f} - p<{:.5f}'.format(
        xstr, ampmean[0], ystr, ampmean[1], ampmean[1]/ampmean[0], p))
lplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 3)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
plt.bar(np.arange(2), umean, color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), umean, yerr=uerr, color='black', linewidth=2)
plt.plot(u_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)
#plt.plot(np.random.normal(0, 0.05, size=u_mtx[show_units, 0].shape),
#         u_mtx[show_units, 0], '.', color=dotcolor)
#plt.plot(np.random.normal(1, 0.05, size=u_mtx[show_units, 0].shape),
#         u_mtx[show_units, 1], '.', color=dotcolor)

w, p = ss.wilcoxon(u_mtx[show_units, 0], u_mtx[show_units, 1])
plt.ylim(u_bounds)
plt.ylabel('STP u')
plt.xlabel('{} {:.3f} - {} {:.3f} - rat {:.3f} - p<{:.5f}'.format(
        xstr, umean[0], ystr, umean[1], umean[1]/umean[0], p))
lplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 4)
plt.plot(str_bounds, str_bounds, 'k--')
plt.plot(str_mtx[~show_units, 0], str_mtx[~show_units, 1], '.', color=dotcolor_ns)
plt.plot(str_mtx[show_units, 0], str_mtx[show_units, 1], '.', color=dotcolor)
plt.xlabel(xstr+' STP str')
plt.ylabel(ystr+' STP str')
plt.ylim(str_bounds)
plt.axis('equal')
lplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 5)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
plt.bar(np.arange(2), strmean, color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), strmean, yerr=strerr, color='black', linewidth=2)
plt.plot(str_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)

w, p = ss.wilcoxon(str_mtx_norm[show_units, 0], str_mtx_norm[show_units, 1])
plt.ylim(str_bounds)
plt.ylabel('STP str')
plt.xlabel('{} {:.3f} - {} {:.3f} - rat {:.3f} - p<{:.5f}'.format(
        xstr, strmean[0], ystr, strmean[1], strmean[1]/strmean[0], p))
lplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 6)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
plt.bar(np.arange(2), np.sqrt(taumean), color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), np.sqrt(taumean), yerr=np.sqrt(tauerr), color='black', linewidth=2)
plt.plot(tau_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)
w, p = ss.wilcoxon(tau_mtx[show_units, 0], tau_mtx[show_units, 1])

plt.ylim((-np.sqrt(np.abs(tau_bounds[0])), np.sqrt(tau_bounds[1])))
plt.ylabel('sqrt(STP tau)')
plt.xlabel('{} {:.3f} - {} {:.3f} - rat {:.3f} - p<{:.5f}'.format(
        xstr, taumean[0], ystr, taumean[1], taumean[1]/taumean[0], p))
lplt.ax_remove_box(ax)

plt.tight_layout()

if save_fig:
    batchstr = str(batch)
    fh1.savefig(outpath + "fig8.beh_pred_scatter_batch"+batchstr+".pdf")
    fh2.savefig(outpath + "fig8.beh_stp_parms_batch"+batchstr+"_"+modelname+".pdf")
