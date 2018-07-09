import os
import sys
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.disabled = True

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems_db.db as nd
import nems_db.params
import numpy as np
import scipy.stats as ss

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
import nems_db.xform_wrappers as nw
import nems_db.db as nd
import nems.plots.api as nplt
from nems.utils import find_module

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)


def stp_parameter_comp(batch, modelname, modelname0=None):

    d = nems_db.params.fitted_params_per_batch(batch, modelname,
                                               stats_keys=[], multi='first')

    u_bounds = np.array([-0.6, 2.1])
    tau_bounds = np.array([-0.1, 1.5])
    str_bounds = np.array([-0.25, 0.55])

    indices = list(d.index)

    for ind in indices:
        if '--u' in ind:
            u_index = ind
        elif '--tau' in ind:
            tau_index = ind
        elif '--fir' in ind:
            fir_index = ind

    u = d.loc[u_index]
    tau = d.loc[tau_index]
    fir = d.loc[fir_index]
    r_test = d.loc['meta--r_test']
    se_test = d.loc['meta--se_test']

    if modelname0 is not None:
        d0 = nems_db.params.fitted_params_per_batch(batch, modelname0, stats_keys=[], multi='first')
        r0_test = d0.loc['meta--r_test']
        se0_test = d0.loc['meta--se_test']

    u_mtx = np.zeros((len(u), 2))
    tau_mtx = np.zeros_like(u_mtx)
    m_fir = np.zeros_like(u_mtx)
    r_test_mtx = np.zeros(len(u))
    r0_test_mtx = np.zeros(len(u))
    se_test_mtx = np.zeros(len(u))
    se0_test_mtx = np.zeros(len(u))
    str_mtx = np.zeros_like(u_mtx)

    i = 0
    for cellid in u.index:
        r_test_mtx[i] = r_test[cellid]
        se_test_mtx[i] = se_test[cellid]
        if modelname0 is not None:
            r0_test_mtx[i] = r0_test[cellid]
            se0_test_mtx[i] = se0_test[cellid]

        t_fir = fir[cellid]
        x = np.mean(t_fir, axis=1) / np.std(t_fir)
        if x[0] > x[1]:
            xidx = np.array([0, 1])
        else:
            xidx = np.array([1, 0])
        m_fir[i, :] = x[xidx]
        u_mtx[i, :] = u[cellid][xidx]
        tau_mtx[i, :] = np.abs(tau[cellid][xidx])
        str_mtx[i,:] = nplt.stp_magnitude(tau_mtx[i,:], u_mtx[i,:], fs=100)[0]
        i += 1

    # EI_units = (m_fir[:,0]>0) & (m_fir[:,1]<0)
    EI_units = (m_fir[:,1]<0)
    good_pred = (r_test_mtx > 0.08)
    mod_units = (r_test_mtx-se_test_mtx) >(r0_test_mtx+se0_test_mtx)

    show_units = mod_units

    tau_mtx[tau_mtx > tau_bounds[1]] = tau_bounds[1]
    str_mtx[str_mtx < str_bounds[0]] = str_bounds[0]
    str_mtx[str_mtx > str_bounds[1]] = str_bounds[1]

    umean = np.median(u_mtx[show_units], axis=0)
    uerr = np.std(u_mtx[show_units], axis=0) / np.sqrt(np.sum(show_units))
    taumean = np.median(tau_mtx, axis=0)
    tauerr = np.std(tau_mtx, axis=0) / np.sqrt(str_mtx.shape[0])
    strmean = np.median(str_mtx, axis=0)
    strerr = np.std(str_mtx, axis=0) / np.sqrt(str_mtx.shape[0])

    fh = plt.figure(figsize=(8,5))

    dotcolor = 'black'
    dotcolor_ns = 'lightgray'
    barcolors = [(235/255, 47/255, 40/255), (115/255, 200/255, 239/255)]
    barwidth = 0.5

    ax = plt.subplot(2, 3, 1)
    plt.plot(np.array([-1, 1]), np.array([-1, 1]), 'k--')
    plt.plot(m_fir[good_pred, 0], m_fir[good_pred, 1], '.', color=dotcolor)
    plt.title('n={}/{} good units'.format(
            np.sum(show_units), u_mtx.shape[0]))
    plt.xlabel('exc channel gain')
    plt.ylabel('inh channel gain')
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 3, 2)
    plt.plot(u_bounds, u_bounds, 'k--')
    plt.plot(u_mtx[~show_units, 0], u_mtx[~show_units, 1], '.', color=dotcolor_ns)
    plt.plot(u_mtx[show_units, 0], u_mtx[show_units, 1], '.', color=dotcolor)
    plt.axis('equal')
    plt.xlabel('exc channel u')
    plt.ylabel('inh channel u')
    plt.ylim(u_bounds)
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 3, 3)
    plt.plot(str_bounds, str_bounds, 'k--')
    plt.plot(str_mtx[~show_units, 0], str_mtx[~show_units, 1], '.', color=dotcolor_ns)
    plt.plot(str_mtx[show_units, 0], str_mtx[show_units, 1], '.', color=dotcolor)
    plt.axis('equal')
    plt.xlabel('exc channel str')
    plt.ylabel('inh channel str')
    plt.ylim(str_bounds)
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 3, 4)
    plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
    plt.bar(np.arange(2), umean, color=barcolors, width=barwidth)
    plt.plot(np.random.normal(0, 0.05, size=u_mtx[show_units, 0].shape),
             u_mtx[show_units, 0], '.', color=dotcolor)
    plt.plot(np.random.normal(1, 0.05, size=u_mtx[show_units, 0].shape),
             u_mtx[show_units, 1], '.', color=dotcolor)
    # plt.errorbar(np.arange(2), umean, yerr=uerr, color='black', linewidth=2)
    w, p = ss.wilcoxon(u_mtx[show_units, 0], u_mtx[show_units, 1])

    plt.ylim(u_bounds)
    plt.ylabel('u')
    plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p<{:.5f}'.format(
            umean[0], umean[1], umean[1]/umean[0], p))
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 3, 5)
    plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
    plt.bar(np.arange(2), np.sqrt(taumean), color=barcolors, width=barwidth)
    plt.plot(np.random.normal(0, 0.05, size=tau_mtx[show_units, 0].shape),
             np.sqrt(tau_mtx[show_units, 0]), '.', color=dotcolor)
    plt.plot(np.random.normal(1, 0.05, size=tau_mtx[show_units, 0].shape),
             np.sqrt(tau_mtx[show_units, 1]), '.', color=dotcolor)
    # plt.errorbar(np.arange(2), taumean, yerr=tauerr, color='black', linewidth=2)
    w, p = ss.wilcoxon(tau_mtx[show_units, 0], tau_mtx[show_units, 1])

    plt.ylim((-np.sqrt(np.abs(tau_bounds[0])), np.sqrt(tau_bounds[1])))
    plt.ylabel('sqrt(tau)')
    plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p<{:.5f}'.format(
            taumean[0], taumean[1], taumean[1]/taumean[0], p))
    lplt.ax_remove_box(ax)

    ax = plt.subplot(2, 3, 6)
    plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
    plt.bar(np.arange(2), strmean, color=barcolors, width=barwidth)
    plt.plot(np.random.normal(0, 0.05, size=str_mtx[show_units, 0].shape),
             str_mtx[show_units, 0], '.', color=dotcolor)
    plt.plot(np.random.normal(1, 0.05, size=str_mtx[show_units, 0].shape),
             str_mtx[show_units, 1], '.', color=dotcolor)
    w, p = ss.wilcoxon(str_mtx[show_units, 0], str_mtx[show_units, 1])

    plt.ylim(str_bounds)
    plt.ylabel('STP str')
    plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p<{:.5f}'.format(
            strmean[0], strmean[1], strmean[1]/strmean[0], p))
    lplt.ax_remove_box(ax)

    plt.tight_layout()

    return fh


# start main code

# figure 6
batch = 259
#modelname="env100_dlog_stp2_fir2x15_lvl1_dexp1_basic"

# shrinkage, normed wc
modelname0 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic"
modelname = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-mt.shr-basic"

# no shrinkage, wc
# modelname0 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
# modelname = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

# no shrinkage, wc normed
# modelname0 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
# modelname = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

fh = stp_parameter_comp(batch, modelname, modelname0=modelname0)

fh.savefig(outpath + "fig6.stp_parms_"+modelname+".pdf")


"""
batch = 259
modelname1 = "env100_dlogf_fir2x15_lvl1_dexp1_basic"
# modelname1 = "env100_dlog_fir2x15_lvl1_dexp1_basic-shr"
#modelname2="env100_dlog_stp2_fir2x15_lvl1_dexp1_basic"
modelname2 = "env100_dlogf_wcc2x3_stp3_fir3x15_lvl1_dexp1_basic"
# modelname2="env100_dlog_wcc2x3_stp3_fir3x15_lvl1_dexp1_basic-shr"
#modelname2="env100_dlog_wcc2x2_stp2_fir2x15_lvl1_dexp1_basic"

modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"


modelnames = [modelname1, modelname2]
df = nd.batch_comp(batch, modelnames)
df['diff'] = df[modelname2] - df[modelname1]
df['cellid'] = df.index
df.sort_values('cellid', inplace=True, ascending=True)
m = df['cellid'].str.startswith('por07') & (df[modelname2] > 0.3)
for index, c in df[m].iterrows():
    print("{}  {:.3f} - {:.3f} = {:.3f}".format(
            index, c[modelname2], c[modelname1], c['diff']))

plt.close('all')
outpath = "/auto/users/svd/docs/current/two_band_spn/eps/"

if 0:
    #cellid="por077a-c1"
    cellid = "por074b-d2"
    fh = lplt.compare_model_preds(cellid, batch, modelname1, modelname2);
    xf1, ctx1 = lplt.get_model_preds(cellid, batch, modelname1)
    xf2, ctx2 = lplt.get_model_preds(cellid, batch, modelname2)
    nplt.diagnostic(ctx2);
    # fh.savefig(outpath + "fig1_model_preds_" + cellid + ".pdf")


elif 0:
    for cellid, c in df[m].iterrows():
        fh = lplt.compare_model_preds(cellid,batch,modelname1,modelname2);
        #fh.savefig(outpath + "fig1_model_preds_" + cellid + ".pdf")

else:
    fh = plt.figure(figsize=(8,10))
    cellcount = np.sum(m)
    colcount = 2
    rowcount = np.ceil((cellcount+1)/colcount)

    i=0
    for cellid, c in df[m].iterrows():
        i += 1
        if i==1:
            ax0 = plt.subplot(rowcount,colcount,i)
            i += 1
            ax = plt.subplot(rowcount,colcount,i)

            lplt.quick_pred_comp(cellid,batch,modelname1,modelname2,
                                 ax=(ax0,ax))
            ax.get_xaxis().set_visible(False)
        else:
            ax = plt.subplot(rowcount,colcount,i)

            lplt.quick_pred_comp(cellid,batch,modelname1,modelname2,
                                 ax=ax);

        if i<cellcount+1:
            ax.get_xaxis().set_visible(False)

    fh.savefig(outpath + "fig2_example_psth_preds.pdf")
"""