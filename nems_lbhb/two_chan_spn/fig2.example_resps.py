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

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
import nems_db.xform_wrappers as nw
import nems.db as nd
import nems.plots.api as nplt
from nems.utils import find_module

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 6,
          'axes.titlesize': 6,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

batch = 259

# shrinkage
modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-mt.shr-basic"
# regular
modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

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
outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev/"

if 0:
    #cellid="por077a-c1"
    cellid = "por074b-d2"
    cellid = "por020a-c1"
    fh, ctx2 = lplt.compare_model_preds(cellid, batch, modelname1, modelname2);
    #xf1, ctx1 = lplt.get_model_preds(cellid, batch, modelname1)
    #xf2, ctx2 = lplt.get_model_preds(cellid, batch, modelname2)
    #nplt.diagnostic(ctx2);

    fh.savefig(outpath + "fig1_model_preds_" + cellid + ".pdf")


elif 0:
    for cellid, c in df[m].iterrows():
        fh = lplt.compare_model_preds(cellid,batch,modelname1,modelname2);
        #fh.savefig(outpath + "fig1_model_preds_" + cellid + ".pdf")

else:
    cellcount = np.sum(m)
    colcount = 3
    rowcount = cellcount+1
    #rowcount = np.ceil((cellcount+1)/colcount)

    i = 0
    fh = plt.figure(figsize=(10,(cellcount+1)*0.8))

    for cellid, c in df[m].iterrows():
        i += 1
        if i==1:
            ax0 = plt.subplot(rowcount,colcount,1)
            ax = plt.subplot(rowcount,colcount,i*colcount+1)

            _, ctx1, ctx2 = lplt.quick_pred_comp(cellid,batch,modelname1,modelname2,
                                                 ax=(ax0,ax))
            ax0.get_xaxis().set_visible(False)
        else:
            ax = plt.subplot(rowcount,colcount,i*colcount+1)

            _, ctx1, ctx2 = lplt.quick_pred_comp(cellid,batch,modelname1,modelname2,
                                                 ax=ax);

        if i<cellcount+1:
            ax.get_xaxis().set_visible(False)

        ctx1['modelspec'][1]['plot_fns'] = ['nems.plots.api.strf_timeseries']
        ctx2['modelspec'][1]['plot_fns'] = ['nems.plots.api.weight_channels_heatmap']
        ctx2['modelspec'][2]['plot_fns'] = ['nems.plots.api.before_and_after_stp']
        ctx2['modelspec'][3]['plot_fns'] = ['nems.plots.api.strf_timeseries']
        ax = plt.subplot(rowcount,colcount*2,i*colcount*2+3)
        ctx1['modelspec'].plot(mod_index=1, ax=ax, rec=ctx1['val'])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(rowcount,colcount*2,i*colcount*2+4)
        ctx2['modelspec'].plot(mod_index=1, ax=ax, rec=ctx2['val'])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax = plt.subplot(rowcount,colcount*2,i*colcount*2+5)

        ctx2['modelspec'].plot(mod_index=2, ax=ax, rec=ctx2['val'])
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        if i==1:
            ax.legend(('1','2','3','in'))
        ax = plt.subplot(rowcount,colcount*2,i*colcount*2+6)
        ctx2['modelspec'].plot(mod_index=3, ax=ax, rec=ctx2['val'])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        if i==1:
            ax.legend(('1','2','3'))

    fh.savefig(outpath + "fig2_example_psth_preds.pdf")


