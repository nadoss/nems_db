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

batch = 259
modelname1 = "env100_dlogf_fir2x15_lvl1_dexp1_basic"
# modelname1 = "env100_dlog_fir2x15_lvl1_dexp1_basic-shr"
#modelname2="env100_dlog_stp2_fir2x15_lvl1_dexp1_basic"
modelname2 = "env100_dlogf_wcc2x3_stp3_fir3x15_lvl1_dexp1_basic"
# modelname2="env100_dlog_wcc2x3_stp3_fir3x15_lvl1_dexp1_basic-shr"
#modelname2="env100_dlog_wcc2x2_stp2_fir2x15_lvl1_dexp1_basic"

modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-mt.shr-basic"


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


