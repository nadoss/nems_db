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
import pandas as pd
import scipy.stats as ss

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.modelspec as ms
import nems.xforms as xforms
import nems_db.xform_wrappers as nw
import nems_db.db as nd
import nems.plots.api as nplt
from nems.utils import find_module

outpath = "/auto/users/svd/docs/current/two_band_spn/eps/"

batch = 259
#modelnames=["env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-stp.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-wc.2x3.c-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"]
modelnames=["env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic",
            "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-stp.1-dexp.1_init-mt.shr-basic",
            "env.fs100-ld-sev_dlog.f-stp.2-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic",
            "env.fs100-ld-sev_dlog.f-wc.2x2.c.n-stp.2-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic",
            "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-mt.shr-basic"]

df = nd.batch_comp(batch,modelnames,stat='r_ceiling')
df_r = nd.batch_comp(batch,modelnames,stat='r_test')
df_e = nd.batch_comp(batch,modelnames,stat='se_test')

n1=modelnames[0]
n2=modelnames[-1]
cellcount = len(df)

beta1 = df[n1]
beta2 = df[n2]
se1 = df_e[n1]
se2 = df_e[n2]

# test for significant improvement
improvedcells = (beta2-se2 > beta1+se1)

# test for signficant prediction at all
goodcells = ((beta2 > se2*3) | (beta1 > se1*3))

fh1 = stateplots.beta_comp(beta1, beta2, n1='LN STRF', n2='RW STP STRF',
                           hist_range=[-.1, 1.1], highlight=improvedcells)

fh2 = plt.figure(figsize=(3, 3))
m = np.array(df.loc[goodcells].median()[modelnames])
plt.bar(np.arange(len(modelnames)), m,
        color='black')
plt.plot(np.array([-1, len(modelnames)]), np.array([0, 0]), 'k--')
plt.ylim((-.05, 0.8))
plt.title("batch {}, n={}/{} good cells".format(
        batch, np.sum(goodcells), len(goodcells)))
plt.ylabel('median pred corr')
plt.xlabel('model architecture')
lplt.ax_remove_box()

for i in range(len(modelnames)-1):

    d1 = np.array(df[modelnames[i]])
    d2 = np.array(df[modelnames[i+1]])
    s, p = ss.wilcoxon(d1, d2)
    plt.text(i+0.5, m[i+1]+0.03, "{:.1e}".format(p), ha='center', fontsize=6)

batchstr = str(batch)
fh1.savefig(outpath + "fig5.pred_scatter_batch"+batchstr+".pdf")
fh2.savefig(outpath + "fig5.pred_sum_bar_batch"+batchstr+".pdf")
