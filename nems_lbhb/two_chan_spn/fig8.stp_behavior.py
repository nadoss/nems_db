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

# figure 8
batch = 274
#batch = 275

modelnames=[["env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
      "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
      "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-stp.1-rep2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
      "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c.n-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic"],
     ["env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
      "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
      "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-stp.1-rep2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
      "env.fs100-ld-st.beh0-ref_dlog.f-wc.2x1.c.n-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic"]]

# prediction analysis



# STP parameter anlaysis
# also compare amplitude!   d.loc['6--dexp2--amplitude']

batch = 274
#batch = 275

modelname0 = None
modelname=modelnames[0][-1]

fh2 = stp_parameter_comp(batch, modelname, modelname0=modelname0)

fh2.savefig(outpath + "fig8.stp_beh_parms_"+modelname+".pdf")


