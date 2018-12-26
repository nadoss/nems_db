import logging
import copy

import numpy as np

import nems.xforms as xforms
import nems.epoch
from nems.recording import load_recording
import nems.xform_helper as xhelp
import nems.modelspec as ms
from nems.utils import escaped_split
from nems_lbhb.lnp_helpers import _lnp_metric

log = logging.getLogger(__name__)


data_path = ("/auto/users/jacob/bin/anaconda3/envs/jacob_nems/"
             "nems/recordings/TAR010c-18-1.tgz")
rec = load_recording(data_path)

stim_dict = {}
stim_sig = rec['stim']
fs = stim_sig.fs
epochs = stim_sig.epochs
resp_sig = rec['resp']

for name in sorted(set(nems.epoch.epoch_names_matching(epochs, '^STIM_'))):
    r = rec.create_mask(epoch=name).apply_mask()
    stim_dict[name] = r

modelname = (
        # Split into training and validation sets (90%/10%, respectively)
        "-timesplit.f9"
        "_"  # loaders -> modules
        # Apply log transformation to the stimulus (fixed, no parameters)
        "dlog.f"
        # Spectral filter (nems.modules.weight_channels)
        "-wc.18x1.g"
        # Temporal filter (nems.modules.fir)
        "-fir.1x15"
        # Scale, currently init to 1.
        "-scl.1"
        # Level shift, usually init to mean response (nems.modules.levelshift)
        "-lvl.1"
        # Nonlinearity (nems.modules.nonlinearity -> double_exponential)
        #"-dexp.1"
        "_"  # modules -> fitters
        # Set initial values and do a rough "pre-fit"
        # Initialize fir coeffs to L2-norm of random values
        "-init.lnp"#.L2f"
        # Do the full fits
        "-lnp.t5"
        #"-nestspec"
        )

result_dict = {}
for name, stim in stim_dict.items():
    ctx = {'rec': stim}
    xfspec = xhelp.generate_xforms_spec(None, modelname, autoPlot=False)
    for i, xf in enumerate(xfspec):
        ctx = xforms.evaluate_step(xf, ctx)

    # Store tuple of ctx, error for each stim
    stim_length = ctx['val'][0]['stim'].shape[1]
    result_dict[name] = (ctx, _lnp_metric(ctx['val'][0])/stim_length)
