import matplotlib.pyplot as plt
import nems.plots.api as nplt
from nems_db.xform_wrappers import fit_model_xforms_baphy, load_model_baphy_xform
from nems.recording import Recording

cellid = 'TAR010c-18-1'
batch = 271
modelname = 'ozgf100ch18_wc18x1_fir1x15_lvl1_dexp1_fit01'

#cellid = 'BRT026c-16-1'
#batch = 301
#modelname = 'nostim20pupbeh_stategain3_fitpjk01'

# fit the model (only need this if it hasn't been done before)
#fit_model_xforms_baphy(
#        cellid='TAR010c-18-1', batch=271,
#        modelname='ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01',
#        saveInDB=True,
#        )

# load the fitted modelspec
xfspec, ctx = load_model_baphy_xform(cellid=cellid, batch=batch,
                                     modelname=modelname, eval_model=True)

# can pass 'figsize' argument to override default sizing.
# otherwise will be   width = 10 inches x width_mult
#                     height = 1 inch/plot x height_mult
# use epoch=str and occurence=int to control which epoch and occurence
# get displayed for epoch-dependent plots.
fig = nplt.quickplot(ctx, height_mult=3.5, width_mult=1.2, epoch='TRIAL',
                     occurrence=0)
plt.show()
