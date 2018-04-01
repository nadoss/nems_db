import matplotlib.pyplot as plt
import nems.plots.api as nplt
from nems_db.xform_wrappers import fit_model_xforms_baphy, load_model_baphy_xform
from nems.recording import Recording

# Put this in nems_db instead of nems since it relies on baphy wrapper

# fit the model (only need this if it hasn't been done before)
#fit_model_xforms_baphy(
#        cellid='TAR010c-18-1', batch=271,
#        modelname='ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01',
#        saveInDB=True,
#        )

# load the fitted modelspec
xfspec, ctx = load_model_baphy_xform(
                    cellid='TAR010c-18-1', batch=271,
                    modelname='ozgf100ch18_wc18x1_fir1x15_lvl1_dexp1_fit01',
                    eval_model=True
                    )

fig = nplt.quickplot(ctx)
plt.show()
