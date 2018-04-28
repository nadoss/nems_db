import nems_db.plot_helpers as dbp
import nems_db.db as db

# Accepted plot types:
# 'Scatter', 'Bar', 'Significance', 'Tabular', 'Pareto'
# Scatter, Bar and Pareto use Bokeh
# Significance uses matplotlib
# Tabular is just a dataframe
# The actual plot object (ex: a matplotlib figure) will be stored in
# <plot_object>.plot after the generate_plot() method is invoked.
# It will be displayed automatically if display is True.

# Save will only work for Significance plot,
# Bokeh plots have to be saved or exported manually.
save = True
savepath = '/auto/users/jacob/testfig.eps'

# Relevant options for plot utilities
batch = 271
models = [
        'ozgf100ch18_wcg18x2_fir2x15_lvl1_dexp1_basic',
        'ozgf100ch18_wcg18x2_fir2x15_lvl1_dexp1_basic-cd',
        'ozgf100ch18_dlog_wcg18x2_fir2x15_lvl1_dexp1_basic',
        'ozgf100ch18_dlog_wcg18x2_fir2x15_lvl1_dexp1_basic-cd',
        'ozgf100ch18_dlog_wcg18x2_fir2x15_lvl1_dexp1_basic-shr',
        'ozgf100ch18_dlog_wcg18x2_fir2x15_lvl1_dexp1_basic-cd-shr',
        ]
measure = 'r_test'
plot_type = 'Significance'
only_fair = True
include_outliers = False
extra_cols = []
display = True
snr = 0.0
iso = 0.7
snr_idx = 0.0


# Plot functions broken down
#cells = db.get_batch_cells(batch)['cellid'].tolist()
#cells = dbp.get_filtered_cells(cells, snr, iso, snr_idx)
#significance = dbp.get_plot(cells, models, measure, plot_type, only_fair,
#                            include_outliers, display)
#significance.generate_plot()


# Same process as all-in-one wrapper
plot = dbp.plot_filtered_batch(batch, models, measure, plot_type, only_fair,
                               include_outliers, display, extra_cols,
                               snr, iso, snr_idx)

if save and (plot_type == 'Significance'):
    plot.plot.savefig(savepath)

# Bokeh plots (Scatter, Bar & Pareto) have no self-contained method for
# displaying, like matplotlib figures do, so if you want to display them
# after-the-fact you will have to use bokeh.plotting.show like:
#
# from bokeh.plotting import show
# show(plot.plot)
