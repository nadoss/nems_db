# Hack to get plugins loaded
import os
from nems_lbhb.rdt import plugins
os.environ['KEYWORD_PLUGINS'] = f'["{plugins.__file__}"]'

from nems import xforms
import nems_db.xform_wrappers as nw

#batch, cellid = 269, 'chn019a-a1'
#batch, cellid = 269, 'oys042c-d1'
#batch, cellid = 273, 'chn041d-b1'
#batch, cellid = 273, 'zee027b-c1'
batch, cellid = 269, 'btn144a-a1'
#modelspec = 'RDTwcg18x2-RDTfir2x15_RDTstreamgain_lvl1_dexp1'
#keywordstring = 'dlog-wc.18x1.g-fir.1x15-lvl.1'
keywordstring = 'rdtwc.18x1.g-rdtfir.1x15-rdtgain.global.NTARGETS-lvl.1'

modelname = 'rdtld-rdtshf.rep-rdtsev-rdtfmt_' + keywordstring + '_init-basic'
savefile = nw.fit_model_xforms_baphy(cellid, batch, modelname, saveInDB=False)

xf,ctx = nw.load_model_baphy_xform(cellid,batch,modelname)

"""
xfspec = [
    ('nems.xforms.init_context', {'batch': batch, 'cellid': cellid, 'keywordstring': keywordstring}),
    ('lbhb.analysis.rdt.io.load_recording', {}),
    ('lbhb.analysis.rdt.preprocessing.split_est_val', {}),
    ('lbhb.analysis.rdt.xforms.format_keywordstring', {}),
    ('nems.xforms.init_from_keywords', {}),
    ('nems.xforms.fit_basic_init', {}),
    ('nems.xforms.fit_basic', {}),
    ('nems.xforms.predict', {}),
    ('nems.xforms.add_summary_statistics', {}),
    ('nems.xforms.plot_summary', {}),
]
#    ('lbhb.analysis.rdt.xforms.fix_modelspec', {}),

context = {}
for step in xfspec:
    context = xforms.evaluate_step(step, context)
"""