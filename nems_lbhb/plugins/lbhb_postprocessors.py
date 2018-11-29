import logging
import re

log = logging.getLogger(__name__)

def ebc(loadkey):
    """
    ebc = evaluate_by_condition
       finds prediction correlations by condition
    """
    ops = loadkey.split('.')[1:]
    if ops[0] == 'rmM':
        use_mask=False
    else:
        use_mask=True
    xfspec = [
        ['nems.xforms.add_summary_statistics', {'use_mask': use_mask}],
        ['nems_lbhb.postprocessing.add_summary_statistics_by_condition',{}]
    ]
    return xfspec

def SPOpf(loadkey):
    xfspec = [['nems.xforms.predict', {}]]
    xfspec = xfspec + ebc('ebc.rmM')
    xfspec.append(['nems.xforms.plot_summary', {}])
    xfspec.append(['nems_lbhb.SPO_helpers.plot_all_vals_',{}])
    return xfspec
