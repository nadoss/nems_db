"""
initializer keywords specific to LBHB models
should occur intermingled with fitter keywords
"""
import logging
import re

from nems.plugins.default_initializers import init as nems_init
from nems_lbhb.lnp_helpers import _lnp_metric

log = logging.getLogger(__name__)


def init(kw):
    '''
    Same as default nems init except adds 'c' option for contrast model.
    '''
    xfspec = nems_init(kw)
    ops = kw.split('.')[1:]
    if 'c' in ops:
        xfspec[0][0] = 'nems_lbhb.contrast_helpers.init_contrast_model'
        if 'strfc' in ops:
            xfspec[0][1]['copy_strf'] = True
    elif 'lnp' in ops:
        xfspec[0][1]['metric'] = lambda data: _lnp_metric(data, 'pred', 'resp')
        #xfspec[0][1]['metric'] = 'likelihood_poisson'

    return xfspec
