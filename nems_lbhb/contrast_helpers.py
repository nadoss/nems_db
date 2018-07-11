import logging
import copy

import numpy as np
from scipy.signal import convolve2d

from nems.utils import find_module
from nems import signal

log = logging.getLogger(__name__)


def static_to_dynamic(modelspec):
    '''
    Changes bounds on contrast model to allow for dynamic modulation
    of the logistic sigmoid output nonlinearity.
    '''
    modelspec = copy.deepcopy(modelspec)
    logsig_idx = find_module('logistic_sigmoid', modelspec)
    wc_idx, ctwc_idx = find_module('weight_channels', modelspec,
                                   find_all_matches=True)
    fir_idx, ctfir_idx = find_module('fir', modelspec, find_all_matches=True)

    modelspec[logsig_idx]['bounds'].update({
            'base_mod': (None, None), 'amplitude_mod': (None, None),
            'shift_mod': (None, None), 'kappa_mod': (None, None)
            })

    # TODO: Do this or not? Doesn't look like it was done in paper,
    #       but makes sense. So maybe save til later. Also, absolute value?
#    modelspec[ctwc_idx]['phi'] = copy.deepcopy(modelspec[wc_idx]['phi'])
#    modelspec[ctfir_idx]['phi'] = copy.deepcopy(modelspec[fir_idx]['phi'])

    return modelspec


def dynamic_logsig(modelspecs, IsReload=False, **context):
    if not IsReload:
        dynamic_mspec = static_to_dynamic(modelspecs[0])
        return {'modelspecs': [dynamic_mspec]}
    else:
        return {'modelspecs': modelspecs}


def make_contrast_signal(rec, name='contrast', source_name='stim', ms=500,
                         bins=None):
    '''
    Creates a new signal whose values represent the degree of variability
    in each channel of the source signal. Each value is based on the
    previous values within a range specified by either <ms> or <bins>.
    Only supports RasterizedSignal.
    '''

    rec = rec.copy()

    source_signal = rec[source_name]
    if not isinstance(source_signal, signal.RasterizedSignal):
        try:
            source_signal = source_signal.rasterize()
        except AttributeError:
            raise TypeError("signal with key {} was not a RasterizedSignal"
                            " and could not be converted to one."
                            .format(source_name))

    array = source_signal.as_continuous().copy()

    if ms:
        history = int((ms/1000)*source_signal.fs)
    elif bins:
        history = int(bins)
    else:
        raise ValueError("Either ms or bins parameter must be specified "
                         "and nonzero.")
    # TODO: Alternatively, base history length on some feature of signal?
    #       Like average length of some epoch ex 'TRIAL'

    array[np.isnan(array)] = 0
    filt = np.concatenate((np.zeros([1, history+1]),
                           np.ones([1, history])), axis=1)
    contrast = convolve2d(array, filt, mode='same')

    contrast_sig = source_signal._modified_copy(contrast)
    rec[name] = contrast_sig

    return rec


def add_contrast(rec, name='contrast', source_name='stim',
                 ms=500, bins=None, IsReload=False, **context):
    '''xforms wrapper for make_contrast_signal'''
    if not IsReload:
        rec_with_contrast = make_contrast_signal(
                rec, name=name, source_name=source_name, ms=ms, bins=bins
                )
        return {'rec': rec_with_contrast}
    else:
        return {'rec': rec}


def reset_single_recording(rec, est, val, IsReload=False, **context):
    '''
    Forces rec, est, and val to be a recording instead of a singleton
    list after a fit.

    Warning: This may mess up jackknifing!
    '''
    if not IsReload:
        if isinstance(rec, list):
            rec = rec[0]
        if isinstance(est, list):
            est = est[0]
        if isinstance(val, list):
            val = val[0]
    return {'rec': rec, 'est': est, 'val': val}
