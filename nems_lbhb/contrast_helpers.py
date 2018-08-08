import logging
import copy

import numpy as np
from scipy.signal import convolve2d

import nems.modelspec as ms
from nems.utils import find_module
from nems import signal
from nems.modules.nonlinearity import _logistic_sigmoid, _double_exponential
from nems.initializers import prefit_to_target, prefit_mod_subset
from nems.analysis.api import fit_basic
import nems.fitters.api
import nems.metrics.api
from nems import priors

log = logging.getLogger(__name__)


def _strf_to_contrast(modelspec):
    '''
    Copy prefitted WC and FIR phi values to contrast-based counterparts.
    '''
    modelspec = copy.deepcopy(modelspec)
    wc_idx, ctwc_idx = find_module('weight_channels', modelspec,
                                   find_all_matches=True)
    fir_idx, ctfir_idx = find_module('fir', modelspec, find_all_matches=True)

    log.info("Updating contrast phi to match prefitted strf ...")

    modelspec[ctwc_idx]['phi'] = copy.deepcopy(modelspec[wc_idx]['phi'])
    modelspec[ctfir_idx]['phi'] = copy.deepcopy(modelspec[fir_idx]['phi'])

    return modelspec


def strf_to_contrast(modelspecs, IsReload=False, **context):
    if not IsReload:
        new_mspec = _strf_to_contrast(modelspecs[0])
        return {'modelspecs': [new_mspec]}
    else:
        return {'modelspecs': modelspecs}


def make_contrast_signal(rec, name='contrast', source_name='stim', ms=500,
                         bins=None, percentile=50, normalize=False):
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

    if normalize:
        contrast /= np.max(np.abs(contrast), axis=0)
        rectified = contrast
    else:
        cutoff = np.nanpercentile(contrast, percentile)
        rectified = np.where(contrast >= cutoff, 1, 0)

    contrast_sig = source_signal._modified_copy(rectified)
    rec[name] = contrast_sig

    return rec


def add_contrast(rec, name='contrast', source_name='stim', percentile=50,
                 ms=500, bins=None, normalize=False, IsReload=False, **context):
    '''xforms wrapper for make_contrast_signal'''
    rec_with_contrast = make_contrast_signal(
            rec, name=name, source_name=source_name, ms=ms, bins=bins,
            percentile=percentile, normalize=normalize,
            )
    return {'rec': rec_with_contrast}


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


def dynamic_sigmoid(rec, i, o, c, base, amplitude, shift, kappa,
                    base_mod=0, amplitude_mod=0, shift_mod=0,
                    kappa_mod=0, eq='logsig'):

    # TODO: Really this could be used with any signal, doesn't have to be
    #       a contrast signal. So rename maybe?
    contrast = rec[c].as_continuous()

    if np.isnan(base_mod):
        b = base
    else:
        b = base+base_mod*contrast

    if np.isnan(amplitude_mod):
        a = amplitude
    else:
        a = amplitude+amplitude_mod*contrast

    if np.isnan(shift_mod):
        s = shift
    else:
        s = shift+shift_mod*contrast

    if np.isnan(kappa_mod):
        k = kappa
    else:
        k = kappa+kappa_mod*contrast

    if eq.lower() in ['logsig', 'logistic_sigmoid', 'l']:
        fn = lambda x: _logistic_sigmoid(x, b, a, s, k)
    elif eq.lower() == ['dexp', 'double_exponential', 'd']:
        fn = lambda x: _double_exponential(x, b, a, s, k)
    else:
        # Not a recognized equation, do logistic_sigmoid by default.
        fn = lambda x: _logistic_sigmoid(x, b, a, s, k)

    return [rec[i].transform(fn, o)]


def init_dsig(rec, modelspec):
    '''
    Initialization of priors for logistic_sigmoid,
    based on process described in methods of Rabinowitz et al. 2014.
    '''
    # Shouldn't need to do this since calling function already copies
    # modelspec = copy.deepcopy(modelspec)

    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        log.warning("No dsig module was found, can't initialize.")
        return modelspec

    modelspec = copy.deepcopy(modelspec)

    if modelspec[dsig_idx]['fn_kwargs'].get('eq', '') in \
            ['dexp', 'd', 'double_exponential']:
        return _init_double_exponential(rec, modelspec, dsig_idx)
    else:
        return _init_logistic_sigmoid(rec, modelspec, dsig_idx)


def _init_logistic_sigmoid(rec, modelspec, dsig_idx):
    pred = rec['pred'].as_continuous()
    resp = rec['resp'].as_continuous()

    mean_pred = np.nanmean(pred)
    min_pred = np.nanmean(pred)-np.nanstd(pred)*3
    max_pred = np.nanmean(pred)+np.nanstd(pred)*3
    pred_range = max_pred - min_pred
    min_resp = max(np.nanmean(resp)-np.nanstd(resp)*3, 0)  # must be >= 0

    max_resp = np.nanmean(resp)+np.nanstd(resp)*3
    resp_range = max_resp - min_resp

    # Rather than setting a hard value for initial phi,
    # set the prior distributions and let the fitter/analysis
    # decide how to use it.
    base0 = min_resp + 0.05*(resp_range)
    amplitude0 = resp_range
    shift0 = mean_pred
    kappa0 = pred_range
    log.info("Initial   base,amplitude,shift,kappa=({}, {}, {}, {})"
             .format(base0, amplitude0, shift0, kappa0))

    base = ('Exponential', {'beta': base0})
    amplitude = ('Exponential', {'beta': amplitude0})
    shift = ('Normal', {'mean': shift0, 'sd': pred_range})
    kappa = ('Exponential', {'beta': kappa0})

    modelspec[dsig_idx]['prior'] = {
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa, 'base_mod': base,
            'amplitude_mod': amplitude, 'shift_mod': shift,
            'kappa_mod': kappa,
            }

    modelspec[dsig_idx]['bounds'] = {
            'base': (1e-15, None),
            'amplitude': (1e-15, None),
            'shift': (None, None),
            'kappa': (1e-15, None),
            }

    return modelspec


def _init_double_exponential(rec, modelspec, target_i):

    if target_i == len(modelspec):
        fit_portion = modelspec
    else:
        fit_portion = modelspec[:target_i]

    # generate prediction from modules preceeding dsig

    # HACK
    for i, m in enumerate(fit_portion):
        if not m.get('phi', None):
            m = priors.set_mean_phi([m])[0]
            modelspec[i] = m

    ms.fit_mode_on(fit_portion)
    rec = ms.evaluate(rec, fit_portion)
    ms.fit_mode_off(fit_portion)

    in_signal = modelspec[target_i]['fn_kwargs']['i']
    pchans = rec[in_signal].shape[0]
    amp = np.zeros([pchans, 1])
    base = np.zeros([pchans, 1])
    kappa = np.zeros([pchans, 1])
    shift = np.zeros([pchans, 1])

    for i in range(pchans):
        resp = rec['resp'].as_continuous()
        pred = rec[in_signal].as_continuous()[i:(i+1), :]
        if resp.shape[0] == pchans:
            resp = resp[i:(i+1), :]

        keepidx = np.isfinite(resp) * np.isfinite(pred)
        resp = resp[keepidx]
        pred = pred[keepidx]

        # choose phi s.t. dexp starts as almost a straight line
        # phi=[max_out min_out slope mean_in]
        # meanr = np.nanmean(resp)
        stdr = np.nanstd(resp)

        # base = np.max(np.array([meanr - stdr * 4, 0]))
        base[i, 0] = np.min(resp)
        # base = meanr - stdr * 3

        # amp = np.max(resp) - np.min(resp)
        amp[i, 0] = stdr * 3

        shift[i, 0] = np.mean(pred)
        # shift = (np.max(pred) + np.min(pred)) / 2

        predrange = 2 / (np.max(pred) - np.min(pred) + 1)
        kappa[i, 0] = np.log(predrange)

    modelspec[target_i]['prior'] = {
            'base': base, 'amplitude': amp, 'shift': shift,
            'kappa': kappa, 'base_mod': base,
            'amplitude_mod': amp, 'shift_mod': shift,
            'kappa_mod': kappa,
            }

    return modelspec


def dsig_phi_to_prior(modelspec):
    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    dsig = modelspec[dsig_idx]

    phi = dsig['phi']
    b = phi['base']
    a = phi['amplitude']
    k = phi['kappa']
    s = phi['shift']

    p = dsig['prior']
    p['base'][1]['beta'] = b
    p['amplitude'][1]['beta'] = a
    p['shift'][1]['mean'] = s  # Do anything to scale std?
    p['kappa'][1]['beta'] = k

    return modelspec


def init_contrast_model(est, modelspecs, IsReload=False,
                        tolerance=10**-5.5, max_iter=1000,
                        fitter='scipy_minimize', metric='nmse', **context):
    if IsReload:
        return {}

    modelspec = copy.deepcopy(modelspecs[0])
    if not find_module('dynamic_sigmoid', modelspec):
        new_ms = nems.initializers.prefit_LN(est, modelspec, tolerance=tolerance,
                                             max_iter=max_iter)
        return {'modelspecs': [new_ms]}

    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}
    fitter_fn = getattr(nems.fitters.api, fitter)
    metric_fn = lambda d: getattr(nems.metrics.api, metric)(d, 'pred', 'resp')

    # fit without STP module first (if there is one)
    modelspec = prefit_to_target(est, modelspec, fit_basic,
                                 target_module='levelshift',
                                 extra_exclude=['stp'],
                                 fitter=fitter_fn,
                                 metric=metric_fn,
                                 fit_kwargs=fit_kwargs)

    # then initialize the STP module (if there is one)
    for i, m in enumerate(modelspec):
        if 'stp' in m['fn']:
            m = priors.set_mean_phi([m])[0]  # Init phi for module
            modelspec[i] = m
            break

    log.info("initializing priors and bounds for dsig ...\n")
    modelspec = init_dsig(est, modelspec)
    modelspec = prefit_mod_subset(
            est, modelspec, fit_basic,
            fit_set=['dynamic_sigmoid'],
            fitter=fitter_fn,
            metric=metric_fn,
            fit_kwargs=fit_kwargs)

    # after prefitting contrast modules, update priors to reflect the
    # prefit values so that random sample fits incorporate the prefit info.
    modelspec = dsig_phi_to_prior(modelspec)

    return {'modelspecs': [modelspec]}
