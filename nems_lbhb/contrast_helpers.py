import logging
import copy

import numpy as np
from scipy.signal import convolve2d

import nems.modelspec as ms
from nems.utils import find_module
from nems import signal
from nems.modules.nonlinearity import (_logistic_sigmoid, _double_exponential,
                                       _dlog)
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
                         dlog=False, bins=None, continuous=False,
                         normalize=False, percentile=50, ignore_zeros=True):
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

    if dlog:
        log.info("Applying dlog transformation to stimulus prior to "
                 "contrast calculation.")
        fn = lambda x: _dlog(x, -1)
        source_signal = source_signal.transform(fn)

    array = source_signal.as_continuous().copy()

    if ms is not None:
        history = int((ms/1000)*source_signal.fs)
    elif bins is not None:
        history = int(bins)
    else:
        raise ValueError("Either ms or bins parameter must be specified.")
    # TODO: Alternatively, base history length on some feature of signal?
    #       Like average length of some epoch ex 'TRIAL'

    array[np.isnan(array)] = 0
    filt = np.concatenate((np.zeros([1, max(2, history+1)]),
                           np.ones([1, max(1, history)])), axis=1)
    contrast = convolve2d(array, filt, mode='same')

    if continuous:
        if normalize:
            # Map raw values to range 0 - 1
            contrast /= np.max(np.abs(contrast), axis=0)
        rectified = contrast

    else:
        # Binary high/low contrast based on percentile cutoff.
        # 50th percentile by default.
        if ignore_zeros:
            # When calculating cutoff, ignore time bins where signal is 0
            # for all spectral channels (i.e. no stimulus present)
            no_zeros = contrast[:, ~np.all(contrast == 0, axis=0)]
            cutoff = np.nanpercentile(no_zeros, percentile)
        else:
            cutoff = np.nanpercentile(contrast, percentile)
        rectified = np.where(contrast >= cutoff, 1, 0)

    contrast_sig = source_signal._modified_copy(rectified)
    rec[name] = contrast_sig

    return rec


def add_contrast(rec, name='contrast', source_name='stim', ms=500, bins=None,
                 continuous=False, normalize=False, dlog=False,
                 percentile=50, ignore_zeros=True, IsReload=False, **context):
    '''xforms wrapper for make_contrast_signal'''
    rec_with_contrast = make_contrast_signal(
            rec, name=name, source_name=source_name, ms=ms, bins=bins,
            percentile=percentile, normalize=normalize, dlog=dlog,
            ignore_zeros=ignore_zeros, continuous=continuous
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
        modelspec = _init_double_exponential(rec, modelspec, dsig_idx)
    else:
        modelspec = _init_logistic_sigmoid(rec, modelspec, dsig_idx)

    # TODO: This doesn't seem to help so far, but might want to revisit
    #       later. Currently just fitting all parameters together.
    # Start with modulation forced to 0, then release bounds after prefit.
#    modelspec[dsig_idx]['bounds'] = {
#            'amplitude_mod': (0, 0),
#            'base_mod': (0, 0),
#            'kappa_mod': (0, 0),
#            'shift_mod': (0, 0),
#            }

    return modelspec


def freeze_dsig_statics(modelspec):
    modelspec = copy.deepcopy(modelspec)
    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        log.warning("No dsig module was found, can't initialize.")
        return modelspec

    p = modelspec[dsig_idx]['phi']
    frozen_bounds = {k: (v, v) for k, v in p.items()}
    modelspec[dsig_idx]['bounds'].update(frozen_bounds)

    return modelspec


def remove_dsig_bounds(modelspec):
    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        log.warning("No dsig module was found, can't initialize.")
        return modelspec
    modelspec = copy.deepcopy(modelspec)
    modelspec[dsig_idx]['bounds'].update({
            'base': (1e-15, None),
            'amplitude': (1e-15, None),
            'shift': (None, None),
            'kappa': (1e-15, None),
            'amplitude_mod': (None, None),
            'base_mod': (None, None),
            'kappa_mod': (None, None),
            'shift_mod': (None, None)
            })
    return modelspec


def _init_logistic_sigmoid(rec, modelspec, dsig_idx):

    if dsig_idx == len(modelspec):
        fit_portion = modelspec
    else:
        fit_portion = modelspec[:dsig_idx]

    # generate prediction from module preceeding dexp

    # HACK
    for i, m in enumerate(fit_portion):
        if not m.get('phi', None):
            old = m.get('prior', {})
            m = priors.set_mean_phi([m])[0]
            m['prior'] = old
            fit_portion[i] = m

    ms.fit_mode_on(fit_portion)
    rec = ms.evaluate(rec, fit_portion)
    ms.fit_mode_off(fit_portion)

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

    #mod = ('Normal', {'mean': 0, 'sd': 1})

    # TODO: Forcing mods to start at 0 wasn't working very well, but
    #       maybe try just Normal(0, 1) or something?
    #       Does seem odd to set these like this, b/c it makes the model
    #       start off with pretty big modulators some times.
    modelspec[dsig_idx]['prior'].update({
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa,
#            'base_mod': mod, 'amplitude_mod': mod, 'shift_mod': mod,
#            'kappa_mod': mod,
            })

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
            old = m.get('prior', {})
            m = priors.set_mean_phi([m])[0]
            m['prior'] = old
            fit_portion[i] = m

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

    amp = ('Normal', {'mean': amp, 'sd': 1.0})
    base = ('Normal', {'mean': base, 'sd': 1.0})
    kappa = ('Normal', {'mean': kappa, 'sd': 1.0})
    shift = ('Normal', {'mean': shift, 'sd': 1.0})

    modelspec[target_i]['prior'].update({
            'base': base, 'amplitude': amp, 'shift': shift, 'kappa': kappa,
#            'base_mod': 0, 'amplitude_mod': 0, 'shift_mod': 0, 'kappa_mod': 0,
            })


    return modelspec


def dsig_phi_to_prior(modelspec):
    modelspec = copy.deepcopy(modelspec)
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
    p['shift'][1]['mean'] = s  # Do anything to scale sd?
    p['kappa'][1]['beta'] = k

    return modelspec


def init_contrast_model(est, modelspecs, IsReload=False,
                        tolerance=10**-5.5, max_iter=1000,
                        fitter='scipy_minimize', metric='nmse', **context):
    if IsReload:
        return {}

    modelspec = copy.deepcopy(modelspecs[0])
    if not find_module('dynamic_sigmoid', modelspec):
        new_ms = nems.initializers.prefit_LN(est, modelspec, max_iter=max_iter,
                                             tolerance=tolerance)
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
    modelspec = _prefit_contrast_modules(
                    est, modelspec, fit_basic,
                    fitter=fitter_fn,
                    metric=metric_fn,
                    fit_kwargs=fit_kwargs
                    )

    # TODO: only necessary if we start initializing bounds at 0 again.
    # Unlock bonds on modulators, then freeze values for statics,
    # then prefit modulators only, then unlock everything again.
#    modelspec = remove_dsig_bounds(modelspec)
#    modelspec = freeze_dsig_statics(modelspec)
#    modelspec = prefit_mod_subset(
#            est, modelspec, fit_basic,
#            fit_set=['dynamic_sigmoid'],
#            fitter=fitter_fn,
#            metric=metric_fn,
#            fit_kwargs=fit_kwargs)
#    modelspec = remove_dsig_bounds(modelspec)

    # after prefitting contrast modules, update priors to reflect the
    # prefit values so that random sample fits incorporate the prefit info.
    modelspec = dsig_phi_to_prior(modelspec)

    return {'modelspecs': [modelspec],
            'est': est}


def _prefit_contrast_modules(rec, modelspec, analysis_function,
                             fitter, metric=None, fit_kwargs={}):
    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    fit_idx = []
    tmodelspec = []
    fit_set = ['ctwc', 'ctfir', 'ctlvl', 'dsig']
    for i, m in enumerate(modelspec):
        m = copy.deepcopy(m)
        for id in fit_set:
            if id in m['id']:
                fit_idx.append(i)
                log.info('Found module %d (%s) for subset prefit', i, id)
        tmodelspec.append(m)

    if len(fit_idx) == 0:
        log.info('No modules matching fit_set for subset prefit')
        return modelspec

    exclude_idx = np.setdiff1d(np.arange(0, len(modelspec)),
                               np.array(fit_idx))
    for i in exclude_idx:
        m = tmodelspec[i]
        if not m.get('phi'):
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            old = m.get('prior', {})
            m = priors.set_mean_phi([m])[0]  # Inits phi
            m['prior'] = old

        log.info('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        tmodelspec[i] = m

    # fit the subset of modules
    if metric is None:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       fit_kwargs=fit_kwargs)[0]
    else:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)[0]

    # reassemble the full modelspec with updated phi values from tmodelspec
    for i in fit_idx:
        modelspec[i] = tmodelspec[i]

    return modelspec
