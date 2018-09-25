import logging

import numpy as np
from scipy import integrate

import nems
import nems.metrics.api as metrics

log = logging.getLogger(__name__)


def lnp_basic(modelspecs, est, max_iter=1000, tolerance=1e-7,
              metric='nmse', IsReload=False, fitter='scipy_minimize',
              cost_function=None, **context):

    if not IsReload:

        fitter_fn = getattr(nems.fitters.api, fitter)
        fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

        modelspecs = [
                nems.analysis.api.fit_basic(est, modelspec,
                                            fit_kwargs=fit_kwargs,
                                            metric=_lnp_metric,
                                            fitter=fitter_fn)[0]
                for modelspec in modelspecs
                ]

    return {'modelspecs': modelspecs}


def _lnp_metric(data, rate_name, spikes_name):
    # NOTE : don't do the averaging step when fitting this model
    # NOTE : 200hz (or maybe higher?) sampling encouraged to get
    #        at most 1 spike per bin as much as possible.

    # For stephen's lab: rate_name usually 'pred'
    rate_vector = data[rate_name]
    spike_train = data[spikes_name]
    spikes = np.argwhere(spike_train).flatten()

    # negative logikelihood: (1-integral of mu dt)(mu dt) (product for all spikes)
    # pseudocode:

    stim_errors = []
    for stim in _stack_reps(spike_train):
        trial_errors = []

        for trial in stim:
            loglikes = []
            t0 = 0
            # TODO: what to set initial to? keep it as 0? random? ISI-based?
            integral = integrate.cumtrapz(rate_vector, initial=0)
            for st in spikes:
                loglike = np.log((integral[st] - integral[t0])*(rate_vector[st]))
                loglikes.append(loglike)
                t0 = st

            trial_errors.append(sum(loglikes)*-1)

        stim_errors.append(np.nanmean(trial_errors))

    error = np.nanmean(stim_errors)

    # TODO: vectorize/broadcast with numpy to improve speed

    # TODO: only do integration once, then just do int(t) - int(t-1) when
    #       iterating through

    # TODO: add something for first spike? maybe add mean ISI before t0 = 0?


    # sanity check: after fitting a model, sample from it and simulate
    # synthetic data, then try to fit that data and see if you can
    # recovery the model. (SVD was talking about this for GC model as well).

    # extra: generate spikes from the fitted models & do stuff with those:

    # ex algorithm from AD:
    # choose to gen 10 spikes
    # pick 10 random numbers from 0 to 1
    # do cum. sum, find times for which random numbers equal integral of mu
    # integrate for long time?
    # supposed to recover spikes of response

    return error
