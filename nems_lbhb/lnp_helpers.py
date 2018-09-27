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


def _lnp_metric(data, pred_name='pred', resp_name='resp'):
    # Translate SVD lab kwargs to be more readable for this model
    rate_name = pred_name
    spikes_name = resp_name

    # NOTE : don't do the averaging step when fitting this model
    # NOTE : 200hz (or maybe higher?) sampling encouraged to get
    #        at most 1 spike per bin as much as possible.

    # For stephen's lab: rate_name usually 'pred'
    rate_vector = data[rate_name].as_continuous().flatten()
    spike_signal = data[spikes_name]
    fs = spike_signal.fs
    spike_train = spike_signal.as_continuous()
    spikes = np.argwhere(spike_train).flatten()

    # negative logikelihood:
    # (1-integral of mu dt)(mu dt) (product for all spikes)

    stim_errors = []
    stim_dict = _stack_reps(spike_signal)

    # TODO: what to set initial to? keep it as 0? random? ISI-based?
    integral = integrate.cumtrapz(rate_vector, initial=0).flatten()

    for stim in stim_dict:
        rep_errors = []

        for rep in stim_dict[stim]:
            start_seconds, end_seconds = rep
            start_bin = round(start_seconds*fs)
            end_bin = round(end_seconds*fs)

            loglikes = []
            t0 = start_bin

            for st in spikes[start_bin: end_bin]:
                try:
                    loglike = np.log(
                            (integral[st] - integral[t0])*(rate_vector[st])
                            )
                except IndexError:
                    raise IndexError("Error when indexing integral for: "
                                     "stim: %s, rep: (%f, %f)"
                                     % (stim, start_seconds, end_seconds))
                loglikes.append(loglike)
                t0 = st

            rep_errors.append(sum(loglikes)*-1)

        stim_errors.append(np.nanmean(rep_errors))

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


def _stack_reps(spike_train, ep='^STIM_'):
    stim_dict = {}
    epochs = spike_train.epochs

    for name in sorted(set(nems.epoch.epoch_names_matching(epochs, ep))):

        rows = epochs[epochs.name == name]
        reps = zip(
                rows['name'].values.tolist(),
                rows['start'].values.tolist(),
                rows['end'].values.tolist()
                )

        for name, start, stop in reps:
            if name in stim_dict:
                stim_dict[name].append((start, stop))
            else:
                stim_dict[name] = [(start, stop)]

    return stim_dict