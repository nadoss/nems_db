import logging

import numpy as np

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


def _lnp_metric(data, prediction_name, response_name):
    # Pass pred into poisson spike generator (optional)

    # TODO: currently using scipy_minimize, so would want negative likelihood?
    #       or just make a wrapper for scipy_maximize (presumably that exists)
    # Assess likelihood to calculate error

    # negative logikelihood: (1-integral of mu dt)(mu dt) (product for all spikes)
    # pseudocode:

    loglikes = []
    t0 = 0
    for st in spkes:
        loglike = log(pred(st) - np.trapz(pred(...)))
        loglikes.append(loglike)
        t0 = st

    error = sum(loglikes)*-1

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


    pass

