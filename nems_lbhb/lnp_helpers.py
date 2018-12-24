import logging

import numpy as np
from scipy import integrate

import nems
import nems.metrics.api as metrics

log = logging.getLogger(__name__)

# TODO: add to initialization for fir
            # c = c/np.norm(c)
            # r, zf = scipy.signal.lfilter( a*c, [1], x_, zi=zi)

# TODO: provide derivative to optimizer

# TODO: look at scikit learn alternatives to be able to get more
#       info out of the fitter?

# TODO: need to change init for levelshift to log of mean firing rate instead?

# TODO: normalize error range for easier tolerance

# TODO: figure out a way to use both relative and absolute precision?

# Note on combining the channels for fir filter:
# they do just get added together to get the single vector afterward.

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

    # For stephen's lab: rate_name usually 'pred'
    rate_vector = data[rate_name].as_continuous().flatten()
    spike_train = data[spikes_name].as_continuous().flatten()
    spike_train = (spike_train > 0).astype('int')

    spikes = np.argwhere(spike_train)

    # TODO: what to set initial to? keep it as 0? random? ISI-based?
    #       Maybe don't need to worry about this with non-cumulative version?
    integral = integrate.trapz(rate_vector)

    epsilon = 1e-100
    # Get bins corresponding to spike times and rectify to epsilon
    rate_at_spikes = rate_vector[spikes]
    rate_at_spikes[rate_at_spikes < epsilon] = epsilon
    log_mu_dts = np.log(rate_at_spikes)
    # multiplying by 1/bins to get per-bin error
    error = (1/rate_vector.size)*(-1*integral + np.sum(log_mu_dts))*-1

    # SVD previous implementation:
    #error = np.mean(spike_train*np.log(rate_vector) - rate_vector)

    # sanity check: after fitting a model, sample from it and simulate
    # synthetic data, then try to fit that data and see if you can
    # recovery the model. (SVD was talking about this for GC model as well).

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


# TODO: Need to come up with some way to verify that this is
#       behaving as expected
def simulate_spikes(rate_vector):
    integral = integrate.cumtrapz(rate_vector, initial=0)
    spikes = np.zeros_like(integral)
    base = 0
    random = np.asscalar(np.random.rand(1))
    for i,r in enumerate(integral):
        if (random < 1-np.exp(-r+base)):
            spikes[i] = 1
            base = r
            random = np.asscalar(np.random.rand(1))

    return spikes
