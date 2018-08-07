#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:37:31 2018

@author: svd
"""

import copy
import numpy as np
import time
import logging

import nems.priors as priors
import nems.modelspec as ms
import nems.analysis.api as analysis
import nems.metrics.api as metrics
import nems.initializers as init

from nems.fitters.api import scipy_minimize, coordinate_descent
from nems.analysis.cost_functions import basic_cost
import nems.fitters.mappers
import nems.metrics.api
import nems.modelspec as ms

log = logging.getLogger(__name__)


def fit_population_slice(rec, modelspec, slice=0, fit_set=None,
                         analysis_function=analysis.fit_basic,
                         metric=metrics.nmse,
                         fitter=scipy_minimize, fit_kwargs={}):

    """
    fits a slice of a population model. modified from prefit_mod_subset

    slice: int
        response channel to fit
    fit_set: list
        list of mod names to fit

    """

    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    slice_count = rec['resp'].shape[0]
    if slice > slice_count:
        raise ValueError("Slice %d > slice_count %d", slice, slice_count)

    if fit_set is None:
        raise ValueError("fit_set list of module indices must be specified")

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    fit_idx = []
    tmodelspec = []
    sliceinfo = []
    for i, m in enumerate(modelspec):
        m = copy.deepcopy(m)
        # need to have phi in place
        if not m.get('phi'):
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            m = priors.set_mean_phi([m])[0]  # Inits phi

        for fn in fit_set:
            if fn in m['fn']:
                fit_idx.append(i)

                s = {}
                for key, value in m['phi'].items():
                    log.debug('Slicing %d (%s) key %s chan %d for fit',
                              i, fn, key, slice)

                    # keep only sliced channel(s)
                    if 'bank_count' in m['fn_kwargs'].keys():
                        bank_count = m['fn_kwargs']['bank_count']
                        filters_per_bank = int(value.shape[0] / bank_count)
                        slices = np.arange(slice*filters_per_bank,
                                           (slice+1)*filters_per_bank)
                        m['phi'][key] = value[slices, ...]
                        s[key] = slices
                        m['fn_kwargs']['bank_count'] = 1
                    elif value.shape[0] == slice_count:
                        m['phi'][key] = value[[slice], ...]
                        s[key] = [slice]
                    else:
                        raise ValueError("Not sure how to slice %s %s",
                                         m['fn'], key)

                # record info about how sliced this module parameter
                sliceinfo.append(s)

        tmodelspec.append(m)
#    print(sliceinfo)
    if len(fit_idx) == 0:
        log.info('No modules matching fit_set for slice fit')
        return modelspec

    exclude_idx = np.setdiff1d(np.arange(0, len(modelspec)),
                               np.array(fit_idx))
    for i in exclude_idx:
        m = tmodelspec[i]
        log.debug('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        tmodelspec[i] = m

    # generate temp recording with only resposnes of interest
    temp_rec = rec.copy()
    slice_chans = [temp_rec['resp'].chans[slice]]
    temp_rec['resp'] = temp_rec['resp'].extract_channels(slice_chans)

    # remove initial modules
    first_idx = fit_idx[0]
    if first_idx > 0:
        print('firstidx {}'.format(first_idx))
        temp_rec = ms.evaluate(temp_rec, tmodelspec, stop=first_idx)
        temp_rec['stim'] = temp_rec['pred'].copy()
        tmodelspec = tmodelspec[first_idx:]
        tmodelspec[0]['fn_kwargs']['i'] = 'stim'
        #print(tmodelspec)
    #print(temp_rec.signals.keys())

    # IS this mask necessary? Does it work?
    #if 'mask' in temp_rec.signals.keys():
    #    print("Data len pre-mask: %d" % (temp_rec['mask'].shape[1]))
    #    temp_rec = temp_rec.apply_mask()
    #    print("Data len post-mask: %d" % (temp_rec['mask'].shape[1]))

    # fit the subset of modules
    temp_rec = ms.evaluate(temp_rec, tmodelspec)
    error_before = metric(temp_rec)

    tmodelspec = analysis_function(temp_rec, tmodelspec, fitter=fitter,
                                   metric=metric, fit_kwargs=fit_kwargs)[0]

    temp_rec = ms.evaluate(temp_rec, tmodelspec)
    error_after = metric(temp_rec)
    dError = error_before - error_after
    if dError < 0:
        print("dError (%.6f - %.6f) = %.6f worse. not updating modelspec"
              % (error_before, error_after, dError))
    else:
        print("dError (%.6f - %.6f) = %.6f better. updating modelspec"
              % (error_before, error_after, dError))

        # reassemble the full modelspec with updated phi values from tmodelspec
        for i, mod_idx in enumerate(fit_idx):
            m = copy.deepcopy(modelspec[mod_idx])
            # need to have phi in place
            if not m.get('phi'):
                log.info('Intializing phi for module %d (%s)', mod_idx, m['fn'])
                m = priors.set_mean_phi([m])[0]  # Inits phi
            for key, value in tmodelspec[mod_idx - first_idx]['phi'].items():
    #            print(key)
    #            print(m['phi'][key].shape)
    #            print(sliceinfo[i][key])
    #            print(value)
                m['phi'][key][sliceinfo[i][key], :] = value

            modelspec[mod_idx] = m

    return modelspec


def fit_population_iteratively(
        est, modelspecs,
        cost_function=basic_cost,
        fitter=coordinate_descent, evaluator=ms.evaluate,
        segmentor=nems.segmentors.use_all_data,
        mapper=nems.fitters.mappers.simple_vector,
        metric=lambda data: nems.metrics.api.nmse(data, 'pred', 'resp'),
        metaname='fit_basic', fit_kwargs={},
        module_sets=None, invert=False, tolerances=None, tol_iter=50,
        fit_iter=10, IsReload=False, **context
        ):
    '''
    Required Arguments:
     est          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     fitter        A function of (sigma, costfn) that tests various points,
                   in fitspace (i.e. sigmas) using the cost function costfn,
                   and hopefully returns a better sigma after some time.
     mapper        A class that has two methods, pack and unpack, which define
                   the mapping between modelspecs and a fitter's fitspace.
     segmentor     An function that selects a subset of the data during the
                   fitting process. This is NOT the same as est/val data splits
     metric        A function of a Recording that returns an error value
                   that is to be minimized.

     module_sets   A nested list specifying which model indices should be fit.
                   Overall iteration will occurr len(module_sets) many times.
                   ex: [[0], [1, 3], [0, 1, 2, 3]]

     invert        Boolean. Causes module_sets to specify the model indices
                   that should *not* be fit.


    Returns
    A list containing a single modelspec, which has the best parameters found
    by this fitter.
    '''

#    if module_sets is None:
#        module_sets = []
#        for i, m in enumerate(modelspec):
#            if 'prior' in m.keys():
#                if 'levelshift' in m['fn'] and 'fir' in modelspec[i-1]['fn']:
#                    # group levelshift with preceding fir filter by default
#                    module_sets[-1].append(i)
#                else:
#                    # otherwise just fit each module separately
#                    module_sets.append([i])
#        log.info('Fit sets: %s', module_sets)

    if IsReload:
        return {}

    modelspec = modelspecs[0]
    data = est.copy()

    mod_set = [m['fn'] for m in modelspec]

    fit_set_all = []
    fit_set_slice = []
    pre_bank = True
    for m in mod_set:
        if 'filter_bank' in m:
            pre_bank = False
        if pre_bank:
            fit_set_all.append(m)
        else:
            fit_set_slice.append(m)

    if tolerances is None:
        tolerances = [1e-6]

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    # if 'mask' in data.signals.keys():
    #    log.info("Data len pre-mask: %d", data['mask'].shape[1])
    #    data = data.apply_mask()
    #    log.info("Data len post-mask: %d", data['mask'].shape[1])

    start_time = time.time()
    ms.fit_mode_on(modelspec)
    # Ensure that phi exists for all modules; choose prior mean if not found
    for i, m in enumerate(modelspec):
        if ('phi' not in m.keys()) and ('prior' in m.keys()):
            m = nems.priors.set_mean_phi([m])[0]  # Inits phi for 1 module
            log.debug('Phi not found for module, using mean of prior: {}'
                      .format(m))
            modelspec[i] = m

    error = np.inf

    slice_count = data['resp'].shape[0]
    step_size = 0.1
    for tol in tolerances:
        log.info("Fitting subsets with tol: %.2E fit_iter %d tol_iter %d",
                 tol, fit_iter, tol_iter)
        print("Fitting subsets with tol: %.2E fit_iter %d tol_iter %d" %
              (tol, fit_iter, tol_iter))
        cd_kwargs = fit_kwargs.copy()
        cd_kwargs.update({'tolerance': tol, 'max_iter': fit_iter,
                           'step_size': step_size})
        sp_kwargs = fit_kwargs.copy()
        sp_kwargs.update({'tolerance': tol, 'max_iter': fit_iter})

        i = 0
        error_reduction = np.inf
        while (error_reduction >= tol) and (i < tol_iter):

            improved_modelspec = copy.deepcopy(modelspec)
            cc = 0
            for s in range(slice_count):
                print('Slice %d set %s' % (s, ",".join(fit_set_slice)))
                improved_modelspec = fit_population_slice(
                        data, improved_modelspec, slice=s,
                        fit_set=fit_set_slice,
                        analysis_function=analysis.fit_basic,
                        metric=metric,
                        fitter=coordinate_descent,
                        fit_kwargs=cd_kwargs)

                # every 5 units, update the spectral filters
                cc += 1
                if (cc % 5 == 0) or (cc == slice_count):
                    print('Slice %d updating pop-wide parameters' % (s))
                    improved_modelspec = init.prefit_mod_subset(
                            data, improved_modelspec, analysis.fit_basic,
                            metric=metric,
                            fit_set=fit_set_all,
                            fit_kwargs=sp_kwargs)

            new_error = metric(data)
            error_reduction = error - new_error
            error = new_error
            log.info("tol=%.2E, iter=%d/%d: deltaE=%.6E",
                     tol, i, tol_iter, error_reduction)
            print("tol=%.2E, iter=%d/%d: max deltaE=%.6E" %
                  (tol, i, tol_iter, error_reduction))
            i += 1
            if error_reduction > 0:
                modelspec = improved_modelspec

        log.info("Done with tol %.2E (i=%d, max_error_reduction %.7f)",
                 tol, i, error_reduction)
        print("Done with tol %.2E (i=%d, max_error_reduction %.7f)" %
                 (tol, i, error_reduction))

        step_size *= 0.5

    elapsed_time = (time.time() - start_time)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)
    results = [copy.deepcopy(improved_modelspec)]

    return {'modelspecs': results}

