#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:37:31 2018

@author: svd
"""

import copy
import numpy as np
import time
import random

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
from nems.utils import find_module

from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import default_keywords
from nems.plugins import default_loaders
from nems.plugins import default_initializers
from nems.plugins import default_fitters

import logging
log = logging.getLogger(__name__)


def init_pop_pca(rec, modelspec):
    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    ifir=find_module('filter_bank', modelspec)
    iwc=find_module('weight_channels', modelspec)

    chan_count=modelspec[ifir]['fn_kwargs']['bank_count']
    chan_per_bank = int(modelspec[iwc]['prior']['mean'][1]['mean'].shape[0]/chan_count)
    rec = rec.copy()
    tmodelspec = copy.deepcopy(modelspec)

    kw=[m['id'] for m in modelspec[:iwc]]

    wc = modelspec[iwc]['id'].split(".")
    wcs = wc[1].split("x")
    wcs[1]=str(chan_per_bank)
    wc[1]="x".join(wcs)
    wc=".".join(wc)

    fir = modelspec[ifir]['id'].split(".")
    fircore = fir[1].split("x")
    fir[1]="x".join(fircore[:-1])
    fir = ".".join(fir)

    kw.append(wc)
    kw.append(fir)
    kw.append("lvl.1")
    keywordstring="-".join(kw)
    keyword_lib = KeywordRegistry()
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

    for pc_idx in range(chan_count):
        r = rec['pca'].extract_channels([rec['pca'].chans[pc_idx]])
        m = np.nanmean(r.as_continuous())
        d = np.nanstd(r.as_continuous())
        rec['resp'] = r._modified_copy((r._data-m) / d)
        tmodelspec = init.from_keywords(keyword_string=keywordstring,
                                        meta={}, registry=keyword_lib, rec=rec)
        tolerance=1e-4
        tmodelspec = init.prefit_LN(rec, tmodelspec,
                                    tolerance=tolerance, max_iter=700)

        # save results back into main modelspec
        itfir=find_module('fir', tmodelspec)
        itwc=find_module('weight_channels', tmodelspec)

        if pc_idx==0:
            for tm, m in zip(tmodelspec[:(iwc+1)], modelspec[:(iwc+1)]):
                m['phi']=tm['phi'].copy()
            modelspec[ifir]['phi']=tmodelspec[itfir]['phi'].copy()
        else:
            for k, v in tmodelspec[iwc]['phi'].items():
                modelspec[iwc]['phi'][k]=np.concatenate((modelspec[iwc]['phi'][k],v))
            for k, v in tmodelspec[itfir]['phi'].items():
                #if k=='coefficients':
                #    v/=100 # kludge
                modelspec[ifir]['phi'][k]=np.concatenate((modelspec[ifir]['phi'][k],v))
    return modelspec


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

    if type(fit_set[0]) is int:
        fit_idx = fit_set
    else:
        fit_idx = []
        for i, m in enumerate(modelspec):
            for fn in fit_set:
                if fn in m['fn']:
                    fit_idx.append(i)

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    tmodelspec = []
    sliceinfo = []
    for i, m in enumerate(modelspec):
        m = copy.deepcopy(m)
        # need to have phi in place
        if not m.get('phi'):
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            m = priors.set_mean_phi([m])[0]  # Inits phi

        if i in fit_idx:
            s = {}
            for key, value in m['phi'].items():
                log.debug('Slicing %d (%s) key %s chan %d for fit',
                          i, m['fn'], key, slice)

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
        #print('firstidx {}'.format(first_idx))
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
        log.info("dError (%.6f - %.6f) = %.6f worse. not updating modelspec"
              % (error_before, error_after, dError))
    else:
        log.info("dError (%.6f - %.6f) = %.6f better. updating modelspec"
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

    if IsReload:
        return {}

    """
    tolerances=[0.001, 0.0001]
    tol_iter=50
    fit_iter=5
    fitter='scipy_minimize'
    est=ctx['est']
    modelspecs=ctx['modelspecs']
    cost_function=basic_cost
    fitter=coordinate_descent
    evaluator=ms.evaluate
    segmentor=nems.segmentors.use_all_data
    mapper=nems.fitters.mappers.simple_vector
    metric=lambda data: nems.metrics.api.nmse(data, 'pred', 'resp')
    metaname='fit_basic'
    fit_kwargs={}
    module_sets=None
    """

    modelspec = copy.deepcopy(modelspecs[0])
    data = est.copy()

    bank_mod=find_module('filter_bank', modelspec, find_all_matches=True)
    wc_mod=find_module('weight_channels', modelspec, find_all_matches=True)

    if len(wc_mod)==2:
        fit_set_all = list(range(wc_mod[1]))
        fit_set_slice = list(range(wc_mod[1], len(modelspec)))
        if bank_mod:
            # provice a trivial non-zero intial condition for each
            # channel of the filterbank
            modelspec[bank_mod[0]]['prior']['coefficients'][1]['mean'][:,1] = 0.1
    elif len(bank_mod)==1:
        fit_set_all = list(range(bank_mod[0]))
        fit_set_slice = list(range(bank_mod[0],len(modelspec)))
    else:
        raise ValueError("Can't figure out how to split all and slices")

    if tolerances is None:
        tolerances = [1e-6]

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    # then delete the mask signal so that it's not reapplied on each fit
    if 'mask' in data.signals.keys():
        log.info("Data len pre-mask: %d", data['mask'].shape[1])
        data = data.apply_mask()
        log.info("Data len post-mask: %d", data['mask'].shape[1])
        del data.signals['mask']

    start_time = time.time()
    ms.fit_mode_on(modelspec)

    modelspec = init_pop_pca(data, modelspec)
    print(modelspec)

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
    if 'nonlinearity' in modelspec[-1]['fn']:
        skip_nl_first=True
        tolerances = [tolerances[0]] + tolerances
    else:
        skip_nl_first=False

    for toli,tol in enumerate(tolerances):

        log.info("Fitting subsets with tol: %.2E fit_iter %d tol_iter %d",
                 tol, fit_iter, tol_iter)
        print("Fitting subsets with tol: %.2E fit_iter %d tol_iter %d" %
              (tol, fit_iter, tol_iter))
        cd_kwargs = fit_kwargs.copy()
        cd_kwargs.update({'tolerance': tol, 'max_iter': fit_iter,
                           'step_size': step_size})
        sp_kwargs = fit_kwargs.copy()
        sp_kwargs.update({'tolerance': tol, 'max_iter': fit_iter})

        if (toli==0) and skip_nl_first:
            log.info('skipping nl on first tolerance loop')
            saved_modelspec = copy.deepcopy(modelspec)
            saved_fit_set_slice = fit_set_slice
            modelspec = modelspec[:-1]
            fit_set_slice = fit_set_slice[:-1]

        i = 0
        error_reduction = np.inf
        while (error_reduction >= tol) and (i < tol_iter):

            improved_modelspec = copy.deepcopy(modelspec)
            cc = 0
            slist = list(range(slice_count))
            #random.shuffle(slist)

            for s in slist:
                log.info('Slice %d set %s' % (s, fit_set_slice))
                improved_modelspec = fit_population_slice(
                        data, improved_modelspec, slice=s,
                        fit_set=fit_set_slice,
                        analysis_function=analysis.fit_basic,
                        metric=metric,
                        fitter=coordinate_descent,
                        fit_kwargs=cd_kwargs)

                cc += 1
                # if (cc % 8 == 0) or (cc == slice_count):
                if (cc == slice_count):
                    log.info('Slice %d updating pop-wide parameters', s)
                    print('Slice %d updating pop-wide parameters' % (s))
                    for m in modelspec:
                        print(m['fn'] + ": ", m['phi'])

                    improved_modelspec = init.prefit_mod_subset(
                            data, improved_modelspec, analysis.fit_basic,
                            metric=metric,
                            fit_set=fit_set_all,
                            fit_kwargs=sp_kwargs)

            data = ms.evaluate(data, improved_modelspec)
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

        if (toli == 0) and skip_nl_first:
            log.info('Restoring NL module after first tol loop')
            modelspec.append(saved_modelspec[-1])
            fit_set_slice = saved_fit_set_slice
            modelspec = init.init_dexp(data, modelspec)

            # just fit the NL
            improved_modelspec = copy.deepcopy(modelspec)

            kwa = cd_kwargs.copy()
            kwa['max_iter'] *= 2
            for s in range(slice_count):
                log.info('Slice %d set %s' % (s, [fit_set_slice[-1]]))
                improved_modelspec = fit_population_slice(
                        data, improved_modelspec, slice=s,
                        fit_set=fit_set_slice,
                        analysis_function=analysis.fit_basic,
                        metric=metric,
                        fitter=coordinate_descent,
                        fit_kwargs=kwa)
            data = ms.evaluate(data, modelspec)
            old_error = metric(data)
            data = ms.evaluate(data, improved_modelspec)
            new_error = metric(data)
            log.info('Init NL fit error change %.5f-%.5f = %.5f',
                     old_error, new_error, old_error-new_error)
            modelspec = improved_modelspec

        else:
            step_size *= 0.25

    elapsed_time = (time.time() - start_time)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)
    results = [copy.deepcopy(improved_modelspec)]

    return {'modelspecs': results}

