from nems.plugins.default_keywords import wc, fir, lvl, stp, dlog
import re
import logging

import numpy as np

log = logging.getLogger(__name__)


def ctwc(kw):
    '''
    Same as nems.plugins.keywords.fir but renamed for contrast
    to avoid confusion in the modelname and allow different
    options to be supported if needed.
    '''
    m = wc(kw[2:])
    m['fn_kwargs'].update({'i': 'contrast', 'o': 'ctpred'})
    return m


def ctfir(kw):
    '''
    Same as nems.plugins.keywords.fir but renamed for contrast
    to avoid confusion in the modelname and allow different
    options to be supported if needed.
    '''
    # TODO: Support separate bank for each logsig parameter?
    #       Or just skip straight to the CD model?

    pattern = re.compile(r'^ctfir\.?(\d{1,})x(\d{1,})x?(\d{1,})?$')
    parsed = re.match(pattern, kw)
    n_outputs = int(parsed.group(1))
    n_coefs = int(parsed.group(2))
    n_banks = parsed.group(3)  # None if not given in keyword string
    if n_banks is None:
        n_banks = 1
    else:
        n_banks = int(n_banks)

    p_coefficients = {
        'mean': np.zeros((n_outputs * n_banks, n_coefs)),
        'sd': np.ones((n_outputs * n_banks, n_coefs)),
    }

    if n_coefs > 2:
        # p_coefficients['mean'][:, 1] = 1
        # p_coefficients['mean'][:, 2] = -0.5
        p_coefficients['mean'][:, 1] = 1
    else:
        p_coefficients['mean'][:, 0] = 1

    template = {
            'fn': 'nems.modules.fir.filter_bank',
            'fn_kwargs': {'i': 'ctpred', 'o': 'ctpred', 'bank_count': n_banks},
            'prior': {
                'coefficients': ('Normal', p_coefficients)},
            }

#    p_coefficients = {'beta': np.full((n_outputs * n_banks, n_coefs), 0.1)}
#    template = {
#            'fn': 'nems.modules.fir.filter_bank',
#            'fn_kwargs': {'i': 'ctpred', 'o': 'ctpred', 'bank_count': n_banks},
#            'prior': {'coefficients': ('Exponential', p_coefficients)},
#            }

    return template


def ctlvl(kw):
    '''
    Same as nems.plugins.keywords.lvl but renamed for
    contrast.
    '''
    m = lvl(kw[2:])
    m['fn_kwargs'].update({'i': 'ctpred', 'o': 'ctpred'})
    return m


def dsig(kw):
    '''
    Note: these priors will typically be overwritten during initialization
          based on the input signal.
    '''
    ops = kw.split('.')[1:]
    eq = 'logsig'
    amp = False
    base = False
    kappa = False
    shift = False
    for op in ops:
        if op in ['logsig', 'l']:
            eq = 'logsig'
        elif op in ['dexp', 'd']:
            eq = 'dexp'
        elif op == 'a':
            amp = True
        elif op == 'b':
            base = True
        elif op == 'k':
            kappa = True
        elif op == 's':
            shift = True

    # Use all by default. Use none not an option (would just be static version)
    if (not amp) and (not base) and (not kappa) and (not shift):
        amp = True; base = True; kappa = True; shift = True

    template = {
        'fn': 'nems_lbhb.contrast_helpers.dynamic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'c': 'ctpred',
                      'eq': eq},
        'prior': {'base': ('Exponential', {'beta': [0.1]}),
                  'amplitude': ('Exponential', {'beta': [2.0]}),
                  'shift': ('Normal', {'mean': [1.0], 'sd': [1.0]}),
                  'kappa': ('Exponential', {'beta': [0.5]})}
#                  'base_mod': ('Exponential', {'beta': [0.1]}),
#                  'amplitude_mod': ('Exponential', {'beta': [2.0]}),
#                  'shift_mod': ('Normal', {'mean': [1.0], 'sd': [1.0]}),
#                  'kappa_mod': ('Exponential', {'beta': [0.5]})}
        }

    zero_norm = ('Normal', {'mean': [0.0], 'sd': [1.0]})

    if amp:
        template['prior']['amplitude_mod'] = zero_norm
    else:
        template['fn_kwargs']['amplitude_mod'] = 0

    if base:
        template['prior']['base_mod'] = zero_norm
    else:
        template['fn_kwargs']['base_mod'] = 0

    if kappa:
        template['prior']['kappa_mod'] = zero_norm
    else:
        template['fn_kwargs']['kappa_mod'] = 0

    if shift:
        template['prior']['shift_mod'] = zero_norm
    else:
        template['fn_kwargs']['shift_mod'] = 0

    return template


def _aliased_keyword(fn, kw):
    '''Forces the keyword fn to use the given kw. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <kw_head>.<option1>.<option2> paradigm.
    '''
    def ignorant_keyword(ignored_key):
        return fn(kw)
    return ignorant_keyword


def _one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def sdexp(kw):
    '''
    Generate and register modulespec for the state_dexp

    Parameters
    ----------
    kw : str
        Expected format: r'^sdexp\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^sdexp\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_vars = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "sdexp.{n_state_variables} \n"
                         "keyword given: %s" % kw)

    zeros = np.zeros(n_vars)
    ones = np.ones(n_vars)
    g_mean = _one_zz(n_vars-1)
    g_sd = ones
    d_mean = zeros
    d_sd = ones
    n_dims = 2 # one for gain, one for dc
    base_mean = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    base_sd = np.ones([n_dims, 1]) if n_dims > 1 else np.array([1])
    amp_mean = base_mean + 0.2
    amp_sd = base_mean + 0.1
    #shift_mean = base_mean
    #shift_sd = base_sd
    kappa_mean = base_mean
    kappa_sd = amp_sd


    template = {
        'fn': 'nems_lbhb.modules.state.state_dexp',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                  'd': ('Normal', {'mean': d_mean, 'sd': d_sd}),
                  'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
        }

    return template


# Old keywords that are identical except for the period
# (e.x. dexp1 vs dexp.1 or wc15x2 vs wc.15x2)
# don't need to be aliased, but more complicated ones that had options
# picked apart (like wc.NxN.n.g.c) will need to be aliased.


# These aren't actually  needed anymore since we separated old models
# from new ones, but gives an example of how aliasing can be done.

#wc_combinations = {}
#wcc_combinations = {}
#
#for n_in in (15, 18, 40):
#    for n_out in (1, 2, 3, 4):
#        for op in ('', 'g', 'g.n'):
#            old_k = 'wc%s%dx%d' % (op.strip('.'), n_in, n_out)
#            new_k = 'wc.%dx%d.%s' % (n_in, n_out, op)
#            wc_combinations[old_k] = _aliased_keyword(wc, new_k)
#
#for n_in in (1, 2, 3):
#    for n_out in (1, 2, 3, 4):
#        for op in ('c', 'n'):
#            old_k = 'wc%s%dx%d' % (op, n_in, n_out)
#            new_k = 'wc.%dx%d.%s' % (n_in, n_out, op)
#
#stp2b = _aliased_keyword(stp, 'stp.2.b')
#stpz2 = _aliased_keyword(stp, 'stp.2.z')
#stpn1 = _aliased_keyword(stp, 'stp.1.n')
#stpn2 = _aliased_keyword(stp, 'stp.2.n')
#dlogz = _aliased_keyword(stp, 'dlog')
#dlogf = _aliased_keyword(dlog, 'dlog.f')
