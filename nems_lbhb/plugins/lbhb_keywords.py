from nems.plugins.default_keywords import wc, stp, dlog
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
    m = fir(kw[2:])
    m['fn_kwargs'].update({'i': 'ctpred', 'o': 'ctpred'})
    return m


def _aliased_keyword(fn, kw):
    '''Forces the keyword fn to use the given kw. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <kw_head>.<option1>.<option2> paradigm.
    '''
    def ignorant_keyword(ignored_key):
        return fn(kw)
    return ignorant_keyword


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
