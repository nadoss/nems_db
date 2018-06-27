from nems.plugins.default_keywords import wc, stp, dlog


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


wc_combinations = {}
wcc_combinations = {}

for n_in in (15, 18, 40):
    for n_out in (1, 2, 3, 4):
        for op in ('', 'g', 'g.n'):
            old_k = 'wc%s%dx%d' % (op.strip('.'), n_in, n_out)
            new_k = 'wc.%dx%d.%s' % (n_in, n_out, op)
            wc_combinations[old_k] = _aliased_keyword(wc, new_k)

for n_in in (1, 2, 3):
    for n_out in (1, 2, 3, 4):
        for op in ('c', 'n'):
            old_k = 'wc%s%dx%d' % (op, n_in, n_out)
            new_k = 'wc.%dx%d.%s' % (n_in, n_out, op)

stp2b = _aliased_keyword(stp, 'stp.2.b')
stpz2 = _aliased_keyword(stp, 'stp.2.z')
stpn1 = _aliased_keyword(stp, 'stp.1.n')
stpn2 = _aliased_keyword(stp, 'stp.2.n')
dlogz = _aliased_keyword(stp, 'dlog')
dlogf = _aliased_keyword(dlog, 'dlog.f')
