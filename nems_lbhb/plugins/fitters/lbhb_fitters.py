from nems.plugins.fitters.default_fitters import basic


def _aliased_fitter(fn, loadkey):
    '''Forces the keyword fn to use the given fitkey. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <all-alpha kw_head><numbers> paradigm.
    '''
    def ignorant_fitter(ignored_key):
        return fn(loadkey)
    return ignorant_fitter


fit01 = _aliased_fitter(basic, 'basic')
# TODO: add other aliases from normalization branch
