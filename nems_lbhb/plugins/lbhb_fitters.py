from nems.plugins.default_fitters import basic


# TODO: This really isn't a fitter but this is where it has to go for new
#       setup for now. Should be a 'postprocessing' function.
def dynamic(fitkey):
    return [['nems_lbhb.contrast_helpers.dynamic_logsig', {}]]


def _aliased_fitter(fn, fitkey):
    '''Forces the keyword fn to use the given fitkey. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <all-alpha kw_head><numbers> paradigm.
    '''
    def ignorant_fitter(ignored_key):
        return fn(fitkey)
    ignorant_fitter.key = fitkey
    return ignorant_fitter


# NOTE: Using the new keyword syntax is encouraged since it improves
#       readability; however, for exceptionally long keywords or ones
#       that get used very frequently, aliases can be implemented as below.
#       If your alias is causing errors, ask Jacob for help.


fitjk01 = _aliased_fitter(basic, 'basic.nf5.epREFERENCE')
