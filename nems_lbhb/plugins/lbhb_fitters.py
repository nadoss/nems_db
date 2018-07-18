from nems.plugins.default_fitters import basic


# TODO: This really isn't a fitter but this is where it has to go for new
#       setup for now. Should be a 'postprocessing' function?

# TODO: Also... is this even really necessary? Maybe just give option to the
#       initializer to set bounds then fit th0 and th1 together.
def dyn(fitkey):
    options = fitkey.split('.')[1:]
    kwargs = {}
    for op in options:
        if op == 'a':
            kwargs['amplitude'] = True
        elif op == 'a0':
            kwargs['amplitude'] = False
        elif op == 'b':
            kwargs['base'] = True
        elif op == 'b0':
            kwargs['base'] = False
        elif op == 's':
            kwargs['shift'] = True
        elif op == 's0':
            kwargs['shift'] = False
        elif op == 'k':
            kwargs['kappa'] = True
        elif op == 'k0':
            kwargs['kappa'] = False
        elif op == ['all']:
            kwargs.update({'amplitude': True, 'base': True, 'shift': True,
                           'kappa': True})
        elif op == ['none']:
            kwargs.update({'amplitude': False, 'base': False, 'shift': False,
                           'kappa': False})

    return [['nems_lbhb.contrast_helpers.dynamic_logsig', kwargs]]


def srec(fitkey):
    return [['nems_lbhb.contrast_helpers.reset_single_recording', {}]]


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
