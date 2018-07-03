from nems.utils import find_module

# TODO: Move init_logsig and make_contrast from main nems repo to here?


def static_to_dynamic(modelspec):
    '''
    Changes bounds on contrast model to allow for dynamic modulation
    of the logistic sigmoid output nonlinearity.
    '''
    logsig_idx = find_module('logistic_sigmoid', modelspec)
    raise NotImplementedError
