import logging

from nems.fitters.api import coordinate_descent, scipy_minimize

log = logging.getLogger(__name__)

"""
template:

user creates file with this function:

def mykeyword(loaderkey, recording_uri):
    xfspec=[]
    return xfspec


user adds this to nems configuration:

nems.config.settings.loader_plugin = ['mylib']


then xform_helper finds out where loader plug-ins are and runs this:

xform_helper.loader_lib['mykeyword'] = mylib.mykeyword
"""

def generate_loader_xfspec(loader, recording_uri):

    recordings = [recording_uri]

    if loader in ["ozgf100ch18", "ozgf100ch18n"]:
        normalize = int(loader == "ozgf100ch18n")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences', {}]]

    elif loader in ["ozgf100ch18pup", "ozgf100ch18npup"]:
        normalize = int(loader == "ozgf100ch18npup")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': ['pupil'], 'permute_signals': [],
                    'new_signalname': 'state'}]]

    elif loader in ["env100", "env100n"]:
        normalize = int(loader == "env100n")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences', {}]]

    elif loader in ["env100pt", "env100ptn"]:
        normalize = int(loader == "env100ptn")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}]]

    elif loader == "nostim10pup":
        # DEPRECATED?
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings}],
                  ['nems.preprocessing.make_state_signal',
                   {'state_signals': ['pupil'], 'permute_signals': [],
                    'new_signalname': 'state'}, ['rec'], ['rec']]]

    elif (loader.startswith("psth") or loader.startswith("nostim") or
          loader.startswith("env")):

        if loader.endswith("beh0"):
            state_signals = ['active']
            permute_signals = ['active']
        elif loader.endswith("beh"):
            state_signals = ['active']
            permute_signals = []
        elif loader.endswith("pup0beh0"):
            state_signals = ['pupil', 'active']
            permute_signals = ['pupil', 'active']
        elif loader.endswith("pup0beh"):
            state_signals = ['pupil', 'active']
            permute_signals = ['pupil']
        elif loader.endswith("pupbeh0"):
            state_signals = ['pupil', 'active']
            permute_signals = ['active']
        elif loader.endswith("pupbeh"):
            state_signals = ['pupil', 'active']
            permute_signals = []

        elif loader.endswith("pup0pre0beh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = ['pupil', 'pre_passive']
        elif loader.endswith("puppre0beh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = ['pre_passive']
        elif loader.endswith("pup0prebeh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = ['pupil']
        elif loader.endswith("pupprebeh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = []

        elif loader.endswith("pre0beh0"):
            state_signals = ['pre_passive', 'active']
            permute_signals = ['pre_passive', 'active']
        elif loader.endswith("pre0beh"):
            state_signals = ['pre_passive', 'active']
            permute_signals = ['pre_passive']
        elif loader.endswith("prebeh0"):
            state_signals = ['pre_passive', 'active']
            permute_signals = ['active']
        elif loader.endswith("prebeh"):
            state_signals = ['pre_passive', 'active']
            permute_signals = []

        elif loader.endswith("predif0beh"):
            state_signals = ['pre_passive', 'puretone_trials',
                             'hard_trials', 'active']
            permute_signals = ['puretone_trials', 'hard_trials']
        elif loader.endswith("predifbeh"):
            state_signals = ['pre_passive', 'puretone_trials',
                             'hard_trials', 'active']
            permute_signals = []
        elif loader.endswith("pbs0pev0beh0"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_bs', 'pupil_ev', 'active']
        elif loader.endswith("pbspev0beh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_ev']
        elif loader.endswith("pbs0pevbeh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_bs']
        elif loader.endswith("pbspevbeh0"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_bs', 'pupil_ev']
        elif loader.endswith("pbs0pev0beh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['active']
        elif loader.endswith("pbspevbeh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = []
        else:
            raise ValueError("invalid loader string")

        if loader.startswith("psths"):
            xfspec = [['nems.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems.xforms.generate_psth_from_resp',
                       {'smooth_resp': True}],
                      ['nems.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}]]
        elif loader.startswith("psth"):
            xfspec = [['nems.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems.xforms.generate_psth_from_resp', {}],
                      ['nems.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}]]
        elif loader.startswith("env"):
            xfspec = [['nems.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}]]
        else:
            xfspec = [['nems.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}]]

    else:
        raise ValueError('unknown loader string')

    return xfspec


def generate_fitter_xfspec(fitkey, fitkey_kwargs=None):

    xfspec = []
    pfolds = 20

    # parse the fit spec: Use gradient descent on whole data set(Fast)
    if fitkey in ["fit01", "basic"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey in ["fit01a", "basicqk"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic',
                       {'max_iter': 1000, 'tolerance': 1e-5}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey in ["fit01b", "basic-shr"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic',
                       {'shrinkage': 1, 'tolerance': 1e-8}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey in ["fit01b", "basic-cd"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic_cd', {'shrinkage': 0}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey in ["fit01b", "basic-cd-shr"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic_cd',
                       {'shrinkage': 1, 'tolerance': 1e-8}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "fitjk01":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "state01-jk":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_state_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif (fitkey == "fitpjk01") or (fitkey == "basic-nf"):

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "basic-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_nfold_shrinkage', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "cd-nf":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_cd_nfold', {'tolerance': 1e-6}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "cd-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_cd_nfold_shrinkage', {}])
        xfspec.append(['nems.xforms.predict',    {}])

#    elif fitkey == "iter-cd-nf-shr":
#
#        log.info("Iterative cd, n-fold, shrinkage fitting...")
#        xfspec.append(['nems.xforms.split_for_jackknife',
#                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
#        #xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
#        xfspec.append(['nems.xforms.fit_iter_cd_nfold_shrink', {}])
#        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "fit02":
        # no pre-fit
        log.info("Performing full fit...")
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey.startswith("fitsubs"):
        xfspec.append(_parse_fitsubs(fitkey))
        xfspec.append(['nems.xforms.predict', {}])

    else:
        raise ValueError('unknown fitter string ' + fitkey)

    return xfspec


def _parse_fitsubs(fit_keyword):
    # ex: fitsubs02-S0x1-S0x1x2x3-it1000-T6
    # fitter: scipy_minimize; subsets: [[0,1], [0,1,2,3]];
    # max_iter: 1000;
    # Note that order does not matter except for starting with
    # 'fitsubs<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = scipy_minimize
    elif fit.endswith('02'):
        fitter = coordinate_descent
    else:
        fitter = coordinate_descent
        log.warn("Unrecognized or unspecified fit algorithm for fitsubs: %s\n"
                 "Using default instead: %s", fit[7:], fitter)

    module_sets = []
    max_iter = None
    tolerance = None

    for c in chunks[1:]:
        if c.startswith('it'):
            max_iter = int(c[2:])
        elif c.startswith('S'):
            indices = [int(i) for i in c[1:].split('x')]
            module_sets.append(indices)
        elif c.startswith('T'):
            power = int(c[1:])*-1
            tolerance = 10**(power)
        else:
            log.warning(
                    "Unrecognized segment in fitsubs keyword: %s\n"
                    "Correct syntax is:\n"
                    "fitsubs<fitter>-S<i>x<j>...-T<tolpower>-it<max_iter>", c
                    )

    if not module_sets:
        module_sets = None

    return ['nems.xforms.fit_iteratively',
            {'module_sets': module_sets, 'fitter': fitter,
             'tolerance': tolerance, 'max_iter': max_iter}]
