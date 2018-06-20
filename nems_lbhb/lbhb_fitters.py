def generate_fitter_xfspec(fitkey, fitkey_kwargs=None):

    xfspec = []
    pfolds = 5

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
        xfspec.append(['nems.xforms.fit_basic_shr_init', {}])
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

    elif fitkey == "state01-jkm":

        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_state_init', {}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "state01-jk":

        xfspec.append(['nems.xforms.mask_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_state_init', {}])
        xfspec.append(['nems.xforms.fit_nfold', {}])  # 'ftol': 1e-6
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "state01-jk-shr":

        xfspec.append(['nems.xforms.mask_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_state_init', {}])
        xfspec.append(['nems.xforms.fit_nfold_shrinkage', {}])
        xfspec.append(['nems.xforms.predict', {}])

    elif (fitkey == "basic-nf"):

        xfspec.append(['nems.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "cd-nf":

        xfspec.append(['nems.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_cd_nfold', {'ftol': 1e-6}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif (fitkey == "fitpjk01") or (fitkey == "basic-nfm"):

        xfspec.append(['nems.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict', {}])

    elif (fitkey == "basic-nftrial"):

        log.info("n-fold fitting...")
        tfolds = 5
        xfspec.append(['nems.xforms.mask_for_jackknife',
                       {'njacks': tfolds, 'epoch_name': 'TRIAL'}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "basic-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_nfold_shrinkage', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "cd-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.mask_for_jackknife',
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

    elif fitkey == "fitsubs":
        '''fit_subsets with scipy_minimize'''
        kw_list = ['module_sets', 'tolerance', 'fitter']
        defaults = [None, 1e-4, coordinate_descent]
        module_sets, tolerance, my_fitter = \
            _get_my_kwargs(fitkey_kwargs, kw_list, defaults)
        xfspec.append([
                'nems.xforms.fit_module_sets',
                {'module_sets': module_sets, 'fitter': scipy_minimize,
                 'tolerance': tolerance}
                ])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey.startswith("fitsubs"):
        xfspec.append(_parse_fitsubs(fitkey))
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "fititer":
        kw_list = ['module_sets', 'tolerances', 'tol_iter', 'fit_iter',
                   'fitter']
        defaults = [None, None, 100, 20, coordinate_descent]
        module_sets, tolerances, tol_iter, fit_iter, my_fitter = \
            _get_my_kwargs(fitkey_kwargs, kw_list, defaults)
        xfspec.append([
                'nems.xforms.fit_iteratively',
                {'module_sets': module_sets, 'fitter': my_fitter,
                 'tolerances': tolerances, 'tol_iter': tol_iter,
                 'fit_iter': fit_iter}
                ])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey.startswith("fititer") or fitkey.startswith("iter"):
        xfspec.append(['nems.xforms.fit_iter_init', {}])
        xfspec.append(_parse_fititer(fitkey))
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey.startswith("state"):
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_state_init', {}])
        xfspec.append(_parse_fititer(fitkey))
        xfspec.append(['nems.xforms.predict', {}])

    else:
        raise ValueError('unknown fitter string ' + fitkey)

    return xfspec


def _get_my_kwargs(kwargs, kw_list, defaults):
    '''Fetch value of kwarg if given, otherwise corresponding default'''
    my_kwargs = []
    for kw, default in zip(kw_list, defaults):
        if kwargs is None:
            a = default
        else:
            a = kwargs.pop(kw, default)
        my_kwargs.append(a)
    return my_kwargs


def _parse_fititer(fit_keyword):
    # ex: fititer01-T4-T6-S0x1-S0x1x2x3-ti50-fi20
    # fitter: scipy_minimize; tolerances: [1e-4, 1e-6]; s
    # subsets: [[0,1], [0,1,2,3]]; tol_iter: 50; fit_iter: 20;
    # Note that order does not matter except for starting with
    # 'fititer<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = 'scipy_minimize'
    elif fit.endswith('02'):
        fitter = 'coordinate_descent'
    else:
        fitter = 'coordinate_descent'
        log.warn("Unrecognized or unspecified fit algorithm for fititer: %s\n"
                 "Using default instead: %s", fit, fitter)

    tolerances = []
    module_sets = []
    fit_iter = None
    tol_iter = None

    for c in chunks[1:]:
        if c.startswith('ti'):
            tol_iter = int(c[2:])
        elif c.startswith('fi'):
            fit_iter = int(c[2:])
        elif c.startswith('T'):
            power = int(c[1:])*-1
            tol = 10**(power)
            tolerances.append(tol)
        elif c.startswith('S'):
            indices = [int(i) for i in c[1:].split('x')]
            module_sets.append(indices)
        else:
            log.warning(
                    "Unrecognized segment in fititer keyword: %s\n"
                    "Correct syntax is:\n"
                    "fititer<fitter>-S<i>x<j>...-T<tolpower>...ti<tol_iter>"
                    "-fi<fit_iter>", c
                    )

    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return ['nems.xforms.fit_iteratively',
            {'module_sets': module_sets, 'fitter': fitter,
             'tolerances': tolerances, 'tol_iter': tol_iter,
             'fit_iter': fit_iter}]


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