from nems.plugins.default_keywords import wc, fir


def ctwc(kw):
    '''
    Same as nems.plugins.keywords.wc but renamed for contrast
    to avoid confusion in the modelname and allow different
    options to be supported if needed.
    '''
    return wc(kw)


def ctfir(kw):
    '''
    Same as nems.plugins.keywords.fir but renamed for contrast
    to avoid confusion in the modelname and allow different
    options to be supported if needed.
    '''
    return fir(kw)


def _aliased_keyword(fn, kw):
    '''Forces the keyword fn to use the given kw. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <kw_head>.<option1>.<option2> paradigm.
    '''
    def ignorant_keyword(ignored_key):
        return fn(kw)
    return ignorant_keyword
