def format_keywordstring(rec, keywordstring=None, **context):
    n_targets = rec.meta['n_targets']
    keywordstring = keywordstring.replace('NTARGETS', str(n_targets))
    return {
        'keywordstring': keywordstring.format(ntargets=n_targets),
    }


def fix_modelspec(modelspec, **context):
    for ms in modelspec.raw:
        ms[-1]['fn_kwargs']['i'] = 'pred'
    return {'modelspec': modelspec}

