# TODO: better name for this module?

import pandas as pd

from nems_db.xform_wrappers import load_model_baphy_xform
import nems.modelspec as ms


def fitted_parms_per_cell(cellids, batch, modelname, include_stats=True):
    # query nems_db results to get a list of modelspecs
    # (should end up with one modelspec per cell)
    modelspecs = _get_modelspecs(cellids, batch, modelname)
    stats = ms.summary_stats(modelspecs)

    index = stats.keys()
    columns = cellids
    data = {}
    for c in cellids:
        for k in index:
            if c in data.keys():
                data[c].append(stats[k]['values'])
            else:
                data[c] = [stats[k]['values']]

    if include_stats:
        columns.insert(0, 'std')
        data['std'] = []
        columns.insert(0, 'mean')
        data['mean'] = []
        for k in index:
            data['std'].append(stats[k]['std'])
            data['mean'].append(stats[k]['mean'])

    # TODO: more useful way to display the arrays?
    #       a list of 10 mxn ndarrays gets hard to show
    return pd.DataFrame(data=data, index=index, columns=columns)


def _get_modelspecs(cellids, batch, modelname):
    modelspecs = []
    for c in cellids:
        _, ctx = load_model_baphy_xform(c, batch, modelname, eval_model=False)
        mspecs = ctx['modelspecs']
        if len(mspecs) > 1:
            # TODO: Take mean? only want one modelspec per cell
            raise NotImplementedError("No support yet for multi-modelspec fit")
        else:
            this_mspec = mspecs[0]
        modelspecs.append(this_mspec)
    return modelspecs
