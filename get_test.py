import json
import requests
from urllib.parse import urlencode

from nems.modelspec import get_modelspec_name, get_modelspec_metadata


HOST='potoroo:3005'  # Include port here too if desired

def as_path(recording, modelname, fitter, date):
    ''' Returns a relative path. '''
    if not recording and modelname and fitter and date:
        raise ValueError('Not all necessary fields defined!')
    path = recording + '/' + modelname + '/' + fitter + '/' + date + '/'
    return path


def as_url(**kwargs):
    return 'http://' + HOST + '/' + as_path(**kwargs)


def load_modelspec(recording, modelname, fitter, date):
    url = as_url(modelname=modelname, recording=recording,
                 fitter=fitter, date=date)
    r = requests.get(url)
    print(str(r))

load_modelspec(
        recording='TAR010c-18-1',
        modelname='TAR010c-18-1.wc18x1_lvl1_fir15x1_dexp1.fit_basic.2018-03-04T03:32:25',
        fitter='fit_basic',
        date='2018-02-26T19:28:57'
        )