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


def save_modelspec(modelspec):
    ''' Saves a single modelspec to the nems_db.'''
    meta = get_modelspec_metadata(modelspec)
    url = as_url(modelname=get_modelspec_name(modelspec),
                 recording=meta['recording'],
                 fitter=meta['fitter'],
                 date=meta['date'])
    print(url, modelspec)
    r = requests.put(url, data=str(modelspec))
    # TODO: check that the put request succeed; if not, throw exception, or save to disk?
    return True


with open('/home/ivar/git/nems/modelspecs/TAR010c-18-1.wc18x1_lvl1_fir15x1_dexp1.fit_basic.2018-02-26T19:28:57.0000.json', 'r') as f:
    modelspec = json.loads(f.read())
    save_modelspec(modelspec)
