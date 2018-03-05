import json
import requests
import urllib.parse

from nems.modelspec import get_modelspec_shortname, get_modelspec_metadata


HOST = 'potoroo:3002'  # Include port here too if desired


def as_path(recording, modelname, fitter, date, filename):
    ''' Returns a relative path. '''
    if not recording and modelname and fitter and date:
        raise ValueError('Not all necessary fields defined!')
    path = recording + '/' + modelname + '/' + fitter + '/' + date + '/' + filename
    return path


def as_url(**kwargs):
    return 'http://' + HOST + '/upload/' + as_path(**kwargs)


def save_modelspec(modelspec):
    '''
    Saves a single modelspec to the nems_db.
    Returns None if it succeeded, else an error message.
    '''
    meta = get_modelspec_metadata(modelspec)
    url = as_url(modelname=get_modelspec_shortname(modelspec),
                 recording=meta['recording'],
                 fitter=meta['fitter'],
                 date=meta['date'],
                 filename='log.txt')
    # r = requests.put(url, json=modelspec)
    r = requests.put(url, data='hello world')
    if r.status_code == 200:
        return None
    else:
        error_message = '{}: {}'.format(r.status_code, r.text)
        return error_message


with open('/home/ivar/git/nems/modelspecs/TAR010c-18-1.wc18x1_lvl1_fir15x1_dexp1.fit_basic.2018-02-26T19:28:57.0000.json', 'r') as f:
    modelspec = json.loads(f.read())
    error = save_modelspec(modelspec)
    


