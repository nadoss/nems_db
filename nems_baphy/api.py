import os
import io
import re

from flask import abort, Response, request
from flask_restful import Resource

# Define some regexes for sanitizing inputs
RECORDING_REGEX = re.compile(r"[\-_a-zA-Z0-9]+\.tar\.gz$")
CELLID_REGEX = re.compile(r"^[\-_a-zA-Z0-9]+$")
BATCH_REGEX = re.compile(r"^\d+$")


def as_path(recording, modelname, fitter, date):
    ''' Returns a relative path. '''
    if not recording and modelname and fitter and date:
        raise ValueError('Not all necessary fields defined!')
    url = recording + '/' + modelname + '/' + fitter + '/' + date + '/'
    return url


def valid_recording_filename(recording_filename):
    ''' Input Sanitizer.  True iff the filename has a valid format. '''
    matches = RECORDING_REGEX.match(recording_filename)
    return matches


def valid_cellid(cellid):
    ''' Input Sanitizer.  True iff the cellid has a valid format. '''
    matches = CELLID_REGEX.match(cellid)
    return matches


def valid_batch(batch):
    ''' Input Sanitizer.  True iff the batch has a valid format. '''
    matches = BATCH_REGEX.match(batch)
    return matches


def ensure_valid_cellid(cellid):
    if not valid_cellid(cellid):
        abort(400, 'Invalid cellid:' + cellid)


def ensure_valid_batch(batch):
    if not valid_batch(batch):
        abort(400, 'Invalid batch:' + batch)


def ensure_valid_recording_filename(rec):
    if not valid_recording_filename(rec):
        abort(400, 'Invalid recording:' + rec)


def not_found():
    abort(404, "Resource not found. ")


class ResultInterface(Resource):
    '''
    An interface for saving JSON files.
    '''
    def __init__(self, **kwargs):
        self.local_dir = kwargs['local_dir']

    def get(self, rec):
        '''
        Serves out a recording file in .tar.gz format.
        TODO: Replace with flask file server or NGINX
        '''
        ensure_valid_recording_filename(rec)
        filepath = os.path.join(self.targz_dir, rec)
        if not os.path.exists(filepath):
            not_found()
        d = io.BytesIO()
        with open(filepath, 'rb') as f:
            d.write(f.read())
            d.seek(0)
        return Response(d, status=200, mimetype='application/gzip')

    def put(self, recording, model, fitter, date):
        # If the put request is NOT a json, crash
        payload = request.json
        if not payload:
            abort(400, "Payload was not a json.")

        local_path = self.local_dir + '/' + as_path(recording, model, fitter, date)

        # TODO: If a file exists already, crash
        # if os.path.exists():
        #    abort(400, 'File already exists; will not overwrite.')
        print(local_path)

        # OK, it's a new file, so it's safe to write to disk
        # with open(local_path)
        print(payload)
        # Send back an OK
        return Response(None, status=200)

    def post(self, rec):
        abort(400, 'Not yet implemented')

    def delete(self, rec):
        abort(400, 'Not yet Implemented')
