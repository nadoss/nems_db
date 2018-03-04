import os
import io
import re

from flask import abort, Response, request
from flask_restful import Resource
from sqlalchemy import and_
#    https://webargs.readthedocs.io/en/latest/
from webargs import fields
from webargs.flaskparser import use_kwargs

from .db import NarfResults, Session

# Define some regexes for sanitizing inputs
RECORDING_REGEX = re.compile(r"[\-_a-zA-Z0-9]+\.tar\.gz$")
CELLID_REGEX = re.compile(r"^[\-_a-zA-Z0-9]+$")
BATCH_REGEX = re.compile(r"^\d+$")

query_args = {
    'batch': fields.Int(required=False),
    'cellid': fields.Str(required=False),
    'recording': fields.Str(required=False),
    # TODO: Want to keep preproc keywords separate?
    'preproc': fields.Str(required=False),
    'modelname': fields.Str(required=False),
    # TODO: Want to keep fit keywords separate?
    'fitter': fields.Str(required=False),
    'date': fields.str(required=False)
}

# results args are almost the same, but some are required.
# smarter way to just change arguments to required=True?
results_args = {
    # modelname incorporates cellid, batch and preproc
    'recording': fields.Str(required=True),
    # TODO: Want to keep preproc keywords separate?
    'modelname': fields.Str(required=True),
    # TODO: Want to keep fit keywords separate?
    'fitter': fields.Str(required=True),
    'date': fields.str(required=False)
}


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
    An interface for saving and retrieving JSON files.
    TODO
    '''
    def __init__(self, **kwargs):
        self.local_dir = kwargs['local_dir']

    def get(self, rec):
        '''
        Serves out a modelspec file in .json.
        TODO: Replace with flask file server or NGINX
        '''
        return abort(400, 'Not implemented')

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

class QueryInterface(Resource):
    '''
    An interface for retrieving lists of matching results,
    which may be retrieved through ResultInterface methods.
    '''
    def __init__(self, **kwargs):
        # TODO: no kwargs needed so far, maybe just leave out definition?
        pass

    @use_kwargs(query_args)
    def get(self, **kwargs):
        '''
        Return a JSON list of tuples:
        (recording, modelname, fitter, <date; default to most recent>)
        that match all provided kwargs:
        batch, cellid, preproc, recording, modelname, fitter, date.
        '''
        abort(400, 'Not yet implemented')
        # compose filters
        # TODO: allow list as well as string arguments?
        filters = []
        for key, val in kwargs.items():
            col = getattr(NarfResults, key)
            if isinstance(val, str):
                filters.append(col.ilike(val))
            # TODO
            #elif isinstance(val, list):
            #    filters.append(col.in_(val))
        # open a database session and retrieve matched results
        session = Session()
        db_objs = session.query(NarfResults).filter(and_(*filters)).all()
        results = [
                (r.recording, r.modelanme, r.fitter, r.date)
                for r in db_objs
                ]
        # close database connection before exiting scope
        session.close()
        return Response(results, status=200, mimetype='application/json')