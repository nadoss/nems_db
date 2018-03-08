from flask import Flask
from flask_restful import Api
from nems_db.api import UploadResultsInterface, QueryInterface, UploadRecordingInterface
from nems_db.util import ensure_env_vars

req_env_vars = [
        'NEMS_DB_API_HOST',
        'NEMS_DB_API_PORT',
        'NEMS_RECORDINGS_DIR',
        'NEMS_RESULTS_DIR',
        'MYSQL_HOST',
        'MYSQL_USER',
        'MYSQL_PASS',
        'MYSQL_DB',
        'MYSQL_PORT'
        ]

# Load the credentials, throwing an error if any are missing
creds = ensure_env_vars(req_env_vars)

app = Flask(__name__)
api = Api(app)

api.add_resource(UploadResultsInterface,
                 '/results/<path:objpath>',
                 resource_class_kwargs={'upload_dir': creds['NEMS_RESULTS_DIR']})

api.add_resource(UploadRecordingInterface,
                 '/recordings/<path:recording_filepath>',
                 resource_class_kwargs={'upload_dir': creds['NEMS_RECORDINGS_DIR']})

api.add_resource(QueryInterface,
                 '/query',
                 resource_class_kwargs={'nems_db_host': creds['NEMS_DB_API_HOST'],
                                        'search_dir': creds['NEMS_RESULTS_DIR'],
                                        'result_route': '/results'})

app.run(port=int(creds['NEMS_DB_API_PORT']),
        host=creds['NEMS_DB_API_HOST'])
