from flask import Flask
from flask_restful import Api
from nems_db.api import UploadInterface, QueryInterface
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

api.add_resource(
        UploadInterface,
        '/upload/<string:recording>/<string:model>/<string:fitter>/<string:date>/<string:filename>',
        resource_class_kwargs={'local_dir': creds['NEMS_RESULTS_DIR']}
        )

api.add_resource(QueryInterface,
                 '/query',
                 resource_class_kwargs={'host': creds['NEMS_RESULTS_DIR']})

app.run(
    port=int(creds['NEMS_DB_API_PORT']),
    host=creds['NEMS_DB_API_HOST']
    )
