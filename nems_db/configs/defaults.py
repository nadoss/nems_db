# Note: Please make a copy of this file rather than editing it.
# By default, all configs in this directory (confifgs) will not be added
# to the git repo, in order to keep your secrets (host, pass, etc) safe.

NEMS_DB_API_HOST = 'localhost'
NEMS_DB_API_PORT = '3002'
NEMS_RESULTS_DIR = "/auto/data/nems_db/results/"
NEMS_RECORDINGS_DIR = "/auto/data/nems_db/recordings/"

SQL_ENGINE = 'sqlite'  # 'sqlite' or 'mysql' #
MYSQL_HOST = None
MYSQL_USER = None
MYSQL_PASS = None
MYSQL_DB = None
MYSQL_PORT ='3306'


# Default paths passed to command prompt for model queue
DEFAULT_EXEC_PATH = '/auto/users/nems/anaconda3/bin/python'
DEFAULT_SCRIPT_PATH = '/auto/users/nems/nems_db/nems_fit_single.py'
