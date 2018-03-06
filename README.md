# NEMS DB API

This repository contains code for saving modelspecs, logfiles, images, and recordings to a database accessible via HTTP. 

## Installation

SSH into the server that will be running this API. Then install the following required packages:

```
pip3 install flask --user
pip3 install flask_restful --user
```

Get a copy of this repository and `cd` into it.

Copy the configuration in "default_config.sh" to a new file and edit the new copy. Then run:

```
source configs/my_config.sh   # Sets environment variables
python3 -m nems_db
```

Note that `NEMS_DB_API_HOST` should be localhost for testing, but be sure to make it an externally-facing address if you want anybody else to be able to use it.


## Testing the API

In a new terminal on that same machine, you should be able to test that the API is working using:

```
curl TODO
```
