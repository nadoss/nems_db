#!/usr/bin/env python3

# This script runs nems_main.fit_single_model from the command line
import os
import sys

import nems
import nems_db.xform_wrappers as nw
import nems.xform_helper as xhelp
import nems.xforms as xforms
import nems.modelspec as ms

import nems.utils
import logging
log = logging.getLogger(__name__)

try:
    import nems_db.db as nd
    db_exists = True
except Exception as e:
    # If there's an error import nems.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems.db, can't update tQueue")
    print(e)
    db_exists = False

if __name__ == '__main__':
    os.chdir("/tmp")
    queueid = os.environ.get('QUEUEID', 0)

    if queueid:
        nems.utils.progress_fun = nd.update_job_tick
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)

    if len(sys.argv) < 4:
        print('syntax: nems_fit_pop siteid batch modelname')
        exit(-1)

    siteid = sys.argv[1]
    batch = sys.argv[2]
    modelname = sys.argv[3]

    # savefile = nw.fit_model_xforms_baphy(cellid,batch,modelname,saveInDB=True)
    savefile = nw.fit_pop_model_xforms_baphy(siteid, batch, modelname, saveInDB=True)

    log.info("Done with fit.")

    # Mark completed in the queue. Note that this should happen last thing!
    # Otherwise the job might still crash after being marked as complete.
    if queueid:
        nd.update_job_complete(queueid)


