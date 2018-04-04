# -*- coding: utf-8 -*-

import nems_db.db as nd
import nems_db.baphy as nb
import nems_db.xform_wrappers as nw

options = {}
options["stimfmt"] = "ozgf"
options["chancount"] = 18
options["rasterfs"] = 100
options['includeprestim'] = 1
#options["average_stim"]=True
#options["state_vars"]=[]
options["cellid"]=['TAR010c-02-1', 'TAR010c-07-1', 'TAR010c-09-1',
       'TAR010c-11-1', 'TAR010c-12-1', 'TAR010c-13-1', 'TAR010c-15-1']

cellid = 'TAR010c-18-1'
batch=271

rec=nb.baphy_load_recording(cellid,batch,options)

#rec2=nb.baphy_load_recording_nonrasterized(cellid,batch,options)


# if this complains about a missing nems_db.baphy function, look in baphy_deprecated

