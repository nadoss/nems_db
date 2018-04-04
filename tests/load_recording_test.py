import nems_db.db as nd
import nems_db.baphy as nb
import nems_db.xform_wrappers as nw
import numpy as np

options = {}
options["stimfmt"] = "ozgf"
options["chancount"] = 18
options["rasterfs"] = 100
options['includeprestim'] = 1
options['stim']=True
# options["average_stim"]=True
# options["state_vars"]=[]

cellid = 'TAR010c-18-1'
batch = 271

rec = nb.baphy_load_recording(cellid, batch, options)
rec2= nb.baphy_load_recording_nonrasterized(cellid, batch, options)

stim1=rec['stim']
stim2=rec2['stim'].rasterize()
resp1=rec['resp']
resp2=rec2['resp'].rasterize()

assert (np.sum(np.square(stim1.as_continuous()-stim2.as_continuous())))==0

assert (np.sum(np.square(resp1.as_continuous()-resp2.as_continuous())))==0

dataroot='/tmp/test/'
#rec['resp'].save(dataroot+'resp1/')
#rec2['resp'].save(dataroot+'resp2/')
#rec['stim'].save(dataroot+'stim1/')
#rec2['stim'].save(dataroot+'stim2/')

rec.save(dataroot+'rec1/')
rec2.save(dataroot+'rec2/')