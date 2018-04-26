import nems_db.db as nd
import nems_db.baphy as nb
import nems_db.xform_wrappers as nw
import numpy as np

batch = 303
test_save = False

if batch == 271:
    cellid = 'TAR010c-18-1'
    options = {}
    options["stimfmt"] = "ozgf"
    options["chancount"] = 18
    options["rasterfs"] = 100
    options['includeprestim'] = 1
    options['stim'] = True

elif batch == 301:
    cellid = 'BRT026c-02-1'
    options = {'rasterfs': 20, 'includeprestim': True, 'stimfmt': 'parm',
               'chancount': 0, 'pupil': True, 'stim': False,
               'pupil_deblink': True, 'pupil_median': 1}

elif batch == 303:
    cellid = 'bbl071d-a1'
    options = {'rasterfs': 20, 'includeprestim': True, 'stimfmt': 'parm',
               'chancount': 0, 'pupil': True, 'stim': False,
               'pupil_deblink': True, 'pupil_median': 1}

elif batch == 306:
    cellid = 'fre196b-15-2'
    options = {}
    options["stimfmt"] = "envelope"
    options["chancount"] = 0
    options["rasterfs"] = 100
    options['includeprestim'] = 1


rec = nb.baphy_load_recording(cellid, batch, options)
rec2 = nb.baphy_load_recording_nonrasterized(cellid, batch, options)

if 'stim' in rec.signals.keys():
    stim1 = rec['stim']
    stim2 = rec2['stim'].rasterize()
    assert (np.sum(np.square(stim1.as_continuous()-stim2.as_continuous())))==0

resp1=rec['resp']
resp2=rec2['resp'].rasterize()
assert (np.sum(np.square(resp1.as_continuous()-resp2.as_continuous())))==0

if test_save:

    dataroot = '/tmp/test/'
    # rec['resp'].save(dataroot+'resp1/')
    # rec2['resp'].save(dataroot+'resp2/')
    # rec['stim'].save(dataroot+'stim1/')
    # rec2['stim'].save(dataroot+'stim2/')

    rec.save(dataroot+'rec1/')
    rec2.save(dataroot+'rec2/')