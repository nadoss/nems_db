import nems_db.db as nd
import nems_db.baphy as nb
import nems_db.xform_wrappers as nw
import nems.signal

cellid = 'TAR010c-18-1'
batch = 271

options = {}
options["stimfmt"] = "ozgf"
options["chancount"] = 18
options["rasterfs"] = 100
options['includeprestim'] = 1
# options["average_stim"]=True
# options["state_vars"]=[]

# copied from load recording to make sure defaults are set
options['rasterfs'] = int(options.get('rasterfs', 100))
options['stimfmt'] = options.get('stimfmt', 'ozgf')
options['chancount'] = int(options.get('chancount', 18))
options['pertrial'] = int(options.get('pertrial', False))
options['includeprestim'] = options.get('includeprestim', 1)

options['pupil'] = int(options.get('pupil', False))
options['pupil_deblink'] = int(options.get('pupil_deblink', 1))
options['pupil_median'] = int(options.get('pupil_deblink', 1))
options['stim'] = int(options.get('stim', True))
options['runclass'] = options.get('runclass', None)
options['cellid'] = options.get('cellid', cellid)
options['batch'] = int(batch)

event_times, spike_dict, stim_dict, state_dict = \
    nb.baphy_load_dataset('/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m',
                          options)

spikes = nems.signal.SignalTimeSeries(fs=100, data=spike_dict, name='resp',
                                      recording=cellid, chans=['Response'],
                                      epochs=event_times)
