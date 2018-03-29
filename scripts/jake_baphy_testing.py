import nems_db.db as nd
import nems_db.baphy as nb
import nems_db.xform_wrappers as nw
import nems.signal

from memory_profiler import profile

#@profile
def setup():
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

    return cellid, batch, options


#@profile
def main(cellid, batch, options):
    event_times, spike_dict, stim_dict, state_dict = \
        nb.baphy_load_dataset(
                '/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m', options
                )

    spikes = nems.signal.SignalTimeSeries(fs=100, data=spike_dict, name='resp',
                                          recording=cellid, chans=['Response'],
                                          epochs=event_times)

    # TODO: How to build an intelligble stim_chans list?
    first_key = list(stim_dict.keys())[0]
    first_stim = stim_dict[first_key]
    spectral_chans, _ = first_stim.shape

    stim_chans = ['%d_kHz' % i for i in range(spectral_chans)]
    stims = nems.signal.SignalDictionary(fs=100, data=stim_dict, name='stim',
                                         recording=cellid, chans=stim_chans,
                                         epochs=event_times)

    # after first _matrix call, memory usage should be comparable
    # to rasterized signal. should see be decrease after delete.
    spikes._matrix
    spikes.delete_cached_matrix
    print("\n")

    stims._matrix
    stims.delete_cached_matrix
    print("\n")

    path = '/auto/users/jacob/sigtest/test_stim.hdf5'
    stims.save(path)
    loaded_stims = nems.signal.SignalDictionary.load(path)
    print(loaded_stims._matrix.shape)
    print(loaded_stims.epochs.size)

    path = '/auto/users/jacob/sigtest/test_spike.hdf5'
    spikes.save(path)
    loaded_spikes = nems.signal.SignalTimeSeries.load(path)
    print(loaded_spikes._matrix.shape)
    print(loaded_spikes.epochs.size)

if __name__ == '__main__':
    cellid, batch, options = setup()
    main(cellid, batch, options)
