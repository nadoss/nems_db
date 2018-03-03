import nems_baphy.db as nd
import nems_baphy.utilities.baphy as baphy


def list_recordings(batch):
    '''
    Returns a list of all cellids for a given batch.
    '''
    cell_data = nd.get_batch_cells(batch=batch)
    cellids = list(cell_data['cellid'].unique())
    return cellids


def load_recording_from_baphy(batch, cellid, **options):
    '''
    Returns a recording object loaded from Baphy, or None
    if no matching object was found.
    '''
    if batch in [271]:
        return _load_from_271(cellid, **options)
    # TODO: OTHER BATCHES HERE
    else:
        ## TODO: log a warning
        return None


# By using functions, we can set some default options for the batch
# in a way that will throw an error if an unexpected option is given.
def _load_from_271(cellid,
                   rasterfs=100,
                   chancount=18,
                   includeprestim=True,
                   pupil=False,
                   stimfmt='ozgf',
                   stim=True):
    '''
    Returns a Recording loaded from a cell in batch 271.
    '''
    batch = 271
    stimfmt = stimfmt if stim else 'none'
    options = {'rasterfs': rasterfs,
               'chancount': chancount,
               'stimfmt': stimfmt,
               'includeprestim': includeprestim,
               'pupil': pupil,
               'stim': stim}
    recording = baphy.baphy_load_recording(cellid, batch, options)
    return recording
