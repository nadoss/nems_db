"""
preprocessor keywords specific to LBHB models
should occur after a loader keyword but before the modelspec keywords
several functions migrated out of old loader keywords
"""

import logging
import re

log = logging.getLogger(__name__)


def pas(loadkey, recording_uri):
    """
    pas = "passive only"
    mask out everything that doesn't fall in a "PASSIVE_EXPERIMENT" epoch
    """

    xfspec = [['nems.preprocessing.mask_keep_passive',
               {}, ['rec'], ['rec']]]

    return xfspec

def ref(kw):
    return [['nems.xforms.mask_all_but_correct_references', {}]]


def evs(loadkey, recording_uri):
    """
    evs = "event stimulus"
    currently this is specific to using target onset events and lick events
    as the "stimuli"

    broken out of evt loader keyword
    """

    pattern = re.compile(r'^evs\.([a-zA-Z0-9\.]*)$')
    parsed = re.match(pattern, loadkey)
    loader = parsed.group(1)

    state_signals = []
    permute_signals = []
    epoch2_shuffle = False

    if loader == ("pupbehtarlic"):
        state_signals = ['active', 'pupil']
        permute_signals = []
    elif loader == ("pup0behtarlic"):
        state_signals = ['active', 'pupil']
        permute_signals = ['pupil']
    elif loader == ("pupbeh0tarlic"):
        state_signals = ['active', 'pupil']
        permute_signals = ['active']
    elif loader == ("pup0beh0tarlic"):
        state_signals = ['active', 'pupil']
        permute_signals = ['active', 'pupil']
    elif loader == ("pupbehtarlic0"):
        state_signals = ['active', 'pupil']
        permute_signals = []
        epoch2_shuffle = True
    elif loader == ("pup0behtarlic0"):
        state_signals = ['active', 'pupil']
        permute_signals = ['pupil']
        epoch2_shuffle = True
    elif loader == ("pupbeh0tarlic0"):
        state_signals = ['active', 'pupil']
        permute_signals = ['active']
        epoch2_shuffle = True
    elif loader == ("pup0beh0tarlic0"):
        state_signals = ['active', 'pupil']
        permute_signals = ['active', 'pupil']
        epoch2_shuffle = True
    elif loader == ("tarlic"):
        state_signals = []
        permute_signals = ['active', 'pupil']
    elif loader == ("tarlic0"):
        state_signals = []
        permute_signals = []
        epoch2_shuffle = True
    else:
        raise ValueError("unknown signals for alt-stimulus initializer")

    xfspec = [['nems.preprocessing.generate_stim_from_epochs',
               {'new_signal_name': 'stim',
                'epoch_regex': '^TAR_', 'epoch_shift': 5,
                'epoch2_regex': 'LICK', 'epoch2_shift': -5,
                'epoch2_shuffle': epoch2_shuffle, 'onsets_only': True},
               ['rec'], ['rec']],
              ['nems.xforms.mask_all_but_targets', {}]]

    return xfspec


def st(loadkey, recording_uri):
    """
    st = "state variable"
    generate a state signal

    broken out of evt/psth/etc loader keywords
    """
    pattern = re.compile(r'^st\.([a-zA-Z0-9\.]*)$')
    parsed = re.match(pattern, loadkey)
    loader = parsed.group(1)

    state_signals = []
    permute_signals = []

    loadset = loader.split(".")
    for l in loadset:
        if l.startswith("beh"):
            this_sig = ["active"]
        elif l.startswith("pup"):
            this_sig = ["pupil"]
        elif l.startswith("pxb"):
            this_sig = ["p_x_a"]
        elif l.startswith("pre"):
            this_sig = ["pre_passive"]
        elif l.startswith("dif"):
            this_sig = ["puretone_trials", "hard_trials"]
        elif l.startswith("pbs"):
            this_sig = ["pupil_bs"]
        elif l.startswith("pev"):
            this_sig = ["pupil_ev"]
        elif l.startswith('pup_cd'):
            this_sig = ["pupil_cd"]
        elif l.startswith("pas"):
            this_sig = ["each_passive"]
        else:
            raise ValueError("unknown signal code %s for state variable initializer", l)

        state_signals.extend(this_sig)
        if l.endswith("0"):
            permute_signals.extend(this_sig)

    xfspec = [['nems.xforms.make_state_signal',
               {'state_signals': state_signals,
                'permute_signals': permute_signals,
                'new_signalname': 'state'}]]
    return xfspec


def hrc(load_key, recording_uri):
    """
    Mask only data during stimuli that were repeated 10 or greater times.
    hrc = high rep count
    """
    # preprocessing is in Charlie's auto users
    xfspec = [['preprocessing.mask_high_repetion_stims',
               {'epoch_regex':'^STIM_'}, ['rec'], ['rec']]]

    return xfspec


def psthfr(load_key, recording_uri):
    """
    Generate psth from resp
    """
    options = load_key.split('.')[1:]
    smooth = ('s' in options)
    epoch_regex = '^STIM_'
    xfspec=[['nems.xforms.generate_psth_from_resp',
                   {'smooth_resp': smooth, 'epoch_regex': epoch_regex}]]
    return xfspec


# TODO: Maybe can keep splitep and avgep as one thing?
#       Would they ever be done separately?
def splitep(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}]]
    return xfspec


def avgep(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    return [['nems.xforms.average_away_stim_occurrences',
             {'epoch_regex': epoch_regex}]]


def sev(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}],
        ['nems.xforms.average_away_stim_occurrences',
         {'epoch_regex': epoch_regex}]]
    return xfspec


