from copy import deepcopy

import numpy as np


def _get_global_gain_sf(rec, gain):
    is_repeating = rec['repeating'].as_continuous()[0].astype('bool')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')

    gain_mapped = gain[t_map]
    gain0 = np.zeros(t_map.shape, dtype=np.double)

    # Set gains for repeating portions of ss and ds
    gain0[is_repeating] = gain_mapped[is_repeating]

    # Convert to scaling factor
    sf = np.exp(gain0)

    return sf


def global_gain(rec, gain):
    sf = _get_global_gain_sf(rec, gain)
    bg = rec['bg_pred'].as_continuous()
    fg = rec['fg_pred'].as_continuous()
    pred = (bg + fg) * sf
    pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')
    return [pred_signal]


def _get_relative_gain_sf(rec, fg_gain, bg_gain, single_gain):
    is_repeating = rec['repeating'].as_continuous()[0].astype('bool')
    dual_stream = rec['dual_stream'].as_continuous()[0].astype('bool')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')

    fg_gain_mapped = fg_gain[t_map]
    bg_gain_mapped = bg_gain[t_map]
    single_gain_mapped = single_gain[t_map] + single_gain[0]
    fg_gain = np.zeros(t_map.shape, dtype=np.double)
    bg_gain = np.zeros(t_map.shape, dtype=np.double)

    # Set gains for repeating portions of ss and ds
    fg_gain[is_repeating & dual_stream] = fg_gain_mapped[is_repeating & dual_stream]
    bg_gain[is_repeating & dual_stream] = bg_gain_mapped[is_repeating & dual_stream]
    fg_gain[is_repeating & ~dual_stream] = single_gain_mapped[is_repeating & ~dual_stream]

    # Convert to scaling factor
    bg_sf = np.exp(bg_gain)
    fg_sf = np.exp(fg_gain)

    return fg_sf, bg_sf


def relative_gain(rec, fg_gain, bg_gain, single_gain):
    fg_sf, bg_sf = _get_relative_gain_sf(rec, fg_gain, bg_gain, single_gain)
    bg = rec['bg_pred'].as_continuous()
    fg = rec['fg_pred'].as_continuous()

    pred = bg * bg_sf + fg * fg_sf
    pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')
    return [pred_signal]
