import numpy as np


def rdtld(key):

    xfspec = [['nems_lbhb.rdt.io.load_recording', {}]]
    return xfspec


def rdtsev(key):

    xfspec = [['nems_lbhb.rdt.preprocessing.split_est_val', {}]]
    return xfspec


def rdtfmt(key):

    xfspec = [['nems_lbhb.rdt.xforms.format_keywordstring', {}]]
    return xfspec

def rdtshf(key):
    ops = key.split(".")
    shuff_streams=False
    shuff_rep=False
    for op in ops:
        if op=="str":
            shuff_streams=True
        if op=="rep":
            shuff_rep=True

    xfspec = [['nems_lbhb.rdt.preprocessing.rdt_shuffle',
               {'shuff_rep': shuff_rep, 'shuff_streams': shuff_streams}]]
    return xfspec
