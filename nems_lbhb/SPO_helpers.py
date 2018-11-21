import pandas as pd

def load(rec, **context):
    eval_conds=['A','B','C','I',['A','B'],['C','I']]
    return {'rec':  add_condition_epochs(rec), 'evaluation_conditions':eval_conds}

def add_condition_epochs(rec):
    df0=rec['resp'].epochs.copy()
    df2=rec['resp'].epochs.copy()
    df0['name']=df0['name'].apply(parse_stim_type)
    df0=df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])
    rec['resp'].epochs=df3
    return rec

def parse_stim_type(stim_name):
    stim_sep = stim_name.split('+')
    if len(stim_sep) == 1:
        stim_type = None
    elif stim_sep[1] == 'null':
        stim_type = 'B'
    elif stim_sep[2] == 'null':
        stim_type = 'A'
    elif stim_sep[1] == stim_sep[2]:
        stim_type = 'C'
    else:
        stim_type = 'I'
    return stim_type