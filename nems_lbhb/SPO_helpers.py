import pandas as pd
import nems.plots.api as nplt
import nems.preprocessing as preproc

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

def plot_all_vals_(modelspecs, val, figures=None, IsReload=False, **context):
    if figures is None:
        figures = []
    if not IsReload:
        fig = plot_all_vals(val[0],modelspecs[0])
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))

    return {'figures': figures}

def add_coherence_as_state(rec,**context):
    rec['Coherence'] = rec['resp'].epoch_to_signal('C')
    inc = rec['resp'].epoch_to_signal('I').as_continuous()
    C = rec['Coherence'].as_continuous() + 2*inc
    rec['Coherence']._data=C
    rec = preproc.concatenate_state_channel(rec, rec['Coherence'], state_signal_name='state')
    rec = preproc.concatenate_state_channel(rec, rec['Coherence'], state_signal_name='state_raw')
    rec.signals['state'].chans=['baseline','Coherence']
    rec.signals['state_raw'].chans=['baseline','Coherence']
    return {'rec': rec}

def plot_all_vals(val,modelspec,signames=['resp','pred']):
    from nems.plots.timeseries import timeseries_from_epoch
    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler
    if val[signames[0]].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    extracted = val[signames[0]].extract_epoch(epochname)
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)
    
    epochs=val[signames[0]].epochs
    epochs=epochs[epochs['name'] ==  epochname].iloc[occurrences]
    st_mask=val[signames[0]].epochs['name'].str.contains('ST')
    inds=[]
    for index, row in epochs.iterrows():
        matchi = (val[signames[0]].epochs['start'] == row['start']) & (val[signames[0]].epochs['end'] == row['end'])
        matchi = matchi & st_mask
        inds.append(np.where(matchi)[0][0])
         
    names=val[signames[0]].epochs['name'].iloc[inds].tolist()
    
    A=[];B=[];
    for name in names:
        nm=name.split('+')
        A.append(nm[1])
        B.append(nm[2])

    plot_order=['STIM_T+si464+null', 'STIM_T+null+si464', 'STIM_T+si464+si464',
                'STIM_T+si516+null', 'STIM_T+null+si516', 'STIM_T+si516+si516',
                'STIM_T+si464+si516', 'STIM_T+si516+si464']
    plot_order.reverse()
    order = np.array([names.index(nm) for nm in plot_order])
    names_short=[n.replace('STIM_T+','').replace('si464','1').replace('si516','2').replace('null','_') for n in names]
#    names2=sorted(names,key=lambda x: plot_order.index(x))
    
 #   idmap = dict((id,pos) for pos,id in enumerate(plot_order))
    
    sigs = [val[s] for s in signames]
    title = ''
    nplt=len(occurrences)
    gs_kw = dict(hspace=0,left=0.04,right=.99)
    fig, ax = plt.subplots(nrows=nplt, ncols=1, figsize=(10, 15),sharey=True,gridspec_kw=gs_kw)
    [axi.set_prop_cycle(cycler('color', ['k','#1f77b4', 'r']) + cycler('linestyle', ['-', '-', '--'])) for axi in ax]
    allsigs =np.hstack([s.as_continuous()[-1,:] for s in sigs])    
    yl=[np.nanmin(allsigs), np.nanmax(allsigs)]
    prestimtime=val['stim'].epochs.loc[0].end
    for i in range(nplt):
        timeseries_from_epoch(sigs, epochname, title=title,
                         occurrences=occurrences[order[i]],ax=ax[i],channels=[0,0,1])
        if names_short[order[i]] in ['1+_','2+_']:
            #timeseries_from_epoch([val['stim']], epochname, title=title,
            #             occurrences=occurrences[order[i]],ax=ax[i])
            ep=val['stim'].extract_epoch(names[order[i]]).squeeze()
            ep=80+20*np.log10(ep.T)
            ep=ep/ep.max()*yl[1]
            time_vector = np.arange(0, len(ep)) / val['stim'].fs
            ax[i].plot(time_vector-prestimtime,ep,'--',color='#ff7f0e')
        ax[i].set_ylabel(names_short[order[i]],rotation=0,horizontalalignment='right',verticalalignment='bottom')

    if modelspec is not None:
        ax[0].set_title('{}: {}'.format(modelspec[0]['meta']['cellid'],modelspec[0]['meta']['modelname']))
    [axi.get_xaxis().set_visible(False) for axi in ax[:-1]]
    [axi.get_yaxis().set_ticks([]) for axi in ax]  
    [axi.legend().set_visible(False) for axi in ax[:-1]]
    [axi.set_xlim([.8-1, 4.5-1]) for axi in ax]
    
    [axi.set_ylim(yl) for axi in ax]
    ax[nplt-1].legend(signames+['Stim'])
    return fig