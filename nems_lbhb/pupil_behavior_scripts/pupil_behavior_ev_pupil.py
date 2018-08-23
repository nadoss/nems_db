import matplotlib.pyplot as plt
import numpy as np
import os
import io

#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils

#import nems_db.baphy as nb
import nems_db.db as nd
import nems_db.xform_wrappers as nw
import nems.recording as recording
import nems.epoch as ep
import nems_lbhb.stateplots as sp

#params = {'legend.fontsize': 8,
#          'figure.figsize': (8, 6),
#          'axes.labelsize': 8,
#          'axes.titlesize': 8,
#          'xtick.labelsize': 8,
#          'ytick.labelsize': 8,
#          'pdf.fonttype': 42,
#          'ps.fonttype': 42}
#plt.rcParams.update(params)


def ev_pupil(cellid, batch, presilence=0.35):
    modelname = "psth.fs20-st.pup.beh_stategain.3_init.st-basic"

    print('Finding recording for cell/batch {0}/{1}...'.format(cellid, batch))

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # generate xfspec, which defines sequence of events to load data,
    # generate modelspec, fit data, plot results and save
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)
    print(recording_uri)

    rec = recording.load_recording(recording_uri)

    pupil = rec['pupil']
    pshift = np.int(pupil.fs * 0.75)
    #pupil = pupil._modified_copy(np.roll(pupil._data, (0, pshift)))
    d = pupil._data
    pupil = pupil._modified_copy(np.roll(d / np.nanmax(d), (0, pshift)))

    trials = pupil.get_epoch_indices('TRIAL')
    targets = pupil.get_epoch_indices('TARGET')
    pt_blocks = pupil.get_epoch_indices('PURETONE_BEHAVIOR').tolist()
    easy_blocks = pupil.get_epoch_indices('EASY_BEHAVIOR').tolist()
    hard_blocks = pupil.get_epoch_indices('HARD_BEHAVIOR').tolist()
    passive_blocks = pupil.get_epoch_indices('PASSIVE_EXPERIMENT').tolist()
    behavior_blocks = pupil.get_epoch_indices('ACTIVE_EXPERIMENT')

    blocks=[]
    for p in passive_blocks:
        p.append('passive')
        blocks.append(p)
    for p in pt_blocks:
        p.append('puretone')
        blocks.append(p)
    for p in easy_blocks:
        p.append('easy')
        blocks.append(p)
    for p in hard_blocks:
        p.append('hard')
        blocks.append(p)

    blocks.sort()
    #print(blocks)
    trialbins = int(pupil.fs * 6)
    prebins = int(pupil.fs *presilence)

    ev=[]
    ev_prenorm=[]
    label=[]
    beh_lickrate=[]
    beh_lickrate_norm=[]
    for block in blocks:
        k = block[-1]
        label.append(k)
        block_trials = ep.epoch_intersection(trials, np.array([block[0:2]]))
        tcount=block_trials.shape[0]
        for t in range(tcount):
            block_trials[t,1]=block_trials[t,0]+trialbins
            if block_trials[t,1]>pupil.shape[1]:
                block_trials[t,1]=pupil.shape[1]

        tev = pupil.extract_epoch(block_trials)[:,0,:]

        tev0 = np.nanmean(tev[:,:prebins],axis=1)
#        m = tev0 > 0.3 * np.nanmax(tev0)
#        print(block)
#        print("{}-{} mean {} ({}/{} big)".format(
#                np.nanmin(tev0),np.nanmax(tev0),np.nanmean(tev0),
#                np.sum(m), len(tev0)))
#        tev = tev[m, :]

        ev.append(tev)
        ev_prenorm.append(tev - np.mean(tev[:,:prebins],axis=1,keepdims=True))

        beh_lickrate.append((k,np.nanmean(ev[-1],axis=0)))
        beh_lickrate_norm.append((k,np.nanmean(ev_prenorm[-1],axis=0)))

        #print("{}: {} trials, {} bins".format(k,tev.shape[0],tev.shape[1]))

    perf_blocks={'hits': pupil.get_epoch_indices('HIT_TRIAL'),
                 'misses': pupil.get_epoch_indices('MISS_TRIAL'),
                 'fas': pupil.get_epoch_indices('FA_TRIAL')}

    ev=[]
    ev_prenorm=[]
    perf_lickrate=[]
    perf_lickrate_norm=[]
    for k,block in perf_blocks.items():
        block_trials = ep.epoch_intersection(trials, block)
        tcount=block_trials.shape[0]
        for t in range(tcount):
            block_trials[t,1]=block_trials[t,0]+trialbins
            if block_trials[t,1]>pupil.shape[1]:
                block_trials[t,1]=pupil.shape[1]
        t = pupil.extract_epoch(block_trials, allow_empty=True)
        if t.size:
            tev = t[:,0,:]
        else:
            tev = np.ones((1,trialbins)) * np.nan

        ev.append(tev)
        ev_prenorm.append(tev - np.mean(tev[:,:prebins],axis=1,keepdims=True))

        perf_lickrate.append((k,np.nanmean(ev[-1],axis=0)))
        perf_lickrate_norm.append((k,np.nanmean(ev_prenorm[-1],axis=0)))

        #print("{}: {} trials, {} bins".format(k,tev.shape[0],tev.shape[1]))

    return beh_lickrate, beh_lickrate_norm, perf_lickrate, perf_lickrate_norm

def ev_pupil_plot(ev_data_tuple_list, title=None, ax=None, fs=100,
                  prestimsilence=0.35, linecolors=None, fillcolors=None):
    if ax is None:
        fh=plt.figure()
        ax=plt.gca()
    trialbins=len(ev_data_tuple_list[0][1])
    tt=np.arange(trialbins) / fs - prestimsilence

    label=[]
    cc=0
    opt={}
    for k, ev in ev_data_tuple_list:
        label.append(k)
        if len(ev.shape)>1 and ev.shape[1]>1:
            if fillcolors is not None:
                opt = {'facecolor': fillcolors[cc]}
            mm=np.nanmean(ev,axis=1)
            ee=np.nanstd(ev,axis=1)/np.sqrt(ev.shape[1])
            plt.fill_between(tt, mm-ee, mm+ee, **opt)
        else:
            mm = ev
        if linecolors is not None:
            opt = {'color': linecolors[cc]}
        plt.plot(tt, mm, **opt)
        cc += 1
    plt.legend(label, loc='upper left', frameon=False)
    plt.xlabel('trial time (s)')
    plt.ylabel('pupil diameter')
    if title is not None:
        plt.title(title)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


batch = 301
d=nd.get_batch_cells(batch=batch)
cellids=list(d['cellid'])

ucellids=[cellids[0]]
batches=[batch]
for c in cellids[1:]:
    site1=ucellids[-1].split("-")[0]
    site2=c.split("-")[0]
    if site1 != site2:
        ucellids.append(c)
        batches.append(batch)

batch=303
d = nd.get_batch_cells(batch=batch)
cellids = list(d['cellid'])

for c in cellids[1:]:
    site1 = ucellids[-1].split("-")[0]
    site2 = c.split("-")[0]
    if site1 != site2:
        ucellids.append(c)
        batches.append(batch)

cellids = ucellids
print(cellids)
print(batches)

active=[]
passive=[]
active_full=[]
passive_full=[]
hard=[]
easy=[]
puretone=[]
hits=[]
fas=[]
misses=[]
hits_full=[]
fas_full=[]
misses_full=[]
presilence=0.35
fs=20

for cellid,batch in zip(cellids,batches):
    beh_lickrate, beh_lickrate_norm, perf_lickrate, perf_lickrate_norm = \
         ev_pupil(cellid, batch, presilence=presilence)

    cc=0
    pset=[]
    aset=[]
    pfullset=[]
    afullset=[]
    for k, ev in beh_lickrate_norm:
        if k=='passive':
            pset.append(np.reshape(ev,[-1,1]))
            pfullset.append(np.reshape(beh_lickrate[cc][1],[-1,1]))
        else:
            aset.append(np.reshape(ev,[-1,1]))
            afullset.append(np.reshape(beh_lickrate[cc][1],[-1,1]))
            if k=='hard':
                hard.append(np.reshape(ev,[-1,1]))
            elif k=='easy':
                easy.append(np.reshape(ev,[-1,1]))
            else:
                puretone.append(np.reshape(ev,[-1,1]))
        cc+=1
    passive.append(np.mean(np.concatenate(pset,axis=1),axis=1, keepdims=True))
    passive_full.append(np.mean(np.concatenate(pfullset,axis=1),axis=1, keepdims=True))
    active.append(np.mean(np.concatenate(aset,axis=1),axis=1, keepdims=True))
    active_full.append(np.mean(np.concatenate(afullset,axis=1),axis=1, keepdims=True))

    cc=0
    for k, ev in perf_lickrate_norm:
        if k=='hits':
            hits.append(np.reshape(ev,[-1,1]))
            hits_full.append(np.reshape(perf_lickrate[cc][1],[-1,1]))
        elif k=='misses':
            misses.append(np.reshape(ev,[-1,1]))
            misses_full.append(np.reshape(perf_lickrate[cc][1],[-1,1]))
        else:
            fas.append(np.reshape(ev,[-1,1]))
            fas_full.append(np.reshape(perf_lickrate[cc][1],[-1,1]))
        cc+=1

f=plt.figure(figsize=(6,8))
ax=plt.subplot(3,2,1)
#d=[('passive',np.concatenate(passive,axis=1)),
#   ('active',np.concatenate(active,axis=1))]
d=[('passive',np.concatenate(passive_full,axis=1)),
   ('active',np.concatenate(active_full,axis=1))]
ev_pupil_plot(d, title="beh vs. d_pupil", fs=fs, prestimsilence=presilence,
              ax=ax,
              linecolors=[sp.line_colors['passive'],sp.line_colors['active']],
              fillcolors=[sp.fill_colors['passive'],sp.fill_colors['active']])

prebins=int(presilence*fs)
print(prebins)
ev_start=int(presilence*fs)
ev_end=int((presilence+3)*fs)
print("Averaging evoked response over {}-{} sec".format(ev_start/fs-presilence, ev_end/fs-presilence))

passive_d=np.mean(np.concatenate(passive,axis=1)[ev_start:ev_end,:],axis=0)
passive_0=np.mean(np.concatenate(passive_full,axis=1)[:prebins,:],axis=0)

active_d=np.mean(np.concatenate(active,axis=1)[ev_start:ev_end,:],axis=0)
active_0=np.mean(np.concatenate(active_full,axis=1)[:prebins,:],axis=0)
print(passive_d.shape)
print(active_d.shape)

ax=plt.subplot(3,2,2)
for i in range(len(passive_0)):
    plt.plot([passive_0[i],active_0[i]],[passive_d[i],active_d[i]],
             color='darkgray',linewidth=0.5)

h1,=plt.plot(passive_0,passive_d,'.',markersize=10, label='passive',
             mfc=sp.fill_colors['passive'], mec=sp.fill_colors['passive'])
h2,=plt.plot(active_0,active_d,'.',markersize=10, label='active',
             mfc=sp.fill_colors['active'], mec=sp.fill_colors['active'])
plt.xlabel('pre-trial pupil diameter')
plt.ylabel('evoked pupil change')
plt.legend(handles=(h1,h2), frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax=plt.subplot(3,2,3)
d=[('hard',np.concatenate(hard,axis=1)),
   ('easy',np.concatenate(easy,axis=1)),
   ('puretone',np.concatenate(puretone,axis=1))]
lc=[sp.line_colors['hard'],sp.line_colors['easy'],sp.line_colors['puretone']]
fc=[sp.fill_colors['hard'],sp.fill_colors['easy'],sp.fill_colors['puretone']]

ev_pupil_plot(d, title="difficulty vs. d_pupil", fs=fs, prestimsilence=presilence,
              linecolors=lc, fillcolors=fc, ax=ax)


ax=plt.subplot(3,2,5)
#d=[('fas',np.concatenate(fas,axis=1)),
#   ('hits',np.concatenate(hits,axis=1)),
#   ('misses',np.concatenate(misses,axis=1))]
d=[('fas',np.concatenate(fas_full,axis=1)),
   ('hits',np.concatenate(hits_full,axis=1)),
   ('misses',np.concatenate(misses_full,axis=1))]
lc=[sp.line_colors['false_alarm'],sp.line_colors['hit'],sp.line_colors['miss']]
fc=[sp.fill_colors['false_alarm'],sp.fill_colors['hit'],sp.fill_colors['miss']]
ev_pupil_plot(d, title="performance vs. d_pupil", fs=fs, prestimsilence=presilence,
              linecolors=lc, fillcolors=fc, ax=ax)

prebins=int(presilence*fs)
evbins=prebins+int(fs)  # look at first second post trial start

hits_d=np.mean(np.concatenate(hits,axis=1)[ev_start:ev_end,:],axis=0)
hits_0=np.mean(np.concatenate(hits_full,axis=1)[:prebins,:],axis=0)
misses_d=np.mean(np.concatenate(misses,axis=1)[ev_start:ev_end,:],axis=0)
misses_0=np.mean(np.concatenate(misses_full,axis=1)[:prebins,:],axis=0)
fas_d=np.mean(np.concatenate(fas,axis=1)[ev_start:ev_end,:],axis=0)
fas_0=np.mean(np.concatenate(fas_full,axis=1)[:prebins,:],axis=0)

print(hits_d.shape)
print(fas_d.shape)

ax=plt.subplot(3,2,6)
#for i in range(len(passive_0)):
#    plt.plot([passive_0[i],active_0[i]],[passive_d[i],active_d[i]],color='darkgray',linewidth=0.5)

h3,=plt.plot(fas_0, fas_d,'.',markersize=12, label='fas',
             mfc=sp.fill_colors['false_alarm'], mec=sp.fill_colors['false_alarm'])
h2,=plt.plot(misses_0, misses_d,'.',markersize=10, label='misses',
             mfc=sp.fill_colors['miss'], mec=sp.fill_colors['miss'])
h1,=plt.plot(hits_0, hits_d,'.',markersize=10, label='hits',
             mfc=sp.fill_colors['hit'], mec=sp.fill_colors['hit'])
plt.xlabel('pre-trial pupil diameter')
plt.ylabel('evoked pupil change')
plt.legend(handles=(h1,h2,h3), frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()

f.savefig('/auto/users/svd/projects/pupil-behavior/PTD_ev_pupil.pdf')
f.savefig('/auto/users/svd/projects/pupil-behavior/PTD_ev_pupil.png')
