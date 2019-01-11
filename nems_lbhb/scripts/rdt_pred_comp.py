import os
import matplotlib.pyplot as plt
from nems import xforms
import nems_db.xform_wrappers as nw
from nems.gui.recording_browser import browse_recording, browse_context
import nems.db as nd
import nems.modelspec as ms
from nems_db.params import fitted_params_per_batch, fitted_params_per_cell, get_batch_modelspecs
import pandas as pd
import numpy as np
from nems_lbhb.stateplots import beta_comp

font_size=8
params = {'legend.fontsize': font_size-2,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

outpath='/auto/users/svd/docs/current/RDT/nems/'

keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1'

loaders = ['rdtld-rdtshf.rep.str-rdtsev-rdtfmt',
           'rdtld-rdtshf.rep-rdtsev-rdtfmt',
           'rdtld-rdtshf.str-rdtsev-rdtfmt',
           'rdtld-rdtshf-rdtsev-rdtfmt']
label0 = ['{}_RS','{}_R','{}_S','{}']
sxticks = ['rep+str', 'rep', 'str', 'noshuff']
modelnames = [l + "_" + keywordstring + "_init-basic" for l in loaders]

batches = [269, 273]
batstring = ['A1','PEG']
batches = [269]
batstring = ['A1']

modelname = modelnames[-1]

#fitted_params_per_cell(cellids, batch, modelname, multi='mean', meta=['r_test', 'r_fit', 'se_test'])
mod_key='id'
multi='mean'
meta=['r_test', 'r_fit', 'se_test']
stats_keys=['mean', 'std', 'sem', 'max', 'min']

for batch, bs in zip(batches, batstring):
    modelspecs = get_batch_modelspecs(batch, modelname, multi=multi, limit=None)
    modelspecs_shf = get_batch_modelspecs(batch, modelnames[2], multi=multi, limit=None)
    stats = ms.summary_stats(modelspecs, mod_key=mod_key,
                             meta_include=meta, stats_keys=stats_keys)
    index = list(stats.keys())
    columns = [m[0].get('meta').get('cellid') for m in modelspecs]

    midx=0
    fields = ['bg_gain','fg_gain']
    b = np.array([])
    f = np.array([])
    c = np.array([])
    cid = []
    r_test = np.array([])
    se_test = np.array([])
    r_test_S = np.array([])
    se_test_S = np.array([])
    for i, m in enumerate(modelspecs):
        r=m.meta['r_test'][0]
        se=m.meta['se_test'][0]
        if r > se*2:
            b = np.append(b, m.phi[midx]['bg_gain'][1:])
            f = np.append(f, m.phi[midx]['fg_gain'][1:])
            s = np.ones(m.phi[midx]['fg_gain'][1:].shape)
            c = np.append(c, s * i)
            cid.extend([m['meta']['cellid']]*len(s))
            r_test = np.append(r_test, s * r)
            se_test = np.append(se_test, s * se)
            r_test_S = np.append(r_test_S, s * modelspecs_shf[i].meta['r_test'][0])
            se_test_S = np.append(se_test_S, s * modelspecs_shf[i].meta['se_test'][0])

    si = (r_test-r_test_S) > (se_test + se_test_S)

    def _rdt_info(i):
        print("{}: f={:.3} b={:.3}".format(cid[i],f[i],b[i]))
        cellid = cid[i]
        xfspec, ctx = nw.load_model_baphy_xform(cellid, batch=batch,
                                                modelname=modelname)
        ctx['modelspec'].quickplot(rec=ctx['val'])

    #plt.figure()
    #ax=plt.subplot(1,1,1)
    #beta_comp(b, f, n1='bg', n2='fg', hist_range=[-0.5, 0.5], click_fun=_rdt_info)
    fig = beta_comp(b, f, n1='bg', n2='fg', hist_range=[-0.75, 0.75], click_fun=_rdt_info,
                    highlight=si, title=bs)

    #fig.savefig(outpath+'gain_comp_'+keywordstring+'_'+bs+'.png')

raise ValueError("Done")


fig = plt.figure(figsize=(10,5))
ax_mean = plt.subplot(1,3,1)
slegend = []

for b, batch in enumerate(batches):
    d=nd.batch_comp(batch=batch, modelnames=modelnames, stat='r_test')
    d.columns = [l0.format('r_test') for l0 in label0]
    dse=nd.batch_comp(batch=batch, modelnames=modelnames, stat='se_test')
    dse.columns = [l0.format('se_test') for l0 in label0]

    r = pd.concat([d,dse], sort=True, axis=1)

    r['sig'] = (((r['r_test']) > (2 * r['se_test'])) | ((r['r_test_RS']) > (2 * r['se_test_RS']))
                & np.isfinite(r['r_test']) & np.isfinite(r['r_test_RS']))
    r['sigdiff'] = (((r['r_test'] - r['r_test_RS']) > (r['se_test'] + r['se_test_RS'])) &
                    r['sig'])
    r['ns'] = ~r['sigdiff'] & r['sig']

    ax_mean.plot(r.loc[r['sig'],d.columns].mean().values, label=batstring[b])
    ax_mean.plot(r.loc[r['sig'],d.columns].median().values,ls='--')

    slegend.append('{} (n={}/{})'.format(batstring[b], r['sig'].sum(), len(r['sig'])))
    print(slegend[-1])
    print(r[r['sig']].mean())

    histbins = np.linspace(-0.1, 0.1, 21)

    ax = plt.subplot(2,3,2+3*b)
    h0,x0=np.histogram(r.loc[r['ns'],'r_test'] - r.loc[r['ns'], 'r_test_S'], bins=histbins)
    h,x=np.histogram(r.loc[r['sigdiff'],'r_test'] - r.loc[r['sigdiff'], 'r_test_S'], bins=histbins)
    d=(x0[1]-x0[0])/2
    plt.bar(x0[:-1]+d, h0, width=d*1.8)
    plt.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    plt.plot([0, 0], ylim, 'k--')
    plt.xlabel('FG/BG stream improvement')
    plt.ylabel('{} units'.format(batstring[b]))
    if b==0:
        plt.title('{}'.format(keywordstring))

    ax = plt.subplot(2,3,3+3*b)
    h0,x0=np.histogram(r.loc[r['ns'],'r_test_S'] - r.loc[r['ns'], 'r_test_RS'], bins=histbins)
    h,x=np.histogram(r.loc[r['sigdiff'],'r_test_S'] - r.loc[r['sigdiff'], 'r_test_RS'], bins=histbins)
    d=(x0[1]-x0[0])/2
    plt.bar(x0[:-1]+d, h0, width=d*1.8)
    plt.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    plt.plot([0, 0], ylim, 'k--')
    plt.xlabel('Rep/no-rep improvement')

ax_mean.legend()
ax_mean.set_xticks(np.arange(0,len(sxticks)))
ax_mean.set_xticklabels(sxticks)
ax_mean.set_ylabel('mean pred corr.')
#plt.suptitle('{}'.format(keywordstring))
plt.tight_layout()

fig.savefig(outpath+'pred_comp_'+keywordstring+'.png')
