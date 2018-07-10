
baphy_set_path
close all

if 0,
    cellid='por074a-d1';
    channum=1;
    unit=3;
    stimfile='/auto/data/daq/Portabello/por074/por074a07_p_SPN';
    respfile='/auto/data/daq/Portabello/por074/sorted/por074a07_p_SPN.spk.mat';
    
    ftcfile='/auto/data/daq/Portabello/por074/sorted/por074a01_p_FTC.spk.mat';

elseif 0,
    cellid='por074b-d2';
    channum=4;
    unit=2;
    stimfile='/auto/data/daq/Portabello/por074/por074b16_p_SPN';
    respfile='/auto/data/daq/Portabello/por074/sorted/por074b16_p_SPN.spk.mat';
    
elseif 1,
    cellid='por077a-c1';
    channum=3;
    unit=1;
    stimfile='/auto/data/daq/Portabello/por077/por077a16_p_SPN';
    respfile='/auto/data/daq/Portabello/por077/sorted/por077a16_p_SPN.spk.mat';
    
elseif 0,
    cellid='por025a-b1';
    channum=2;
    unit=1;
    stimfile='/auto/data/daq/Portabello/por025/por025a06_p_SPN';
    respfile='/auto/data/daq/Portabello/por025/sorted/por025a06_p_SPN.spk.mat';
else
    cellid='por027b-c1';
    channum=3;
    unit=1;
    stimfile='/auto/data/daq/Portabello/por027/por027b06_p_SPN';
    respfile='/auto/data/daq/Portabello/por027/sorted/por027b06_p_SPN.spk.mat';
end

%ftc_tuning(ftcfile,channum,unit)
dbtuningcheck(cellid);
set(1,'PaperPosition',[0.25 4 8 3]);
figure(1);
colormap(jet);

options=[];
options.rasterfs=100;
options.channel=channum;
options.unit=unit;
options.includeprestim=1;

LoadMFile(stimfile)
[stim,stimparams]=loadstimfrombaphy(stimfile,[],[],'ozgf',options.rasterfs,30,0,options.includeprestim);

env=loadstimfrombaphy(stimfile,[],[],'envelope',options.rasterfs,0,0,options.includeprestim);

r=loadspikeraster(respfile,options);


stimidx=1;
tt=(1:size(stim,2))./options.rasterfs;

colors={[237 45 38]./256, [114 201 241]./256};

figure(2);
clf;
subplot(4,1,1);
imagesc(tt,1:size(stim,1),stim(:,:,stimidx,1));
axis xy;

subplot(4,1,2);
if size(stim,4)>1,
    imagesc(tt,1:size(stim,1),stim(:,:,stimidx,2));
    axis xy;
else
    plot(tt,env(1,:,stimidx)','Color',colors{1});
    hold on
    plot(tt,env(2,:,stimidx)','Color',colors{2});
    hold off
    legend('stream 1','stream 2');
end

subplot(4,1,3);
xx=r(:,:,stimidx)';
xx(xx>2)=2;
xx=xx./2;
im=repmat(1-xx,[1 1 3]);
imagesc(tt,1:size(im,1),im);

subplot(4,1,4);
rr=gsmooth(r(:,:,stimidx),[1 0.01]);
plot(tt,mean(rr,2).*options.rasterfs,'k');

fullpage portrait;

ref=exptparams.TrialObject.ReferenceHandle;
s = eval(ref.BaseSound);
s = set(s, 'Subsets', ref.Subsets);
s = set(s, 'PreStimSilence', ref.PreStimSilence);
s = set(s, 'PostStimSilence', ref.PostStimSilence);
s = set(s, 'Duration', ref.Duration);
n=get(s,'Names');

stimname=stimparams.tags{stimidx};
stimstrings=strsep(stimname,'+');

s1=find(strcmp(stimstrings{2},n));
s2=find(strcmp(stimstrings{3},n));
filtfmt = 'ozgf';
fsin = get(s, 'SamplingRate');
fsout=100;
chancount=64;


w1=waveform(s, s1);
shiftrange=(ref.PreStimSilence*fsin)+(1:ref.Duration*fsin);
shiftbins=ref.ShuffledOnsetTimes(stimidx,1)*fsin;
w1(shiftrange)=shift(w1(shiftrange),[shiftbins,1]);
sg1=wav2spectral(w1,filtfmt,fsin,fsout,chancount);

w2=waveform(s, s2);
shiftbins=ref.ShuffledOnsetTimes(stimidx,2)*fsin;
w2(shiftrange)=shift(w2(shiftrange),[shiftbins,1]);
sg2=wav2spectral(w2,filtfmt,fsin,fsout,chancount);
t=(1:size(sg1,1))/fsout - ref.PreStimSilence;

figure(3);
clf;
subplot(4,1,1);
plot(w1, 'Linewidth', 0.5, 'Color', colors{1});
set(gca,'YLim',[-5 5]);

subplot(4,1,2);
imagesc(t,1:size(sg1,2),sqrt(sg1)');
axis xy; 
hold on;
plot(t,env(1,:,stimidx)*300, 'Color', colors{1});
hold off;

subplot(4,1,3);
plot(w2, 'Linewidth', 0.5, 'Color', colors{2});
set(gca,'YLim',[-5 5]);

subplot(4,1,4);
imagesc(t,1:size(sg2,2),sqrt(sg2)');
axis xy; 
hold on;
plot(t,env(2,:,stimidx)*300, 'Color', colors{2});
hold off;

colormap(1-gray);

fullpage portrait;


disp('To print:');
disp(['print -f1 -dpdf ' cellid '_strf.pdf']);
disp(['print -f2 -dpdf ' cellid '_spn_data.pdf']);
disp(['print -f3 -dpdf ' cellid '_spn2_stim.pdf']);




