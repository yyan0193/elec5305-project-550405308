function [x,names,frames] = features(y,cfg)
% MATLAB-native acoustic representation. Not claimed numerically identical to librosa.
N=cfg.fftLength; H=cfg.hopLength; fs=cfg.sampleRate;
assert(isvector(y)&&numel(y)>=N,'Audio is shorter than one frame.');
y=double(y(:)); win=hann(N,'periodic');
starts=1:H:(numel(y)-N+1);
idx=(0:N-1)'+starts; grains=y(idx);
S=abs(fft(grains.*win,N,1)); S=S(1:N/2+1,:);
f=(0:N/2)'*fs/N; total=sum(S,1); den=max(total,eps);
centroid=sum(f.*S,1)./den;
bandwidth=sqrt(sum((f-centroid).^2.*S,1)./den);
rolloff=zeros(1,numel(starts));
for j=1:numel(starts)
    if total(j)>eps
        q=find(cumsum(S(:,j))>=.85*total(j),1); rolloff(j)=f(q);
    end
end
contrast=zeros(7,numel(starts)); edges=[0,200,400,800,1600,3200,6400,fs/2];
for b=1:7
    if b==7, band=f>=edges(b)&f<=edges(b+1); else, band=f>=edges(b)&f<edges(b+1); end
    a=sort(S(band,:),1); n=max(1,ceil(.02*size(a,1)));
    contrast(b,:)=20*log10(max(mean(a(end-n+1:end,:),1),eps)./max(mean(a(1:n,:),1),eps));
end
rmsValue=sqrt(mean(grains.^2,1));
zcr=sum((grains(1:end-1,:)>=0)~=(grains(2:end,:)>=0),1)/N;
M=mfcc(y,fs,'Window',win,'OverlapLength',N-H,'FFTLength',N, ...
    'NumCoeffs',cfg.numMFCC,'LogEnergy','ignore');
assert(size(M,1)==numel(starts)&&size(M,2)==20,'Unexpected MATLAB MFCC framing.');
frames=[M';centroid;bandwidth;rolloff;contrast;rmsValue;zcr];
base=[compose("mfcc_%02d",1:20),"spectral_centroid","spectral_bandwidth", ...
    "spectral_rolloff",compose("spectral_contrast_%02d",1:7),"rms","zcr"];
names=reshape([base+"_mean";base+"_std"],1,[]);
x=reshape([mean(frames,2),std(frames,1,2)]',1,[]);
assert(numel(x)==64&&all(isfinite(x)),'Invalid classical representation.');
end
