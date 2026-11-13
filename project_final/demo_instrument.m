function output = demo_instrument(segmentID,modeName)
% DEMO_INSTRUMENT Genuine MATLAB prediction on a manifest's held-out source.
% demo_instrument('dizi_r5_s001','live') computes BOTH representations anew.
% demo_instrument('dizi_r5_s001','cached') uses saved MERT FEATURES (not labels).
if nargin<1, segmentID='dizi_r5_s001'; end
if nargin<2, modeName='live'; end
assert(ismember(string(modeName),["live","cached"]),'Mode must be live or cached.');
cfg=ti.config(); m=ti.manifest(cfg); row=m(m.segment_id==string(segmentID),:);
assert(height(row)==1,'Choose a known segment ID from metadata/data_manifest.csv.');
saved=load(fullfile(cfg.modelsDir,'heldout_models.mat'),'allModels'); fold=row.fold;
[y,fs,path]=ti.audio(cfg,row); t=tic; [x,names]=ti.features(y,cfg); classicalSeconds=toc(t);
[a,pa]=ti.predict(saved.allModels{1,fold},x);
if strcmp(modeName,'cached')
    cache=ti.cache(fullfile(cfg.cacheDir,'mert_embeddings.mat'),m,768);
    v=cache.X(m.segment_id==row.segment_id,:); mertSeconds=NaN;
    explanation='Cached MERT embedding; MATLAB prediction computed now. Encoder NOT run.';
else
    out=[tempname,'.json']; cleaner=onCleanup(@()deleteIfPresent(out)); %#ok<NASGU>
    parts={cfg.python,fullfile(cfg.root,'python','mert_bridge.py'),'single', ...
        '--audio',path,'--start',num2str(row.start_seconds),'--output',out,'--device','cpu'};
    assert(~any(cellfun(@(p)contains(p,'"')||contains(p,newline),parts)),'Invalid path.');
    cmd=strjoin(cellfun(@(p)['"',p,'"'],parts,'UniformOutput',false),' ');
    [status,log]=system(cmd,'-echo'); assert(status==0,'Python MERT encoder failed: %s',log);
    z=jsondecode(fileread(out)); v=reshape(z.embedding,1,[]);
    assert(numel(v)==768,'Invalid embedding.'); mertSeconds=z.extraction_seconds;
    explanation=sprintf('Fresh Python MERT encoder; loading %.2f s, audio + embedding %.2f s.',z.loading_seconds,mertSeconds);
end
[b,pb]=ti.predict(saved.allModels{2,fold},v);
fprintf('\nInput: %s, %.1f-%.1f s; true instrument: %s\n',row.segment_id,row.start_seconds,row.start_seconds+5,row.instrument);
fprintf('Both MATLAB classifiers exclude source fold %d from training.\n',fold);
fprintf('Classical: %s (score %.3f); MERT: %s (score %.3f).\n',cfg.names(a),max(pa),cfg.names(b),max(pb));
fprintf('%s\nScores are not calibrated confidence. Closed-set solo-instrument task.\n',explanation);
keys=["spectral_centroid_mean","spectral_bandwidth_mean","spectral_rolloff_mean","rms_mean","zcr_mean"];
for j=1:numel(keys), fprintf('%s: %.4g\n',keys(j),x(names==keys(j))); end
% Keep the title visible across different screen resolutions.
f=figure('Color','w','Units','normalized','Position',[0.08 0.08 0.84 0.80]);
subplot(3,1,1); plot((0:numel(y)-1)/fs,y); xlabel('Time (s)'); ylabel('Amplitude');
title("True: "+row.instrument+" | Classical: "+cfg.names(a)+" | MERT: "+cfg.names(b));
subplot(3,1,2); [S,F,T]=spectrogram(y,hann(cfg.fftLength,'periodic'), ...
    cfg.fftLength-cfg.hopLength,cfg.fftLength,fs);
imagesc(T,F/1000,20*log10(max(abs(S),eps)/max(max(abs(S(:))),eps))); axis xy;
ylim([0 8]); caxis([-70 0]); colorbar; xlabel('Time (s)'); ylabel('Frequency (kHz)');
subplot(3,1,3); bar([pa;pb]'); xticks(1:11); xticklabels(cfg.names); xtickangle(30);
ylabel('Normalised classifier score'); legend('MATLAB classical','Frozen MERT');
output=struct('segment_id',row.segment_id,'true_instrument',row.instrument, ...
    'classical_prediction',cfg.names(a),'mert_prediction',cfg.names(b),'mode',modeName, ...
    'classical_feature_seconds',classicalSeconds,'mert_feature_seconds',mertSeconds);
fprintf('To play the actual input, use: sound(y,fs) inside this function, or play_segment(''%s'').\n',row.segment_id);
end

function deleteIfPresent(path)
if isfile(path), delete(path); end
end
