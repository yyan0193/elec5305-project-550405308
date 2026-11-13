function extract_classical()
cfg=ti.config(); m=ti.manifest(cfg); X=zeros(height(m),64); timer=tic;
for k=1:height(m)
    y=ti.audio(cfg,m(k,:)); [X(k,:),feature_names]=ti.features(y,cfg);
    if mod(k,50)==0, fprintf('MATLAB classical: %d / %d\n',k,height(m)); end
end
elapsed=toc(timer); segment_ids=m.segment_id;
metadata=struct('representation','MATLAB classical timbre','dimension',64, ...
    'sample_rate',cfg.sampleRate,'fft_length',cfg.fftLength,'hop_length',cfg.hopLength, ...
    'centred_frames',false,'std_ddof',0,'mfcc_implementation','MATLAB Audio Toolbox mfcc', ...
    'extraction_seconds',elapsed,'includes_audio_io',true,'matlab_version',version);
save(fullfile(cfg.cacheDir,'classical_features.mat'),'X','segment_ids','feature_names','metadata');
fprintf('Saved genuine MATLAB features: %d x 64, %.2f s.\n',height(m),elapsed);
end
