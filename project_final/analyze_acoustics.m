function analyze_acoustics()
% Select actual new errors, rather than assuming the Stage 2 error pairs persist.
cfg=ti.config(); m=ti.manifest(cfg);
s=readtable(fullfile(cfg.resultsDir,'matlab_segment_predictions.csv'),'TextType','string');
A=sortrows(s(s.representation=="matlab_classical",:),'segment_id');
B=sortrows(s(s.representation=="frozen_mert",:),'segment_id');
assert(isequal(A.segment_id,B.segment_id),'Unmatched predictions.');
assert(isequal(A.instrument_id,B.instrument_id),'Truth mismatch.');
cases={find(A.prediction~=A.instrument_id&B.prediction==B.instrument_id,1), ...
    find(A.prediction==A.instrument_id&B.prediction~=B.instrument_id,1), ...
    find(A.prediction~=A.instrument_id&B.prediction~=B.instrument_id,1)};
titles=["MERT corrects classical","Classical corrects MERT","Both remain incorrect"];
caseTable=table();
for k=1:3
    if isempty(cases{k}), fprintf('No observed example: %s\n',titles(k)); continue; end
    j=cases{k}; row=m(m.segment_id==A.segment_id(j),:); y=ti.audio(cfg,row);
    [x,names,frames]=ti.features(y,cfg);
    f=figure('Color','w','Position',[100 100 1000 700]);
    subplot(3,1,1); plot((0:numel(y)-1)/cfg.sampleRate,y); xlabel('Time (s)'); ylabel('Amplitude');
    title(titles(k)+": "+row.segment_id+"; true "+row.instrument,'Interpreter','none');
    subplot(3,1,2); [z,freq,t]=spectrogram(y,hann(cfg.fftLength,'periodic'), ...
        cfg.fftLength-cfg.hopLength,cfg.fftLength,cfg.sampleRate);
    d=20*log10(max(abs(z),eps)/max(max(abs(z(:))),eps));
    imagesc(t,freq/1000,d); axis xy; ylim([0 8]); caxis([-70 0]); colorbar;
    ylabel('Frequency (kHz)'); xlabel('Time (s)'); title('Peak-normalised STFT magnitude (dB)');
    subplot(3,1,3); imagesc((0:size(frames,2)-1)*cfg.hopLength/cfg.sampleRate,2:20,frames(2:20,:));
    axis xy; colorbar; ylabel('MFCC index'); xlabel('Frame start (s)'); title('MFCCs 2-20');
    exportgraphics(f,fullfile(cfg.figuresDir,sprintf('acoustic_case_%d.png',k)),'Resolution',200); close(f);
    q=table(titles(k),row.segment_id,row.instrument,A.prediction(j),B.prediction(j), ...
        x(names=="spectral_centroid_mean"),x(names=="spectral_bandwidth_mean"), ...
        x(names=="spectral_rolloff_mean"),'VariableNames',{'case_type','segment_id','true_instrument', ...
        'classical_prediction','mert_prediction','centroid_hz','bandwidth_hz','rolloff_hz'});
    caseTable=[caseTable;q]; %#ok<AGROW>
end
if ~isempty(caseTable), writetable(caseTable,fullfile(cfg.resultsDir,'matlab_acoustic_cases.csv')); end
% Independent source averages provide acoustic context, not causal model explanations.
c=ti.cache(fullfile(cfg.cacheDir,'classical_features.mat'),m,64);
names=string(c.feature_names); names=strip(names(:));
idx=[find(names=="spectral_centroid_mean"),find(names=="spectral_bandwidth_mean")];
ids=unique(m.recording_id,'stable'); source=table();
for j=1:numel(ids)
    use=m.recording_id==ids(j); instrument=m.instrument(find(use,1)); v=mean(c.X(use,idx),1);
    source=[source;table(ids(j),instrument,v(1),v(2),'VariableNames', ...
        {'recording_id','instrument','centroid_hz','bandwidth_hz'})]; %#ok<AGROW>
end
writetable(source,fullfile(cfg.resultsDir,'matlab_source_acoustics.csv'));
f=figure('Color','w','Position',[100 100 1050 500]);
for j=1:2
    subplot(1,2,j); hold on;
    for c=1:11
        use=source.instrument==cfg.names(c);
        if j==1, v=source.centroid_hz(use); else, v=source.bandwidth_hz(use); end
        scatter(v,c*ones(size(v)),35,'filled');
    end
    yticks(1:11); yticklabels(cfg.names); grid on; xlabel('Mean per original recording (Hz)');
    if j==1, title('Spectral centroid'); else, title('Spectral bandwidth'); end
end
exportgraphics(f,fullfile(cfg.figuresDir,'matlab_source_acoustics.png'),'Resolution',200); close(f);
end
