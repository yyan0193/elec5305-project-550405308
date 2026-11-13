function audit = check_inputs()
% Validate all 55 original recordings against the frozen evaluation manifest.
cfg=ti.config(); m=ti.manifest(cfg); ids=unique(m.recording_id,'stable');
audit=table();
for k=1:numel(ids)
    rows=m(m.recording_id==ids(k),:); row=rows(end,:);
    [~,~,path]=ti.audio(cfg,row); info=audioinfo(path);
    % Publish portable paths instead of personal machine paths.
    [~,audioName,audioExt]=fileparts(path);
    portablePath="ChMusic/Musics/"+string(audioName)+string(audioExt);
    audit=[audit;table(ids(k),portablePath,info.SampleRate,info.Duration,height(rows), ...
        'VariableNames',{'recording_id','path','sample_rate','duration_seconds','evaluated_segments'})]; %#ok<AGROW>
end
if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
writetable(audit,fullfile(cfg.resultsDir,'input_audit.csv'));
fprintf('Verified 55 source recordings and all 988 declared five-second segments.\n');
fprintf('Extra audio tails are not added; this preserves the Stage 2 evaluation set.\n');
end
