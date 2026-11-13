function run_mfcc_knn()
% Optional MATLAB adaptation of the published benchmark, NOT an exact librosa reproduction.
% This is deliberately separate from run_all; the Stage 2 reproduction is preserved.
cfg=ti.config(); m=ti.manifest(cfg); sr=22050; X=cell(height(m),1);
timer=tic;
for k=1:height(m)
    y=ti.audio(cfg,m(k,:),sr);
    X{k}=mfcc(y,sr,'Window',hann(2048,'periodic'),'OverlapLength',1536, ...
        'FFTLength',2048,'NumCoeffs',20,'LogEnergy','ignore');
end
featureSeconds=toc(timer); train=m.fold~=5; rows=find(train); labels=cell(numel(rows),1);
for j=1:numel(rows), labels{j}=repmat(m.instrument_id(rows(j)),size(X{rows(j)},1),1); end
t=tic; model=fitcknn(vertcat(X{train}),vertcat(labels{:}),'NumNeighbors',15, ...
    'Standardize',false,'Distance','euclidean','NSMethod','kdtree','BreakTies','smallest');
fitSeconds=toc(t); test=m(m.fold==5,:); ix=find(m.fold==5); prediction=zeros(height(test),1); t=tic;
for j=1:numel(ix), prediction(j)=mode(predict(model,X{ix(j)})); end
predictSeconds=toc(t); test.prediction=prediction; s=ti.metrics(test.instrument_id,prediction,cfg.labels);
writetable(test,fullfile(cfg.resultsDir,'matlab_knn_segment_predictions.csv'));
summary=table(s.accuracy,s.macro_f1,featureSeconds,fitSeconds,predictSeconds, ...
    'VariableNames',{'segment_accuracy','segment_macro_f1','feature_seconds','fit_seconds','predict_seconds'});
writetable(summary,fullfile(cfg.resultsDir,'matlab_knn_metrics.csv'));
fprintf('MATLAB-adapted MFCC-KNN accuracy %.4f; NOT assumed to equal the published 94.15%%.\n',s.accuracy);
end
