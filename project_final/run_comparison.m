function run_comparison()
cfg=ti.config(); rng(cfg.seed); m=ti.manifest(cfg);
paths={fullfile(cfg.cacheDir,'classical_features.mat'),fullfile(cfg.cacheDir,'mert_embeddings.mat')};
reps=["matlab_classical","frozen_mert"]; dims=[64,768]; runs=table(); segments=table(); recordings=table();
allModels=cell(2,5); costs=table();
for a=1:2
    s=ti.cache(paths{a},m,dims(a));
    assert(isfield(s.metadata,'extraction_seconds'),'Missing extraction timing.');
    costs=[costs;table(reps(a),dims(a),double(s.metadata.extraction_seconds), ...
        'VariableNames',{'representation','dimension','extraction_seconds'})]; %#ok<AGROW>
    for fold=1:5
        remaining=setdiff(1:5,fold); test=m.fold==fold;
        for budget=1:4
            sets=nchoosek(remaining,budget);
            for choice=1:size(sets,1)
                train=ismember(m.fold,sets(choice,:));
                assert(isempty(intersect(m.recording_id(train),m.recording_id(test))),'Leakage.');
                t=tic; model=ti.train(s.X(train,:),m.instrument_id(train),m.recording_id(train),cfg); fitSeconds=toc(t);
                t=tic; [pred,p]=ti.predict(model,s.X(test,:)); predictSeconds=toc(t);
                segScore=ti.metrics(m.instrument_id(test),pred,cfg.labels);
                [rec,~]=ti.aggregate(m(test,:),p,cfg.labels);
                recScore=ti.metrics(rec.instrument_id,rec.prediction,cfg.labels);
                trainNumbers=strjoin(string(sets(choice,:)),"-");
                run=table(reps(a),fold,budget,trainNumbers,segScore.accuracy,segScore.macro_f1, ...
                    recScore.accuracy,recScore.macro_f1,fitSeconds,predictSeconds, ...
                    'VariableNames',{'representation','fold','budget','training_recordings', ...
                    'segment_accuracy','segment_macro_f1','recording_accuracy','recording_macro_f1', ...
                    'fit_seconds','predict_seconds'});
                runs=[runs;run]; %#ok<AGROW>
                if budget==4
                    q=m(test,{'segment_id','recording_id','instrument_id','instrument','fold'});
                    q.prediction=pred; q.representation=repmat(reps(a),height(q),1);
                    segments=[segments;q]; %#ok<AGROW>
                    rec.fold=repmat(fold,height(rec),1); rec.representation=repmat(reps(a),height(rec),1);
                    recordings=[recordings;rec]; allModels{a,fold}=model; %#ok<AGROW>
                end
                fprintf('%s fold %d budget %d subset %s\n',reps(a),fold,budget,trainNumbers);
            end
        end
    end
end
assert(height(runs)==150&&height(segments)==1976&&height(recordings)==110,'Incomplete experiment.');
writetable(runs,fullfile(cfg.resultsDir,'matlab_label_budget_runs.csv'));
writetable(segments,fullfile(cfg.resultsDir,'matlab_segment_predictions.csv'));
writetable(recordings,fullfile(cfg.resultsDir,'matlab_recording_predictions.csv'));
costs.seconds_per_segment=costs.extraction_seconds/height(m);
writetable(costs,fullfile(cfg.resultsDir,'hybrid_extraction_cost.csv'));
% Runtime paths belong in local_settings.mat, not shared classifier files.
save(fullfile(cfg.modelsDir,'heldout_models.mat'),'allModels','reps','-v7.3');
fprintf('150 MATLAB fits complete. Old Python predictions were not used.\n');
end
