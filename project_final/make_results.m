function make_results()
cfg=ti.config(); r=readtable(fullfile(cfg.resultsDir,'matlab_label_budget_runs.csv'),'TextType','string');
s=readtable(fullfile(cfg.resultsDir,'matlab_segment_predictions.csv'),'TextType','string');
reps=["matlab_classical","frozen_mert"]; summary=table(); paired=table();
for a=1:2
    for b=1:4
        q=r(r.representation==reps(a)&r.budget==b,:);
        summary=[summary;table(reps(a),b,height(q),mean(q.segment_macro_f1),std(q.segment_macro_f1), ...
            mean(q.recording_macro_f1),std(q.recording_macro_f1),mean(q.segment_accuracy), ...
            'VariableNames',{'representation','budget','runs','segment_f1_mean','segment_f1_sd', ...
            'recording_f1_mean','recording_f1_sd','segment_accuracy_mean'})]; %#ok<AGROW>
    end
end
for b=1:4
    a=sortrows(r(r.representation==reps(1)&r.budget==b,:),{'fold','training_recordings'});
    z=sortrows(r(r.representation==reps(2)&r.budget==b,:),{'fold','training_recordings'});
    assert(isequal(a.fold,z.fold)&&isequal(a.training_recordings,z.training_recordings),'Unpaired runs.');
    delta=z.segment_macro_f1-a.segment_macro_f1;
    paired=[paired;table(b,numel(delta),mean(delta),sum(delta>1e-10),sum(abs(delta)<=1e-10),sum(delta< -1e-10), ...
        'VariableNames',{'budget','runs','mean_mert_minus_classical','wins','ties','losses'})]; %#ok<AGROW>
end
writetable(summary,fullfile(cfg.resultsDir,'matlab_summary.csv'));
writetable(paired,fullfile(cfg.resultsDir,'matlab_paired_comparisons.csv'));
recordings=readtable(fullfile(cfg.resultsDir,'matlab_recording_predictions.csv'),'TextType','string');
sensitivity=table();
for a=1:2
    q=s(s.representation==reps(a),:); ids=unique(q.recording_id); perSource=zeros(numel(ids),1);
    for j=1:numel(ids)
        use=q.recording_id==ids(j); perSource(j)=mean(q.prediction(use)==q.instrument_id(use));
    end
    z=recordings(recordings.representation==reps(a),:);
    score=ti.metrics(z.instrument_id,z.majority_prediction,cfg.labels);
    sensitivity=[sensitivity;table(reps(a),mean(perSource),score.accuracy,score.macro_f1, ...
        'VariableNames',{'representation','source_equal_segment_accuracy', ...
        'majority_recording_accuracy','majority_recording_macro_f1'})]; %#ok<AGROW>
end
writetable(sensitivity,fullfile(cfg.resultsDir,'matlab_source_sensitivity.csv'));
f=figure('Color','w','Position',[100 100 1100 420]);
for panel=1:2
    subplot(1,2,panel); hold on;
    for a=1:2
        q=summary(summary.representation==reps(a),:);
        if panel==1, mu=q.segment_f1_mean; sd=q.segment_f1_sd;
        else, mu=q.recording_f1_mean; sd=q.recording_f1_sd; end
        errorbar(q.budget,mu,sd,'-o','LineWidth',1.5,'DisplayName',reps(a));
    end
    xticks(1:4); xlim([.8 4.2]); grid on; ylabel('Macro-F1');
    xlabel('Labelled source recordings per instrument');
    if panel==1, title('Five-second segment'); else, title('Original recording'); end
    legend('Interpreter','none','Location','southeast');
end
exportgraphics(f,fullfile(cfg.figuresDir,'matlab_learning_curves.png'),'Resolution',200); close(f);
for a=1:2
    q=s(s.representation==reps(a),:); score=ti.metrics(q.instrument_id,q.prediction,cfg.labels);
    C=score.confusion;
    f=figure('Color','w','Position',[100 100 900 750]); imagesc(C); colorbar; axis square;
    xticks(1:11); yticks(1:11); xticklabels(cfg.names); yticklabels(cfg.names); xtickangle(45);
    xlabel('Predicted'); ylabel('True'); title(reps(a)+" full-budget held-out clips",'Interpreter','none');
    for i=1:11
        for j=1:11
            if C(i,j)>0, text(j,i,num2str(C(i,j)),'HorizontalAlignment','center','Color',[.8 .1 .1]); end
        end
    end
    exportgraphics(f,fullfile(cfg.figuresDir,char(reps(a)+"_confusion.png")),'Resolution',200); close(f);
    writematrix(C,fullfile(cfg.resultsDir,char(reps(a)+"_confusion.csv")));
end
end
