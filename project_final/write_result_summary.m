function write_result_summary()
% Generates factual text ONLY after reading a completed MATLAB run.
cfg=ti.config(); s=readtable(fullfile(cfg.resultsDir,'matlab_summary.csv'),'TextType','string');
p=readtable(fullfile(cfg.resultsDir,'matlab_paired_comparisons.csv'));
path=fullfile(cfg.resultsDir,'MATLAB_RESULTS_SUMMARY.md'); fid=fopen(path,'w');
assert(fid>=0,'Cannot open summary.'); cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'# MATLAB downstream results\n\nGenerated: %s. MATLAB: %s.\n\n',datestr(now,30),version);
fprintf(fid,'MERT embeddings are extracted in Python; classical features, fitting, evaluation and plots are MATLAB.\n\n');
fprintf(fid,'Classifier: identical one-vs-rest L2 logistic models for both representations; this differs from the Stage 2 multinomial classifier. Positive binary scores are row-normalised for aggregation, not calibrated confidence.\n\n');
fprintf(fid,'|Sources/class|Classical segment F1|MERT segment F1|Classical recording F1|MERT recording F1|\n|---:|---:|---:|---:|---:|\n');
for b=1:4
    a=s(s.representation=="matlab_classical"&s.budget==b,:);
    z=s(s.representation=="frozen_mert"&s.budget==b,:);
    fprintf(fid,'|%d|%.4f +/- %.4f|%.4f +/- %.4f|%.4f +/- %.4f|%.4f +/- %.4f|\n', ...
        b,a.segment_f1_mean,a.segment_f1_sd,z.segment_f1_mean,z.segment_f1_sd, ...
        a.recording_f1_mean,a.recording_f1_sd,z.recording_f1_mean,z.recording_f1_sd);
end
fprintf(fid,'\nSD describes overlapping configurations, not independent confidence intervals. No significance claim is made.\n\n');
for b=1:4
    q=p(p.budget==b,:);
    fprintf(fid,'Budget %d: MERT minus classical mean segment macro-F1 = %+.4f; %d wins, %d ties, %d losses.\n\n', ...
        b,q.mean_mert_minus_classical,q.wins,q.ties,q.losses);
end
fprintf(fid,'Label efficiency must be judged from these curves, not assumed from Stage 2. A better mean is not guaranteed savings on future data.\n\n');
fprintf(fid,'Only 55 source recordings; no claim of performer/session independence, cultural-bias causality, verified pretraining nonoverlap, mixture recognition or open-set recognition.\n');
fprintf('Generated %s\n',path);
end
