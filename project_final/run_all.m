function run_all()
% RUN_ALL Full genuine MATLAB phase, using Python only for MERT extraction.
cfg=ti.config();
assert(isfolder(cfg.resultsDir),'Run setup_project first.');
diary(fullfile(cfg.resultsDir,'matlab_run_log.txt')); cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
fprintf('MATLAB run started: %s\n%s\n',datestr(now,30),version); ver;
tests=runtests(fullfile(cfg.root,'tests')); disp(tests); assert(all([tests.Passed]),'Tests failed.');
check_inputs(); extract_classical();
prepare_mert();
run_comparison(); make_results(); analyze_acoustics(); write_result_summary();
save(fullfile(cfg.resultsDir,'completed_run.mat'),'cfg','tests');
fprintf('Hybrid MATLAB phase completed. Review figures, CSVs and summary before submission.\n');
end
