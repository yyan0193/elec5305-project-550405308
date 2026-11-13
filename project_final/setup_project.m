function cfg = setup_project(audioRoot, pythonExecutable)
% SETUP_PROJECT Add MATLAB code to path and check dependencies. No experiments run.
root = fileparts(mfilename('fullpath'));
addpath(root);
cfg = ti.config();
if nargin >= 1 && strlength(string(audioRoot)) > 0
    cfg.audioRoot = char(audioRoot);
end
if nargin >= 2 && strlength(string(pythonExecutable)) > 0
    cfg.python = char(pythonExecutable);
end
required = {'mfcc','resample','fitclinear','fitcknn'};
for k = 1:numel(required)
    assert(exist(required{k},'file') ~= 0, 'Missing %s. Install Audio, Signal Processing, and Statistics and Machine Learning Toolboxes.', required{k});
end
folders = {cfg.cacheDir,cfg.resultsDir,cfg.figuresDir,cfg.modelsDir};
for k = 1:numel(folders)
    if ~isfolder(folders{k}), mkdir(folders{k}); end
end
save(fullfile(root,'local_settings.mat'),'cfg');
fprintf('MATLAB hybrid project configured.\nAudio: %s\nPython: %s\n',cfg.audioRoot,cfg.python);
fprintf('Run check_inputs next. No old Python metrics will be reused.\n');
end
