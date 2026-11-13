function cfg = config()
% Fixed settings. Local paths may be overridden by setup_project.
cfg.root = fileparts(fileparts(mfilename('fullpath')));
cfg.projectRoot = fileparts(cfg.root);
cfg.audioRoot = fullfile(cfg.projectRoot,'data','raw','ChMusic','Musics');
cfg.python = 'python';
cfg.manifest = fullfile(cfg.root,'metadata','data_manifest.csv');
cfg.cacheDir = fullfile(cfg.root,'data');
cfg.resultsDir = fullfile(cfg.root,'results');
cfg.figuresDir = fullfile(cfg.root,'figures');
cfg.modelsDir = fullfile(cfg.root,'models');
cfg.sampleRate = 24000;
cfg.segmentSeconds = 5;
cfg.fftLength = 2048;
cfg.hopLength = 512;
cfg.numMFCC = 20;
cfg.C = 1;
cfg.seed = 550405308;
cfg.names = ["Erhu","Pipa","Sanxian","Dizi","Suona","Zhuiqin", ...
    "Zhongruan","Liuqin","Guzheng","Yangqin","Sheng"];
cfg.labels = (1:11)';
settings = fullfile(cfg.root,'local_settings.mat');
if isfile(settings)
    s = load(settings,'cfg');
    % Only machine-specific settings persist; algorithms stay in this file.
    cfg.audioRoot = s.cfg.audioRoot;
    cfg.python = s.cfg.python;
end
end
