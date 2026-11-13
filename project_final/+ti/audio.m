function [y,fs,path] = audio(cfg,row,targetRate)
% Locate original recording by dataset identity, not a machine-specific path.
if nargin<3, targetRate=cfg.sampleRate; end
name=sprintf('%d.%d.wav',row.instrument_id,row.recording_number);
paths={fullfile(cfg.audioRoot,name),fullfile(cfg.audioRoot,num2str(row.instrument_id),name)};
path='';
for k=1:numel(paths)
    if isfile(paths{k}), path=paths{k}; break; end
end
assert(~isempty(path),'Missing %s below %s. Use setup_project with your Musics folder.',name,cfg.audioRoot);
info=audioinfo(path); start=round(row.start_seconds*info.SampleRate);
n=round(row.duration_seconds*info.SampleRate);
assert(start+n<=info.TotalSamples, 'Audio is too short for %s.',row.segment_id);
[y,fs]=audioread(path,[start+1,start+n]); y=mean(y,2);
if fs~=targetRate, y=resample(y,targetRate,fs); end
fs=targetRate; expected=round(row.duration_seconds*fs);
if numel(y)<expected, y(end+1:expected)=0; end
y=y(1:expected);
assert(all(isfinite(y)),'Nonfinite audio.');
end
