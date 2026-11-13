function prepare_mert()
% Execute the Python-only MERT component in a separate process.
cfg=ti.config(); script=fullfile(cfg.root,'python','mert_bridge.py');
args={'extract','--manifest',cfg.manifest,'--audio-root',cfg.audioRoot, ...
    '--output',fullfile(cfg.cacheDir,'mert_embeddings.mat'),'--device','cpu'};
parts=[{cfg.python,script},args];
assert(~any(cellfun(@(x)contains(x,'"')||contains(x,newline),parts)),'Paths cannot contain quotes or newlines.');
command=strjoin(cellfun(@(x)['"',x,'"'],parts,'UniformOutput',false),' ');
[status,out]=system(command,'-echo');
assert(status==0,'MERT extraction failed. Check your Python environment.\n%s',out);
end
