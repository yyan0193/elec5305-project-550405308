function m = manifest(cfg)
m = readtable(cfg.manifest,'TextType','string');
required = {'segment_id','instrument_id','recording_id','recording_number', ...
    'start_seconds','duration_seconds','fold'};
assert(all(ismember(required,m.Properties.VariableNames)), 'Invalid manifest columns.');
assert(height(m)==988 && numel(unique(m.recording_id))==55, 'Expected frozen 988-segment / 55-recording manifest.');
assert(numel(unique(m.segment_id))==height(m), 'Duplicate segment IDs.');
assert(all(m.duration_seconds==cfg.segmentSeconds), 'Expected five-second clips.');
assert(isequal(sort(unique(m.instrument_id)),cfg.labels), 'Expected all 11 instruments.');
assert(all(m.fold==m.recording_number), 'Fold / recording-number mismatch.');
ids = unique(m.recording_id);
for k=1:numel(ids)
    r=m(m.recording_id==ids(k),:);
    assert(numel(unique(r.fold))==1 && numel(unique(r.instrument_id))==1, 'Source metadata inconsistent.');
    assert(isequal(sort(r.start_seconds),(0:5:5*(height(r)-1))'), 'Noncontiguous or overlapping segments.');
end
for f=1:5
    test=m(m.fold==f,:); train=m(m.fold~=f,:);
    assert(isempty(intersect(test.recording_id,train.recording_id)), 'Source leakage.');
    for c=1:11
        assert(numel(unique(test.recording_id(test.instrument_id==c)))==1, 'Invalid test fold.');
    end
end
end
